# train.py
import argparse
import json
import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.utils.config import load_yaml
from src.utils.seed import set_seed
from src.utils.metrics import compute_binary_metrics_from_logits
from src.utils.checkpoint import save_checkpoint
from src.data.dataset import CbisDdsmDataset
from src.data.transforms import build_transforms
from src.models.eff_mamba import EffMambaClassifier
from src.losses.focal import FocalLossWithLogits


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    labels = labels.astype(int)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, tta_loader=None):
    model.eval()
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(x).squeeze(1)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy().astype(int)

    if tta_loader is not None:
        all_logits_tta = []
        for x, _ in tta_loader:
            x = x.to(device, non_blocking=True)
            lt = model(x).squeeze(1)
            all_logits_tta.append(lt.detach().cpu())
        logits_tta = torch.cat(all_logits_tta).numpy()
        logits = 0.5 * logits + 0.5 * logits_tta

    return compute_binary_metrics_from_logits(y_true, logits)


def _make_amp_tools(use_amp: bool, device: str):
    """
    Returns (scaler, autocast_context).
    Uses torch.amp when available, otherwise falls back to torch.cuda.amp.
    """
    if not use_amp or device != "cuda":
        return None, nullcontext

    # GradScaler
    scaler = None
    try:
        scaler = torch.amp.GradScaler(enabled=True)
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True)

    return scaler, autocast_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    outdir = cfg["output"]["dir"]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(outdir, "train_log.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data transforms
    img_size = int(cfg["data"]["image_size"])
    use_clahe = bool(cfg["data"].get("use_clahe", False))
    grayscale = bool(cfg["data"].get("grayscale", True))
    train_tf, val_tf, tta_tf = build_transforms(img_size, use_clahe)

    splits_csv = cfg["data"]["splits_csv"]

    train_ds = CbisDdsmDataset(splits_csv, "train", transform=train_tf, grayscale=grayscale)
    val_ds = CbisDdsmDataset(splits_csv, "val", transform=val_tf, grayscale=grayscale)
    test_ds = CbisDdsmDataset(splits_csv, "test", transform=val_tf, grayscale=grayscale)

    # Sampler
    sampler_mode = str(cfg["data"].get("sampler", "none")).lower()
    sampler = None
    shuffle = True
    if sampler_mode == "weighted":
        labels = train_ds.df["label"].to_numpy()
        sampler = make_weighted_sampler(labels)
        shuffle = False

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
        num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Optional TTA loaders
    tta_val_loader = None
    tta_test_loader = None
    if bool(cfg.get("eval", {}).get("tta", False)):
        val_tta_ds = CbisDdsmDataset(splits_csv, "val", transform=tta_tf, grayscale=grayscale)
        test_tta_ds = CbisDdsmDataset(splits_csv, "test", transform=tta_tf, grayscale=grayscale)
        tta_val_loader = DataLoader(val_tta_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        tta_test_loader = DataLoader(test_tta_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Model
    model = EffMambaClassifier(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=bool(cfg["model"]["pretrained"]),
        out_index=int(cfg["model"]["out_index"]),
        image_size=img_size,
        mamba_dim=int(cfg["model"]["mamba_dim"]),
        mamba_layers=int(cfg["model"]["mamba_layers"]),
        mamba_d_state=int(cfg["model"]["mamba_d_state"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    # Loss
    loss_name = str(cfg["train"].get("loss", "bce")).lower()
    if loss_name == "focal":
        crit = FocalLossWithLogits(
            alpha=float(cfg["train"].get("focal_alpha", 0.25)),
            gamma=float(cfg["train"].get("focal_gamma", 2.0)),
        )
    else:
        y = train_ds.df["label"].to_numpy().astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train params
    epochs = int(cfg["train"]["epochs"])
    freeze_epochs = int(cfg["train"].get("freeze_epochs", 0))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    lr_head = float(cfg["train"]["lr_head"])
    lr_backbone = float(cfg["train"]["lr_backbone"])
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    accum_steps = int(cfg["train"].get("accum_steps", 1))
    use_amp = bool(cfg["train"].get("amp", True))
    patience = int(cfg["train"].get("early_stop_patience", 8))

    scaler, autocast_ctx = _make_amp_tools(use_amp, device)
    if scaler is None:
        autocast_ctx = nullcontext

    # Stage 1 optimizer
    model.freeze_backbone(True)
    opt = torch.optim.AdamW(model.head_params(), lr=lr_head, weight_decay=wd)

    # Scheduler
    scheduler_name = str(cfg["train"].get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    bad = 0
    best_path = os.path.join(outdir, "best.pt")
    last_path = os.path.join(outdir, "last.pt")

    for epoch in range(1, epochs + 1):
        model.train()

        # Switch to stage 2 after freeze_epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.freeze_backbone(False)
            opt = torch.optim.AdamW(
                [
                    {"params": model.backbone_params(), "lr": lr_backbone},
                    {"params": model.head_params(), "lr": lr_head},
                ],
                weight_decay=wd,
            )
            if scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - epoch + 1))
            elif scheduler_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

        # If backbone is frozen, keep it in eval to stop BN running stats updates
        if freeze_epochs > 0 and epoch <= freeze_epochs:
            model.backbone.eval()

        total_loss = 0.0
        n_seen = 0

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for i, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)

            with (autocast_ctx() if device == "cuda" and use_amp else nullcontext()):
                logits = model(x)
                loss = crit(logits, y) / max(1, accum_steps)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if i % accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                opt.zero_grad(set_to_none=True)

            bs_cur = x.size(0)
            total_loss += float(loss.detach().cpu()) * bs_cur * max(1, accum_steps)
            n_seen += bs_cur
            pbar.set_postfix(loss=total_loss / max(1, n_seen))

        train_loss = total_loss / max(1, n_seen)

        # Metrics
        val_m = evaluate(model, val_loader, device, tta_loader=tta_val_loader)
        test_m = evaluate(model, test_loader, device, tta_loader=tta_test_loader)

        val_auc = float(val_m["auc"]) if not np.isnan(val_m["auc"]) else -1.0

        # Scheduler step
        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_auc)
            else:
                scheduler.step()

        # Save checkpoint every epoch
        epoch_path = os.path.join(outdir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(epoch_path, model, cfg, epoch, best_auc)

        # Save last checkpoint (overwrites each epoch)
        save_checkpoint(last_path, model, cfg, epoch, best_auc)

        # Save best checkpoint when val AUC improves
        if val_auc > best_auc:
            best_auc = val_auc
            bad = 0
            save_checkpoint(best_path, model, cfg, epoch, best_auc)
        else:
            bad += 1

        # Log
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_m,
            "test": test_m,
            "best_val_auc": best_auc,
            "saved": {
                "epoch_ckpt": os.path.basename(epoch_path),
                "best_ckpt": os.path.basename(best_path),
                "last_ckpt": os.path.basename(last_path),
            },
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val auc {val_m['auc']:.4f} acc {val_m['acc']:.4f} sens {val_m['sens']:.4f} spec {val_m['spec']:.4f} | "
            f"test auc {test_m['auc']:.4f} acc {test_m['acc']:.4f} | "
            f"best val auc {best_auc:.4f}"
        )

        if bad >= patience:
            print(f"Early stopping: no val AUC improvement for {patience} epochs.")
            break

    print("Training done.")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)


if __name__ == "__main__":
    main()
