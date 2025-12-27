import argparse
import json
import os
from pathlib import Path

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
        for x, _y in tta_loader:
            x = x.to(device, non_blocking=True)
            lt = model(x).squeeze(1)
            all_logits_tta.append(lt.detach().cpu())
        logits_tta = torch.cat(all_logits_tta).numpy()
        logits = 0.5 * logits + 0.5 * logits_tta

    return compute_binary_metrics_from_logits(y_true, logits)


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

    # Data
    img_size = int(cfg["data"]["image_size"])
    use_clahe = bool(cfg["data"]["use_clahe"])
    grayscale = bool(cfg["data"]["grayscale"])

    train_tf, val_tf, tta_tf = build_transforms(img_size, use_clahe)

    splits_csv = cfg["data"]["splits_csv"]

    train_ds = CbisDdsmDataset(splits_csv, "train", transform=train_tf, grayscale=grayscale)
    val_ds   = CbisDdsmDataset(splits_csv, "val", transform=val_tf, grayscale=grayscale)
    test_ds  = CbisDdsmDataset(splits_csv, "test", transform=val_tf, grayscale=grayscale)

    # sampler
    sampler_mode = str(cfg["data"]["sampler"]).lower()
    sampler = None
    shuffle = True
    if sampler_mode == "weighted":
        labels = train_ds.df["label"].to_numpy()
        sampler = make_weighted_sampler(labels)
        shuffle = False

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Optional TTA loaders (same labels)
    tta_val_loader = None
    tta_test_loader = None
    if bool(cfg["eval"]["tta"]):
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
    loss_name = str(cfg["train"]["loss"]).lower()
    if loss_name == "focal":
        crit = FocalLossWithLogits(alpha=float(cfg["train"]["focal_alpha"]), gamma=float(cfg["train"]["focal_gamma"]))
    else:
        # BCE with pos_weight computed from train labels
        y = train_ds.df["label"].to_numpy().astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optim + schedule
    epochs = int(cfg["train"]["epochs"])
    freeze_epochs = int(cfg["train"]["freeze_epochs"])
    wd = float(cfg["train"]["weight_decay"])
    lr_head = float(cfg["train"]["lr_head"])
    lr_backbone = float(cfg["train"]["lr_backbone"])
    grad_clip = float(cfg["train"]["grad_clip"])
    accum_steps = int(cfg["train"]["accum_steps"])
    use_amp = bool(cfg["train"]["amp"])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Stage 1: freeze backbone
    model.freeze_backbone(True)
    opt = torch.optim.AdamW(model.head_params(), lr=lr_head, weight_decay=wd)

    scheduler_name = str(cfg["train"]["scheduler"]).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best_auc = -1.0
    best_path = os.path.join(outdir, "best.pt")
    patience = int(cfg["train"]["early_stop_patience"])
    bad = 0

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()

        # Stage 2: unfreeze backbone and use two LR groups
        if epoch == freeze_epochs + 1:
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

        # When backbone is frozen, keep it in eval to freeze BN running stats
        if epoch <= freeze_epochs:
            model.backbone.eval()

        total_loss = 0.0
        n_seen = 0

        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y) / max(1, accum_steps)

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            bs_cur = x.size(0)
            total_loss += float(loss.detach().cpu()) * bs_cur * max(1, accum_steps)
            n_seen += bs_cur
            pbar.set_postfix(loss=total_loss / max(1, n_seen))

        train_loss = total_loss / max(1, n_seen)

        val_m = evaluate(model, val_loader, device, tta_loader=tta_val_loader)
        # Avoid peeking at test every epoch in serious reporting, but it is useful for debugging
        test_m = evaluate(model, test_loader, device, tta_loader=tta_test_loader)

        row = {"epoch": epoch, "train_loss": train_loss, "val": val_m, "test": test_m}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        val_auc = float(val_m["auc"]) if not np.isnan(val_m["auc"]) else -1.0

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_auc)
            else:
                scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            bad = 0
            save_checkpoint(best_path, model, cfg, epoch, best_auc)
        else:
            bad += 1

        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val auc {val_m['auc']:.4f} acc {val_m['acc']:.4f} sens {val_m['sens']:.4f} | "
            f"test auc {test_m['auc']:.4f} acc {test_m['acc']:.4f} | best val auc {best_auc:.4f}"
        )

        if bad >= patience:
            print(f"Early stopping: no val AUC improvement for {patience} epochs.")
            break

    print("Saved best checkpoint:", best_path)


if __name__ == "__main__":
    main()
