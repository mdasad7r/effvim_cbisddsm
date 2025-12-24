import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CbisDdsmDataset
from src.data.transforms import build_transforms
from src.models.effvim import EffVimClassifier
from src.utils.config import load_yaml
from src.utils.metrics import compute_binary_metrics_from_logits
from src.utils.seed import set_seed


def _ensure_outdir(outdir: str) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)


def _save_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(x).squeeze(1)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    all_logits = torch.cat(all_logits).numpy()
    all_y = torch.cat(all_y).numpy().astype(int)
    return compute_binary_metrics_from_logits(all_y, all_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    outdir = cfg["output"]["dir"]
    _ensure_outdir(outdir)
    log_path = os.path.join(outdir, "train_log.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    img_size = int(cfg["data"]["image_size"])
    train_tf, val_tf = build_transforms(img_size)

    splits_csv = cfg["data"]["splits_csv"]
    train_ds = CbisDdsmDataset(splits_csv, split="train", transform=train_tf)
    val_ds   = CbisDdsmDataset(splits_csv, split="val", transform=val_tf)
    test_ds  = CbisDdsmDataset(splits_csv, split="test", transform=val_tf)

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # pos_weight for weighted loss
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=np.int64)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)

    # Model
    model = EffVimClassifier(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=bool(cfg["model"]["pretrained"]),
        out_index=int(cfg["model"]["out_index"]),
        mamba_dim=int(cfg["model"]["mamba_dim"]),
        mamba_layers=int(cfg["model"]["mamba_layers"]),
        mamba_d_state=int(cfg["model"]["mamba_d_state"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    wd = float(cfg["train"]["weight_decay"])
    lr_head = float(cfg["train"]["lr_head"])
    lr_backbone = float(cfg["train"]["lr_backbone"])
    freeze_epochs = int(cfg["train"]["freeze_epochs"])
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    use_amp = bool(cfg["train"]["amp"])
    grad_clip = float(cfg["train"]["grad_clip"])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Stage 1: freeze backbone
    model.freeze_backbone(True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr_head, weight_decay=wd)

    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_path = os.path.join(outdir, "best.pt")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()

        # Switch to stage 2
        if epoch == freeze_epochs + 1:
            model.freeze_backbone(False)
            opt = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr_backbone},
                    {"params": model.head_parameters(), "lr": lr_head},
                ],
                weight_decay=wd,
            )

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        total_loss = 0.0
        n_seen = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            bs_cur = x.size(0)
            total_loss += float(loss.detach().cpu()) * bs_cur
            n_seen += bs_cur
            pbar.set_postfix(loss=total_loss / max(1, n_seen))

        train_loss = total_loss / max(1, n_seen)

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "test": test_metrics,
        }
        _save_jsonl(log_path, row)

        val_auc = float(val_metrics["auc"])
        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"val auc {val_metrics['auc']:.4f} acc {val_metrics['acc']:.4f} | "
            f"test auc {test_metrics['auc']:.4f} acc {test_metrics['acc']:.4f} | "
            f"best val auc {best_auc:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
