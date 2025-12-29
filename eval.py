import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score

from src.utils.config import load_yaml
from src.utils.checkpoint import load_checkpoint
from src.data.dataset import CbisDdsmDataset
from src.data.transforms import build_transforms
from src.models.eff_mamba import EffMambaClassifier


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def infer_logits(model, loader, device):
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
    return y_true, logits


@torch.no_grad()
def infer_logits_tta(model, loader, tta_loader, device):
    y_true, logits = infer_logits(model, loader, device)
    y_true2, logits_tta = infer_logits(model, tta_loader, device)

    # safety check: same ordering/length
    if len(y_true) != len(y_true2):
        raise RuntimeError("TTA loader length mismatch. Ensure same split and no shuffle.")

    logits = 0.5 * logits + 0.5 * logits_tta
    return y_true, logits


def best_threshold_youden(y_true: np.ndarray, logits: np.ndarray) -> float:
    # Returns threshold on probability scale (0..1)
    y_true = np.asarray(y_true).astype(int)
    probs = sigmoid(np.asarray(logits))

    # If only one class, ROC curve is undefined
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k])


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, thr: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = sigmoid(np.asarray(logits))
    y_pred = (probs >= thr).astype(int)

    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probs))

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float("nan")
    if len(np.unique(y_true)) == 2:
        f1 = float(f1_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "thr": float(thr),
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sens": sens,
        "spec": spec,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", default="0.5", help="float threshold like 0.5, or 'auto' (Youden on val)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = int(cfg["data"]["image_size"])
    use_clahe = bool(cfg["data"].get("use_clahe", False))
    grayscale = bool(cfg["data"].get("grayscale", True))
    use_tta = bool(cfg.get("eval", {}).get("tta", False))

    _, val_tf, tta_tf = build_transforms(img_size, use_clahe)

    splits_csv = cfg["data"]["splits_csv"]

    # Dataloaders
    val_ds = CbisDdsmDataset(splits_csv, "val", transform=val_tf, grayscale=grayscale)
    test_ds = CbisDdsmDataset(splits_csv, "test", transform=val_tf, grayscale=grayscale)

    bs = int(cfg["data"]["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    val_tta_loader = None
    test_tta_loader = None
    if use_tta:
        val_tta_ds = CbisDdsmDataset(splits_csv, "val", transform=tta_tf, grayscale=grayscale)
        test_tta_ds = CbisDdsmDataset(splits_csv, "test", transform=tta_tf, grayscale=grayscale)
        val_tta_loader = DataLoader(val_tta_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
        test_tta_loader = DataLoader(test_tta_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = EffMambaClassifier(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=False,
        out_index=int(cfg["model"]["out_index"]),
        image_size=img_size,
        mamba_dim=int(cfg["model"]["mamba_dim"]),
        mamba_layers=int(cfg["model"]["mamba_layers"]),
        mamba_d_state=int(cfg["model"]["mamba_d_state"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = load_checkpoint(args.ckpt, device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Inference (val and test)
    if use_tta:
        yv, lv = infer_logits_tta(model, val_loader, val_tta_loader, device)
        yt, lt = infer_logits_tta(model, test_loader, test_tta_loader, device)
    else:
        yv, lv = infer_logits(model, val_loader, device)
        yt, lt = infer_logits(model, test_loader, device)

    # Threshold selection
    thr_arg = str(args.thr).strip().lower()
    if thr_arg == "auto":
        thr = best_threshold_youden(yv, lv)
    else:
        thr = float(thr_arg)

    # Report metrics at chosen thr, and also at 0.5 for comparison
    m_val = compute_metrics(yv, lv, thr=thr)
    m_test = compute_metrics(yt, lt, thr=thr)

    m_val_05 = compute_metrics(yv, lv, thr=0.5)
    m_test_05 = compute_metrics(yt, lt, thr=0.5)

    print(f"Threshold used: {thr:.4f} (arg={args.thr})")
    print("VAL  @thr:", m_val)
    print("TEST @thr:", m_test)
    print("VAL  @0.5:", m_val_05)
    print("TEST @0.5:", m_test_05)


if __name__ == "__main__":
    main()
