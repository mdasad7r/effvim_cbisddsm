import argparse
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_yaml
from src.utils.metrics import compute_binary_metrics_from_logits
from src.utils.checkpoint import load_checkpoint
from src.data.dataset import CbisDdsmDataset
from src.data.transforms import build_transforms
from src.models.eff_mamba import EffMambaClassifier


@torch.no_grad()
def run_eval(model, loader, device, tta_loader=None):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = int(cfg["data"]["image_size"])
    use_clahe = bool(cfg["data"]["use_clahe"])
    grayscale = bool(cfg["data"]["grayscale"])

    _, val_tf, tta_tf = build_transforms(img_size, use_clahe)

    splits_csv = cfg["data"]["splits_csv"]
    ds = CbisDdsmDataset(splits_csv, "test", transform=val_tf, grayscale=grayscale)
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=2)

    tta_loader = None
    if bool(cfg["eval"]["tta"]):
        tds = CbisDdsmDataset(splits_csv, "test", transform=tta_tf, grayscale=grayscale)
        tta_loader = DataLoader(tds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=2)

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

    m = run_eval(model, loader, device, tta_loader=tta_loader)
    print("TEST METRICS:", m)


if __name__ == "__main__":
    main()
