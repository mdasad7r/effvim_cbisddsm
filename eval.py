import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CbisDdsmDataset
from src.data.transforms import build_transforms
from src.models.effvim import EffVimClassifier
from src.utils.config import load_yaml
from src.utils.metrics import compute_binary_metrics_from_logits


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
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = int(cfg["data"]["image_size"])
    _, val_tf = build_transforms(img_size)

    ds = CbisDdsmDataset(cfg["data"]["splits_csv"], split="test", transform=val_tf)
    loader = DataLoader(ds, batch_size=int(cfg["data"]["batch_size"]), shuffle=False, num_workers=2)

    model = EffVimClassifier(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=False,
        out_index=int(cfg["model"]["out_index"]),
        mamba_dim=int(cfg["model"]["mamba_dim"]),
        mamba_layers=int(cfg["model"]["mamba_layers"]),
        mamba_d_state=int(cfg["model"]["mamba_d_state"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    m = evaluate(model, loader, device)
    print("TEST METRICS:", m)


if __name__ == "__main__":
    main()
