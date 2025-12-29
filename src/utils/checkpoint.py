import os
import torch


def save_checkpoint(path: str, model, cfg: dict, epoch: int, best_metric: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: str, device: str):
    return torch.load(path, map_location=device)
