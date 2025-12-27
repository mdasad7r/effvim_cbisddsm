import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset


class CbisDdsmDataset(Dataset):
    def __init__(self, splits_csv: str, split: str, transform=None, grayscale: bool = True):
        self.df = pd.read_csv(splits_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples for split={split}. Check {splits_csv}")
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["matched_path"])
        y = int(row["label"])

        if self.grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(path)
            img = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            out = self.transform(image=img)
            img = out["image"]

        return img, y
