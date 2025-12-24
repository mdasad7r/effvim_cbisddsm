import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CbisDdsmDataset(Dataset):
    def __init__(self, splits_csv: str, split: str, transform=None):
        self.df = pd.read_csv(splits_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split={split}. Check {splits_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["matched_path"]
        y = int(row["label"])

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, y
