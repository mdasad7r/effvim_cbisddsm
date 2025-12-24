import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CSV_NAMES = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
]


def find_first(root: str, filename: str) -> str:
    rootp = Path(root)
    hits = list(rootp.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return str(hits[0])


def extract_uid_folder(p: str) -> str:
    parts = str(p).split("/")
    return parts[-2] if len(parts) >= 2 else ""


def choose_best_jpg(paths):
    if not paths:
        return None
    for p in paths:
        name = os.path.basename(p).lower()
        if ("roi" in name) or ("crop" in name) or ("cropped" in name):
            return p
    return paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Folder that contains csv/ and jpeg/ (Kaggle unzip output)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path, eg data/processed/splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = args.raw_dir
    seed = args.seed

    # Locate CSVs
    csv_paths = {name: find_first(raw_dir, name) for name in CSV_NAMES}

    dfs = []
    for name, p in csv_paths.items():
        df = pd.read_csv(p)
        df["source_csv"] = name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Labels
    df["pathology"] = df["pathology"].astype(str)
    df["label"] = df["pathology"].str.upper().str.contains("MALIGNANT").astype(int)

    # Prefer ROI crops when present
    roi_col = "cropped image file path" if "cropped image file path" in df.columns else None
    img_col = "image file path" if "image file path" in df.columns else None
    if roi_col is None and img_col is None:
        raise KeyError("Could not find 'cropped image file path' or 'image file path' columns in CSVs")

    use_col = roi_col if roi_col is not None else img_col
    df["uid_folder"] = df[use_col].astype(str).apply(extract_uid_folder)

    if "patient_id" not in df.columns:
        raise KeyError("CSV must contain patient_id for patient-level split")

    # Locate all jpgs under jpeg/
    jpeg_root = None
    for cand in ["jpeg", "JPEG", "images", "Images"]:
        cpath = os.path.join(raw_dir, cand)
        if os.path.isdir(cpath):
            jpeg_root = cpath
            break
    if jpeg_root is None:
        jpeg_root = raw_dir  # fallback, scan all

    jpgs = [str(p) for p in Path(jpeg_root).rglob("*.jpg")]
    if not jpgs:
        jpgs = [str(p) for p in Path(jpeg_root).rglob("*.png")]
    if not jpgs:
        raise FileNotFoundError(f"No .jpg/.png found under {jpeg_root}")

    uid_map = defaultdict(list)
    for p in jpgs:
        uid = os.path.basename(os.path.dirname(p))
        uid_map[uid].append(p)

    matched = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="matching"):
        uid = row["uid_folder"]
        cand = uid_map.get(uid, [])
        best = choose_best_jpg(cand)
        matched.append(best)

    df["matched_path"] = matched
    df = df.dropna(subset=["matched_path"]).reset_index(drop=True)

    # Patient-level stratified split 70/15/15
    patient_tbl = df.groupby("patient_id", as_index=False)["label"].max()

    train_pat, temp_pat = train_test_split(
        patient_tbl["patient_id"],
        test_size=0.30,
        stratify=patient_tbl["label"],
        random_state=seed,
    )
    temp_tbl = patient_tbl[patient_tbl["patient_id"].isin(temp_pat)]

    val_pat, test_pat = train_test_split(
        temp_tbl["patient_id"],
        test_size=0.50,
        stratify=temp_tbl["label"],
        random_state=seed,
    )

    def assign_split(pid):
        if pid in set(train_pat):
            return "train"
        if pid in set(val_pat):
            return "val"
        return "test"

    df["split"] = df["patient_id"].apply(assign_split)

    out_csv = args.out_csv
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    keep_cols = ["split", "patient_id", "label", "matched_path", "source_csv"]
    df[keep_cols].to_csv(out_csv, index=False)

    # Print stats
    print("Saved:", out_csv)
    print(df["split"].value_counts())
    print("Label counts:")
    print(df.groupby("split")["label"].value_counts())


if __name__ == "__main__":
    main()
