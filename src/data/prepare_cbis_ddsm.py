import argparse
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


CSV_NAMES = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
]


def find_file(root: str, filename: str) -> str:
    hits = list(Path(root).rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return str(hits[0])


def extract_uid_folder(p: str) -> str:
    parts = str(p).split("/")
    return parts[-2] if len(parts) >= 2 else ""


def choose_best_img(paths):
    if not paths:
        return None
    for p in paths:
        name = os.path.basename(p).lower()
        if "roi" in name or "crop" in name or "cropped" in name:
            return p
    return paths[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = args.raw_dir
    seed = args.seed

    # Load CSVs
    dfs = []
    for name in CSV_NAMES:
        p = find_file(raw_dir, name)
        df = pd.read_csv(p)
        df["source_csv"] = name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    df["label"] = df["pathology"].astype(str).str.upper().str.contains("MALIGNANT").astype(int)

    use_col = None
    if "cropped image file path" in df.columns:
        use_col = "cropped image file path"
    elif "image file path" in df.columns:
        use_col = "image file path"
    else:
        raise KeyError("No cropped/image path columns found in CBIS-DDSM CSVs")

    if "patient_id" not in df.columns:
        raise KeyError("patient_id column missing from CSVs")

    df["uid_folder"] = df[use_col].astype(str).apply(extract_uid_folder)

    # Collect images (jpg, png)
    img_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        img_paths += [str(p) for p in Path(raw_dir).rglob(ext)]
    if not img_paths:
        raise FileNotFoundError(f"No images found under {raw_dir}")

    uid_map = defaultdict(list)
    for p in img_paths:
        uid = os.path.basename(os.path.dirname(p))
        uid_map[uid].append(p)

    matched = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="matching"):
        uid = row["uid_folder"]
        matched.append(choose_best_img(uid_map.get(uid, [])))

    df["matched_path"] = matched
    df = df.dropna(subset=["matched_path"]).reset_index(drop=True)

    # Patient-level stratified 70/15/15
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

    train_set = set(train_pat)
    val_set = set(val_pat)

    def assign_split(pid):
        if pid in train_set:
            return "train"
        if pid in val_set:
            return "val"
        return "test"

    df["split"] = df["patient_id"].apply(assign_split)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep = ["split", "patient_id", "label", "matched_path", "source_csv"]
    df[keep].to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print(df["split"].value_counts())
    print(df.groupby("split")["label"].value_counts())


if __name__ == "__main__":
    main()
