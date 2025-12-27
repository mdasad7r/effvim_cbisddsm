CBIS-DDSM ROI classification using CNN backbone + bidirectional Mamba encoder.

Colab flow:
1) Clone repo
2) Install requirements
3) Download CBIS-DDSM via Kaggle into data/raw/cbis_ddsm
4) Prepare splits: python -m src.data.prepare_cbis_ddsm --raw_dir data/raw/cbis_ddsm --out_csv data/processed/splits.csv
5) Train: python train.py --config configs/cbis_roi.yaml
6) Eval: python eval.py --config configs/cbis_roi.yaml --ckpt outputs/best.pt
