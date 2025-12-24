EfficientNetV2-M + Bidirectional Mamba for CBIS-DDSM ROI classification.

Quickstart (Colab):
1) pip install -r requirements.txt
2) Download dataset into data/raw/cbis_ddsm (Kaggle recommended)
3) python -m src.data.prepare_cbis_ddsm --raw_dir data/raw/cbis_ddsm --out_csv data/processed/splits.csv
4) python train.py --config configs/default.yaml
5) python eval.py --config configs/default.yaml --ckpt outputs/best.pt
