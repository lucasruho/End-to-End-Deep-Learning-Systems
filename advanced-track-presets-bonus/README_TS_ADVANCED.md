
Track D — Time-series / Tabular Regression
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

## D•Intermediate — Rossmann Store Sales (tabular)
**Why:** calendar effects, promotions, store metadata.
**Preset:** `configs/reg_tabular_rossmann.yaml`.

**Download (Kaggle)**
1) Upload your Kaggle API token (`kaggle.json`) to the notebook:  
   `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`
2) Then:
```bash
pip -q install kaggle
kaggle competitions download -c rossmann-store-sales -p data/
unzip -qo data/rossmann-store-sales.zip -d data/rossmann
```
**Prepare**
```python
# scripts/rossmann_prep.py will join/store metadata, engineer simple features, and write data/train.csv
!python scripts/rossmann_prep.py --in_dir data/rossmann --out_csv data/train.csv
```
Run:
```bash
python src/train.py --config configs/reg_tabular_rossmann.yaml
python src/evaluate.py --config configs/reg_tabular_rossmann.yaml --ckpt outputs/best.pt
```

## D•Difficult — M5 Forecasting (time-series)
**Why:** hierarchical demand, many intermittent series.
**Preset:** `configs/reg_timeseries_m5.yaml`.

**Download (Kaggle)**
```bash
pip -q install kaggle
kaggle competitions download -c m5-forecasting-accuracy -p data/
unzip -qo data/m5-forecasting-accuracy.zip -d data/m5
```
**Prepare a manageable subset**
```bash
!python scripts/m5_prep.py --in_dir data/m5 --out_csv data/series.csv --n_series 200 --start 2013-01-01 --end 2014-12-31
```
Run:
```bash
python src/train.py --config configs/reg_timeseries_m5.yaml
python src/evaluate.py --config configs/reg_timeseries_m5.yaml --ckpt outputs/best.pt
```
