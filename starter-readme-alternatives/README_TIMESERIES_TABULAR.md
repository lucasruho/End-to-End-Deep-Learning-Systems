
Track D — Time-series / Tabular Regression • Dataset Setup
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

Two options: one **tabular** and one **time-series**.

## D1) California Housing (tabular, primary)
```python
!pip -q install scikit-learn pandas
from sklearn.datasets import fetch_california_housing
import pandas as pd
b = fetch_california_housing(as_frame=True)
df = b.frame
df.rename(columns={"MedHouseVal":"target"}, inplace=True)
df.to_csv("data/train.csv", index=False)
```
Then:
```bash
python src/train.py --config configs/reg_tabular_mlp.yaml
python src/evaluate.py --config configs/reg_tabular_mlp.yaml --ckpt outputs/best.pt
```

## D2) Synthetic Sine + Trend (time-series, secondary)
Generate a tiny dataset without external downloads:
```python
import numpy as np, pandas as pd
n = 10000; t = np.arange(n)
y = 0.5*np.sin(2*np.pi*t/48) + 0.001*t + 0.1*np.random.randn(n)
df = pd.DataFrame({"timestamp": t, "target": y, "feat1": np.sin(2*np.pi*t/24), "feat2": np.cos(2*np.pi*t/24)})
df.to_csv("data/series.csv", index=False)
```
Set in `configs/reg_timeseries_lstm.yaml`:
```yaml
data:
  csv_path: data/series.csv
  time_col: timestamp
  target_col: target
  seq_len: 48
  horizon: 1
```
Then:
```bash
python src/train.py --config configs/reg_timeseries_lstm.yaml
python src/evaluate.py --config configs/reg_timeseries_lstm.yaml --ckpt outputs/best.pt
```

## D3) Your own CSVs
- **Tabular:** ensure all feature columns are numeric; set `target_col` and optional `feature_cols` in the YAML.  
- **Time-series:** ensure a sortable `time_col`; numeric features; pick `seq_len` and `horizon`. Splits are chronological.
