
# scripts/m5_prep.py
# Build a manageable multivariate series CSV from M5 (accuracy) files.
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main(in_dir, out_csv, n_series, start, end):
    in_dir = Path(in_dir)
    cal = pd.read_csv(in_dir / "calendar.csv")
    price = pd.read_csv(in_dir / "sell_prices.csv")
    sales = pd.read_csv(in_dir / "sales_train_validation.csv")
    # Melt sales to long format
    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    value_cols = [c for c in sales.columns if c.startswith("d_")]
    sales_long = sales[id_cols].join(sales[value_cols].melt(var_name="d", value_name="target"))
    # Merge calendar to dates
    cal = cal[["d","date"]]
    sales_long = sales_long.merge(cal, on="d", how="left")
    sales_long["date"] = pd.to_datetime(sales_long["date"])
    # Filter range and sample series
    rng = (sales_long["date"]>=pd.to_datetime(start)) & (sales_long["date"]<=pd.to_datetime(end))
    sales_long = sales_long[rng]
    # Build features: simple weekday and id encodings
    sales_long["dow"] = sales_long["date"].dt.dayofweek
    # Select a subset of series
    unique_ids = sales_long["id"].drop_duplicates().sample(n_series, random_state=42).tolist()
    sub = sales_long[sales_long["id"].isin(unique_ids)].copy()
    # Pivot to per-date rows, aggregate sum across sampled series as an example target
    grp = sub.groupby("date")["target"].sum().reset_index()
    grp = grp.rename(columns={"date":"timestamp"})
    # Add simple cyclical features
    t = np.arange(len(grp))
    grp["sin_dow"] = np.sin(2*np.pi*(t % 7)/7)
    grp["cos_dow"] = np.cos(2*np.pi*(t % 7)/7)
    grp.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "with", grp.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--n_series", type=int, default=200)
    ap.add_argument("--start", type=str, default="2013-01-01")
    ap.add_argument("--end", type=str, default="2014-12-31")
    a = ap.parse_args()
    main(a.in_dir, a.out_csv, a.n_series, a.start, a.end)
