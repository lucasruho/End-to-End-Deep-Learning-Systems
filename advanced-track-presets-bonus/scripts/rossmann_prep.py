
# scripts/rossmann_prep.py
# Minimal join/clean to produce data/train.csv with numeric engineered features.
import argparse, pandas as pd, numpy as np
from pathlib import Path

def main(in_dir, out_csv):
    in_dir = Path(in_dir)
    train = pd.read_csv(in_dir / "train.csv")
    store = pd.read_csv(in_dir / "store.csv")
    df = train.merge(store, on="Store", how="left")
    # Basic cleaning
    df = df[df["Open"]==1].copy()
    # Dates
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dow"] = df["Date"].dt.dayofweek
    # Fill NA
    for c in ["Promo2SinceYear","CompetitionOpenSinceYear"]:
        if c in df: df[c] = df[c].fillna(df[c].median())
    # Encode categoricals minimally
    for c in ["StateHoliday","StoreType","Assortment"]:
        if c in df: df[c] = df[c].astype("category").cat.codes
    # Target
    df = df[df["Sales"]>0]
    # Select numeric
    feats = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Sales" not in feats:
        raise SystemExit("Sales column missing after clean")
    feats.remove("Sales")
    out = df[feats + ["Sales"]].rename(columns={"Sales":"Sales"})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Wrote", out_csv, "with", out.shape[0], "rows and", len(feats), "features.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()
    main(args.in_dir, args.out_csv)
