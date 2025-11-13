Time-series / Tabular Regression Starter (PyTorch)
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

> *This is the starter code for Track D (TS • Time-series / Tabular Regression) of the Deep Learning Fil Rouge project. It uses PyTorch to build, train, and evaluate regression models on tabular and time-series data.*

## 1) Choose your workflow
Pick **one** path; both lead to the same outputs.

### A. Notebook-first (Colab or Kaggle)
1. Open `notebooks/00_colab_quickstart.ipynb` in Colab/Kaggle.
2. Switch the runtime to **GPU (T4)**  
   - Colab: `Runtime → Change runtime type → GPU → Save`.  
   - Kaggle: gear icon → enable **Accelerator** → select `T4 x1`.
3. Upload or clone the `ts-project` folder so the notebook sees `/content/ts-project` (Colab) or the working directory (Kaggle).
4. Run the notebook cells sequentially:
   - Step 0 checks `nvidia-smi` for a GPU.
   - Step 1 sets the working directory.
   - Step 2 installs dependencies directly from `requirements.txt`.
   - Step 3 executes `src/smoke_check.py`, which loads the CSV defined in your config and runs one optimizer step (helpful error if the CSV is missing).
5. Choose your stopping point:
   - **Smoke test only:** stop after Step 3 if you only need to confirm the environment/data wiring.
   - **Full run in the notebook:** run the two “Full training run” cells (train + evaluate). They call the same commands as the CLI below. Once they succeed you can skip the CLI steps.

### B. Local CLI
Create and activate a reusable conda environment (Python 3.10 recommended), then install the requirements:
```bash
conda create -n dl_project python=3.10 -y
conda activate dl_project
pip install -r requirements.txt
```
You can also use `conda run -n dl_project <command>` if you prefer not to activate the env.

> **Keeping dependencies fresh:** whenever `requirements.txt` changes, run  
> `conda run -n dl_project pip install -r requirements.txt`  
> to sync updated packages (pandas, numpy, etc.).

## 2) Tabular regression (MLP)
1. Place a CSV at `data/train.csv` with a numeric column called `target` and numerical feature columns.
2. Edit `configs/reg_tabular_mlp.yaml` to match your column names (or use `feature_cols` to specify them).
3. Run:
```bash
python src/train.py --config configs/reg_tabular_mlp.yaml
python src/evaluate.py --config configs/reg_tabular_mlp.yaml --ckpt outputs/best.pt
```
Artifacts:
- `outputs/best.pt` (weights + scaler metadata),
- `outputs/log.csv` (epoch, loss, MSE/MAE/R²),
- `outputs/metrics.json` (best validation MSE),
- `outputs/eval.json` (val/test metrics).

## 3) Time-series regression (LSTM)
1. Provide a CSV at `data/series.csv` containing at least: `timestamp`, `target`, and numeric feature columns.
2. Edit `configs/reg_timeseries_lstm.yaml` (set `time_col`, `target_col`, `feature_cols`, `seq_len`, `horizon`).
3. Run:
```bash
python src/train.py --config configs/reg_timeseries_lstm.yaml
python src/evaluate.py --config configs/reg_timeseries_lstm.yaml --ckpt outputs/best.pt
```
Evaluation reports the same metrics (MSE/MAE/R²).

## 4) Predict on new data (optional)
```bash
python src/predict.py --ckpt outputs/best.pt --csv path/to/file.csv --features "f1,f2,f3"
# For time-series windows:
python src/predict.py --ckpt outputs/best.pt --csv path/to/series.csv --features "f1,f2" --timeseries --seq_len 48
```
Predictions are written to `outputs/predictions.csv`.

## 5) Quick FAQ
- **“Do I run both the notebook and the CLI?”** No—pick the approach that fits. Notebook users stop after the full-run cells; CLI users can ignore the notebook.
- **“Smoke check fails saying the CSV is missing.”** Create the CSV referenced in the config (even a tiny sample) before running the smoke test or training.
- **“Evaluation can’t find `best.pt`.”** Training must finish successfully first. Check `outputs/log.csv` and rerun `python src/train.py ...`.
- **“My metrics are off the charts.”** Examine feature scaling: the starter stores the `StandardScaler` with the checkpoint—inspect `outputs/metrics.json` to confirm which features were used.
- **“Can I switch between tabular and time-series?”** Yes; swap configs and make sure the corresponding CSV + columns exist. The workflow (GPU check → install → smoke test → train → evaluate) stays the same.

---
# Appendix: Project Architecture
> The project follows this structure:

```
ts-project/
  configs/
    reg_tabular_mlp.yaml      # tabular regression config
    reg_timeseries_lstm.yaml   # time-series regression config
  data/                       # sample CSVs or links to larger datasets
    train.csv                 # tabular regression sample
    series.csv                # time-series regression sample
  outputs/                    # training/evaluation/prediction artifacts
    best.pt                   # best model checkpoint
    log.csv                   # training log
    metrics.json              # best validation metrics
    eval.json                 # evaluation metrics
    predictions.csv           # prediction results
  src/                        # training, evaluation, prediction scripts
    train.py
    evaluate.py
    predict.py
    smoke_check.py            # quick test: load data, run 1 optimizer step
  notebooks/
    00_colab_quickstart.ipynb # smoke test + full training run in Colab/Kaggle
  README.md                   # how-to + results table + ethical note
```