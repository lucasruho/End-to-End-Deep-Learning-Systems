NLP Text Classification Starter (PyTorch · LSTM)
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

> *This is the starter code for Track C (NLP • Text Classification) of the Deep Learning Fil Rouge project. It uses PyTorch and the `datasets` library to build, train, and evaluate an LSTM-based text classification model on AG News.*
## 1) Choose your workflow
Pick **one** of the paths below. Both produce the same outputs.

### A. Notebook-first (Colab or Kaggle)
1. Open `notebooks/00_colab_quickstart.ipynb` in Colab/Kaggle.
2. Switch the runtime to **GPU (T4)**  
   - Colab: `Runtime → Change runtime type → GPU → Save`.  
   - Kaggle: gear icon → enable **Accelerator** → select `T4 x1`.
3. Upload or clone the `nlp-project` folder so the notebook sees `/content/nlp-project` (Colab) or the working directory (Kaggle).
4. Run the notebook cells from top to bottom:
   - Step 0 checks `nvidia-smi` to confirm the GPU.
   - Step 1 sets the working directory to the project root.
   - Step 2 installs dependencies straight from `requirements.txt`.
   - Step 3 executes the smoke test (`src/smoke_check.py`) which builds the vocab and runs one training step.
5. Decide how far you want to go:
   - **Smoke test only:** stop after Step 3 if you just need an environment check.
   - **Full run in the notebook:** execute the two “Full training run” cells at the end. They call the same commands as the CLI section, producing `best.pt`, metrics, and evaluation artifacts. Once those cells finish successfully you can skip the CLI instructions below.

### B. Local CLI
Create and activate a reusable conda environment (Python 3.10 recommended), then install the requirements:
```bash
conda create -n dl_project python=3.10 -y
conda activate dl_project
pip install -r requirements.txt
```
If you prefer to keep the environment ephemeral, run commands via `conda run -n dl_project <command>`.

> **Keeping dependencies fresh:** whenever `requirements.txt` changes, run  
> `conda run -n dl_project pip install -r requirements.txt`  
> to install the newest packages (datasets / torchmetrics updates, etc.).

## 2) Train on AG News
```bash
python src/train.py --config configs/nlp_agnews.yaml
```
Artifacts written to `outputs/`:
- `best.pt` (weights + vocabulary snapshot),
- `log.csv` (epoch, loss, accuracy, macro-F1),
- `metrics.json` (best validation macro-F1 and label names).

## 3) Evaluate (val/test)
```bash
python src/evaluate.py --config configs/nlp_agnews.yaml --ckpt outputs/best.pt
```
Produces `outputs/eval.json` with validation metrics and, if configured, test metrics (loss, accuracy, macro-F1).

## 4) Predict on your own texts (optional)
Create `texts.json` as a JSON list of strings:
```json
["Apple unveils new chip", "The team won the match"]
```
Run:
```bash
python src/predict.py --ckpt outputs/best.pt --texts texts.json
```
Predictions are written to `outputs/predictions.json` and printed to the console.

## 5) Bring your own CSV dataset
- Place CSV files under `data/` and edit `configs/nlp_agnews.yaml`:
```yaml
data:
  dataset: csv
  train_csv: data/train.csv
  val_csv: data/val.csv
  test_csv: data/test.csv     # optional
  text_col: text
  label_col: label            # strings or 0..C-1
  delimiter: ","
```
- String labels are auto-mapped to integer IDs in alphabetical order.
- Adjust `data.max_vocab`, `data.min_freq`, and `data.max_len` as needed.

## 6) Quick FAQ
- **“Do I run both the notebook and the CLI?”** No—choose one workflow. Notebook users stop after the full-run cells; CLI users can ignore the notebook.
- **“Smoke test failed with a dataset error.”** Ensure you ran Step 2 (dependencies) and that the GPU runtime has internet access to download AG News. Rerun the smoke cell once fixed.
- **“Evaluation says `best.pt` is missing.”** Training must complete first. Check `outputs/log.csv` for clues, then rerun `python src/train.py ...`.
- **“How do I monitor metrics during training?”** Open `outputs/log.csv` in your notebook or spreadsheet app and plot the loss/accuracy/F1 columns.
- **“Can I switch to another text dataset?”** Yes; update the CSV paths or create a new config. The workflow stays the same: GPU check → install → smoke test → train → evaluate → predict.

---
# Appendix: Project Architecture
> The project follows this structure:

```
nlp-project/
  configs/
    nlp_agnews.yaml           # dataset paths, model params, training hyperparams
  data/                       # sample CSVs or links to larger datasets
  outputs/                    # training/evaluation/prediction artifacts
    best.pt                   # best model checkpoint
    log.csv                   # training log
    metrics.json              # best validation metrics
    eval.json                 # evaluation metrics
    predictions.json          # prediction results
  src/                        # training, evaluation, prediction scripts
    train.py
    evaluate.py
    predict.py
    smoke_check.py            # quick test: load model, run 1 train step
  notebooks/
    00_colab_quickstart.ipynb  # smoke test + full training run in Colab/Kaggle
  requirements.txt            # dependencies: torch, datasets, torchmetrics, etc.
```   