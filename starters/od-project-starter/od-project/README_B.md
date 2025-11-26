Object Detection Starter (YOLOv8 · PyTorch)
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

> *This is the starter code for Track B (CV • Object Detection) of the Deep Learning Fil Rouge project. It uses PyTorch and the Ultralytics YOLOv8 library to build, train, and evaluate an object detection model on the coco128 dataset.*

## 1) Choose your workflow
Pick **one** path; both end up with the same artifacts.

### A. Notebook-first (Colab or Kaggle)
1. Open `notebooks/00_colab_quickstart.ipynb` in Colab/Kaggle.
2. Switch the runtime to **GPU (T4)**  
   - Colab: `Runtime → Change runtime type → GPU → Save`.  
   - Kaggle: gear icon → enable **Accelerator** → select `T4 x1`.
3. Upload or clone the `od-project` folder so the notebook sees `/content/od-project` (Colab) or the working directory (Kaggle).
4. Run the notebook cells in order:
   - Step 0 checks `nvidia-smi` for a GPU.
   - Step 1 switches into the project directory.
   - Step 2 installs everything listed in `requirements.txt` (Ultralytics + friends).
   - Step 3 runs `src/smoke_check.py` (one YOLO prediction on a sample image) to confirm the toolkit works.
5. Decide how far to go:
   - **Smoke test only:** stop after Step 3 if you just need an environment check.
   - **Full run inside the notebook:** execute the “Full training run” cells. They call the same commands as the CLI below, producing weights, metrics, and logs. Once those cells succeed you do **not** need to repeat the CLI steps.

### B. Local CLI
Create and activate a reusable conda environment (Python 3.10 recommended), then install the requirements:
```bash
conda create -n dl_project python=3.10 -y
conda activate dl_project
pip install -r requirements.txt
```
You can also run commands via `conda run -n dl_project <command>` to avoid activating the env.

> **Keeping dependencies fresh:** when `requirements.txt` changes, run  
> `conda run -n dl_project pip install -r requirements.txt`  
> to sync new Ultralytics/Torch releases.

## 2) Train on coco128
```bash
python src/train.py --config configs/yolo_coco128.yaml
```
Ultralytics saves artifacts under `outputs/train/`:
- `weights/best.pt` (best checkpoint) and `weights/last.pt`,
- `results.csv`, `results.png`, and TensorBoard logs in `events/`,
- `metrics.json` summarising the run.

## 3) Evaluate
```bash
python src/evaluate.py --config configs/yolo_coco128.yaml --ckpt outputs/train/weights/best.pt
```
Writes `outputs/eval.json` with validation mAP50/mAP/precision/recall.

## 4) Predict on sample images (optional)
```bash
python src/predict.py --config configs/yolo_coco128.yaml --ckpt outputs/train/weights/best.pt --source ultralytics/assets
```
Annotated results are saved under `outputs/predict/`.

## 5) Switch to your own dataset
- Duplicate `data/data.template.yaml` to `data/data.yaml` and fill in train/val image and label paths.
- Update `configs/yolo_custom.yaml` to point at your new `data.yaml` and adjust epochs/imgsz/batch size.
- Train/evaluate/predict using `configs/yolo_custom.yaml` and the corresponding weight paths.

## 6) Quick FAQ
- **“Do I run both the notebook and the CLI?”** No—pick the workflow that suits you. Notebook users stop after the full-run cells; CLI users can skip the notebook.
- **“Smoke check says it can’t find an image.”** Ensure the repo includes `ultralytics/assets` (it ships with the package). Reinstall requirements if needed.
- **“Evaluation complains about missing `best.pt`.”** Training must complete first. Check `outputs/train/` for errors, then rerun `python src/train.py ...`.
- **“How do I monitor training?”** Open `outputs/train/results.csv` or launch TensorBoard in that directory; Ultralytics logs every epoch.
- **“Dataset in custom format?”** Convert to YOLO format and update `data.yaml`. The workflow (GPU check → install → smoke test → train → evaluate → predict) stays the same.

---
# Appendix: Project Architecture
> The project follows this structure:

```
od-project/
  configs/
    yolo_coco128.yaml         # dataset path, batch_size, lr, epochs, scheduler, seed
  data/                       # small samples or links to larger datasets
    data.template.yaml        # template for custom datasets in YOLO format
  outputs/                    # training/evaluation/prediction artifacts
    train/                    # Ultralytics training outputs
    eval.json                 # evaluation metrics
    predict/                  # prediction results
  src/                       # training, evaluation, prediction scripts
    train.py
    evaluate.py
    predict.py
    smoke_check.py            # quick test: load model, run 1 prediction
  notebooks/
    00_colab_quickstart.ipynb # smoke test + full training run in Colab/Kaggle
  requirements.txt            # Ultralytics YOLOv8 + dependencies
```