
Track A — CV Classification
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

## A•Intermediate — Tiny-ImageNet-200
**Why:** 200 classes, 100k images @ 64×64. Harder than CIFAR-10.
**Preset:** `configs/cv_tinyimagenet.yaml` (works with the CV starter).

**Download & prepare (Colab/Kaggle)**
```python
import os, zipfile, urllib.request, shutil, pathlib
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
path = "data/tiny-imagenet-200.zip"
os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve(url, path)
with zipfile.ZipFile(path, "r") as z: z.extractall("data")
# Optional: convert val annotations to class folders for ImageFolder
val_dir = pathlib.Path("data/tiny-imagenet-200/val")
val_map = {}
with open(val_dir / "val_annotations.txt") as f:
    for line in f:
        parts = line.strip().split("\t")
        val_map[parts[0]] = parts[1]
imgs = val_dir / "images"
for img, wnid in val_map.items():
    (val_dir / wnid).mkdir(exist_ok=True)
    src = imgs / img; dst = val_dir / wnid / img
    if src.exists(): shutil.move(src, dst)
```
**Run**
```bash
python src/train.py --config configs/cv_tinyimagenet.yaml
python src/evaluate.py --config configs/cv_tinyimagenet.yaml --ckpt outputs/best.pt
```

## A•Difficult — ImageNet-1k Subset (30–50%)
**Why:** high intra-class variance; real-world difficulty. We train on a curated subset to fit free GPUs.
**Preset:** `configs/cv_imagenet_subset.yaml` (expects ImageFolder under `data/imagenet_subset/` with `train/` and `val/`).

**Prepare (you curate the subset locally)**
```
data/imagenet_subset/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```
Optionally, adapt `scripts/make_imagenet_subset.py` to sample a fixed list of classes and move files.
