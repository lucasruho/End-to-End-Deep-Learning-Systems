
Track B — Object Detection
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

## B•Intermediate — PASCAL VOC 2007+2012
**Why:** classic detection benchmark with 20 classes.
**Preset:** `configs/yolo_voc.yaml` + YOLO-format `data_templates/voc_yolo.yaml`.
Ultralytics includes VOC yaml, but we provide a local version for consistency.

**Download (torchvision) & Convert to YOLO**
```python
import torchvision as tv, os, json
from PIL import Image
from pathlib import Path
# Download VOC Annotations & JPEGImages via torchvision or manual download
# Then run converter to YOLO format:
!python scripts/convert_voc_to_yolo.py --voc_root data/VOCdevkit --out_root data/voc_yolo --splits 2007_trainval 2012_trainval 2007_test
```
**Run**
```bash
python src/train.py --config configs/yolo_voc.yaml
```

## B•Difficult — COCO 2017 Subset (30–50%)
**Why:** small/crowded objects, 80 classes, class imbalance.
**Preset:** `configs/yolo_coco2017_subset.yaml` + `data_templates/coco_subset.yaml`.
**Prepare**
```python
# Use scripts/make_coco_subset.py to create a smaller subset in YOLO format
!python scripts/make_coco_subset.py --coco_root data/coco --out_root data/coco50p --fraction 0.5
```
**Run**
```bash
python src/train.py --config configs/yolo_coco2017_subset.yaml
```
