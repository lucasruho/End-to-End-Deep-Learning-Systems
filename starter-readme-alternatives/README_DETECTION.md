
Track B — Object Detection • Dataset Setup
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

Two ready options with **Ultralytics YOLOv8**.

## B1) coco128 (primary, tiny)
No manual download required; Ultralytics pulls it.
```bash
python src/train.py --config configs/yolo_coco128.yaml
python src/evaluate.py --config configs/yolo_coco128.yaml --ckpt outputs/train/weights/best.pt
python src/predict.py --config configs/yolo_coco128.yaml --ckpt outputs/train/weights/best.pt --source ultralytics/assets
```

## B2) coco8 (secondary, minimal)
Swap the dataset yaml to `coco8.yaml` for a super-tiny run:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='coco8.yaml', epochs=3, imgsz=640)
```
To use the config-driven path, duplicate `configs/yolo_coco128.yaml` as `configs/yolo_coco8.yaml` and set:
```yaml
data:
  yaml: coco8.yaml
  imgsz: 640
  batch: 16
```

## B3) Your own YOLO-format dataset
Layout:
```
data/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt   # lines: class x_center y_center width height  (normalized 0..1)
  labels/val/*.txt
data/data.yaml:  # edit names and paths
  path: .
  train: data/images/train
  val: data/images/val
  names: {0: class0, 1: class1}
```
Config:
```yaml
# configs/yolo_custom.yaml
data:
  yaml: data/data.yaml
```
Run:
```bash
python src/train.py --config configs/yolo_custom.yaml
```
mAP@0.5 and mAP@0.5:0.95 are written to `outputs/metrics.json` / `outputs/eval.json`.
