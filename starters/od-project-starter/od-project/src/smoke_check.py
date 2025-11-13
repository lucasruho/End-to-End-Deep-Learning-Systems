"""
Lightweight sanity check for the YOLO starter.

Loads the pretrained weights listed in the config, runs a single prediction on
the sample assets shipped with Ultralytics, and stores a short metrics summary
under `outputs/smoke_metrics.json`.
"""

from pathlib import Path
from ultralytics import YOLO

from utils import load_yaml, save_json


def run_smoke(cfg_path: str = "configs/yolo_coco128.yaml") -> Path:
    cfg = load_yaml(cfg_path)
    device = cfg.get("device", "auto")
    imgsz = int(cfg["data"]["imgsz"])
    weights = cfg["model"]["weights"]

    model = YOLO(weights)
    results = model.predict(
        source="ultralytics/assets/bus.jpg",
        imgsz=imgsz,
        device=device,
        verbose=False,
        max_det=10,
    )
    pred = results[0]
    boxes = pred.boxes
    num_boxes = int(boxes.xyxy.shape[0]) if boxes is not None else 0
    mean_conf = float(boxes.conf.mean().item()) if num_boxes > 0 else 0.0

    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_metrics.json"
    save_json(
        {
            "num_boxes": num_boxes,
            "mean_confidence": round(mean_conf, 4),
            "imgsz": imgsz,
            "weights": weights,
        },
        out_path,
    )
    return out_path


if __name__ == "__main__":
    path = run_smoke()
    print(f"Smoke check succeeded. Metrics written to {path}.")
