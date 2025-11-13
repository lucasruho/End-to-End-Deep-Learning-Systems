import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import load_yaml

def main(cfg_path, ckpt_path=None, source=None):
    cfg = load_yaml(cfg_path)
    device = cfg.get("device", "auto")
    outdir = Path(cfg.get("output_dir", "outputs")) / "pred"
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(ckpt_path) if ckpt_path else YOLO(cfg["model"]["weights"])
    source = source or "ultralytics/assets"
    model.predict(source=source, save=True, project=str(outdir), name="vis", device=device, imgsz=cfg["data"]["imgsz"])
    print("Saved predictions to", outdir / "vis")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/yolo_coco128.yaml")
    p.add_argument("--ckpt", type=str, default="outputs/train/weights/best.pt")
    p.add_argument("--source", type=str, default="ultralytics/assets")
    a = p.parse_args()
    main(a.config, a.ckpt if Path(a.ckpt).exists() else None, a.source)
