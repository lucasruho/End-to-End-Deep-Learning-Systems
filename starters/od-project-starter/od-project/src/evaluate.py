import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import load_yaml, save_json

def main(cfg_path, ckpt_path=None):
    cfg = load_yaml(cfg_path)
    data_yaml = cfg["data"]["yaml"]
    imgsz = int(cfg["data"]["imgsz"])
    device = cfg.get("device", "auto")
    outdir = Path(cfg.get("output_dir", "outputs"))

    model = YOLO(ckpt_path) if ckpt_path else YOLO(cfg["model"]["weights"])
    val = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=True)

    metrics = {}
    try:
        mbox = getattr(val, "box", getattr(val, "boxes", None))
        if mbox is not None:
            metrics["map50"] = getattr(mbox, "map50", None)
            metrics["map"] = getattr(mbox, "map", None)
            metrics["precision"] = getattr(mbox, "mp", None)
            metrics["recall"] = getattr(mbox, "mr", None)
    except Exception:
        pass

    save_json(metrics, outdir / "eval.json")
    print("Saved", outdir / "eval.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/yolo_coco128.yaml")
    p.add_argument("--ckpt", type=str, default="outputs/train/weights/best.pt")
    a = p.parse_args()
    main(a.config, a.ckpt if a.ckpt and Path(a.ckpt).exists() else None)
