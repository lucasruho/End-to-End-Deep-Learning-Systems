import argparse
from pathlib import Path
from ultralytics import YOLO
from utils import load_yaml, save_json, ensure_dir, set_seed

def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg.get("seed", 42))

    weights = cfg["model"]["weights"]
    data_yaml = cfg["data"]["yaml"]
    imgsz = int(cfg["data"]["imgsz"])
    batch = int(cfg["data"]["batch"])
    epochs = int(cfg["train"]["epochs"])
    lr0 = float(cfg["train"]["lr0"])
    weight_decay = float(cfg["train"]["weight_decay"])
    patience = int(cfg["train"]["patience"])
    device = cfg.get("device", "auto")
    outdir = Path(cfg.get("output_dir", "outputs"))
    ensure_dir(outdir)

    model = YOLO(weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        weight_decay=weight_decay,
        patience=patience,
        device=device,
        project=str(outdir),
        name="train",
        seed=cfg.get("seed", 42),
        verbose=True
    )

    best = outdir / "train" / "weights" / "best.pt"
    model = YOLO(str(best)) if best.exists() else model
    val = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=True)

    metrics = {}
    try:
        mbox = getattr(val, "box", getattr(val, "boxes", None))
        if mbox is not None:
            metrics.update({
                "map50": getattr(mbox, "map50", None),
                "map": getattr(mbox, "map", None),
                "precision": getattr(mbox, "mp", None),
                "recall": getattr(mbox, "mr", None),
            })
    except Exception:
        pass

    summary = {
        "run_dir": str(outdir / "train"),
        "best_weights": str(best if best.exists() else "unknown"),
        "metrics": metrics
    }
    save_json(summary, outdir / "metrics.json")
    print("Saved summary to", outdir / "metrics.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/yolo_coco128.yaml")
    args = p.parse_args()
    main(args.config)
