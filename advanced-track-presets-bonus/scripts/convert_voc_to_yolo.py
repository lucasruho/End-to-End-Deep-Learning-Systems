
# scripts/convert_voc_to_yolo.py
# Convert PASCAL VOC XML annotations to YOLO txt format.
import argparse, os, xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
           "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

def convert_bbox(size, box):
    w, h = size
    x_min, y_min, x_max, y_max = box
    x = (x_min + x_max) / 2.0 / w
    y = (y_min + y_max) / 2.0 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return x, y, bw, bh

def process_split(voc_root: Path, out_root: Path, year_split: str):
    year, split = year_split.split("_")
    imgset = voc_root / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    img_dir = voc_root / f"VOC{year}" / "JPEGImages"
    ann_dir = voc_root / f"VOC{year}" / "Annotations"
    out_img = out_root / "images" / ("train" if split!="test" else "test")
    out_lbl = out_root / "labels" / ("train" if split!="test" else "test")
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)
    for img_id in imgset.read_text().splitlines():
        ann_path = ann_dir / f"{img_id}.xml"
        if not ann_path.exists(): continue
        tree = ET.parse(str(ann_path)); root = tree.getroot()
        w = int(root.find("size").find("width").text)
        h = int(root.find("size").find("height").text)
        ytxt = []
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in CLASSES: continue
            cls_id = CLASSES.index(cls)
            bnd = obj.find("bndbox")
            box = [int(bnd.find(t).text) for t in ["xmin","ymin","xmax","ymax"]]
            x,y,bw,bh = convert_bbox((w,h), box)
            ytxt.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
        (out_lbl / f"{img_id}.txt").write_text("\n".join(ytxt))
        # Copy image lazily by symlink if possible
        src = img_dir / f"{img_id}.jpg"
        dst = out_img / f"{img_id}.jpg"
        try:
            if not dst.exists(): os.symlink(src, dst)
        except Exception:
            if not dst.exists():
                import shutil; shutil.copy(src, dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--splits", nargs="+", default=["2007_trainval","2012_trainval","2007_test"])
    args = ap.parse_args()
    voc_root = Path(args.voc_root); out_root = Path(args.out_root)
    for sp in args.splits:
        process_split(voc_root, out_root, sp)
    print("Done.")
