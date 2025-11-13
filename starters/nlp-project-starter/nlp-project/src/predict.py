import argparse, json
from pathlib import Path
import torch
from utils import get_device
from model import LSTMClassifier
from data import Vocab, SimpleTextDataset, collate_pad
from utils import tokenize

@torch.no_grad()
def main(ckpt_path, texts_json):
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)

    itos = ckpt["vocab"]; label_names = ckpt.get("label_names", None)
    stoi = {t:i for i,t in enumerate(itos)}
    vocab = type("V", (), {})()
    vocab.itos = itos; vocab.stoi = stoi; vocab.pad_idx = 0

    def encode(texts):
        seqs = []
        for t in texts:
            ids = [stoi.get(tok, 1) for tok in tokenize(t)][:256]
            seqs.append(ids)
        return seqs

    texts = json.loads(Path(texts_json).read_text())
    seqs = encode(texts)
    batch = [ type("E", (), {"tokens": s, "label": 0, "length": len(s)}) for s in seqs ]
    toks, lens, _ = collate_pad(batch, vocab.pad_idx)

    num_classes = len(label_names) if label_names else 4
    model = LSTMClassifier(vocab_size=len(itos), emb_dim=128, hidden_dim=256, num_layers=1, bidirectional=True, dropout=0.2, num_classes=num_classes, pad_idx=0)
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()

    logits = model(toks.to(device), lens.to(device))
    pred = torch.argmax(logits, dim=1).cpu().tolist()

    out = []
    for i, p in enumerate(pred):
        name = label_names[p] if label_names else str(p)
        out.append({"text": texts[i], "pred": int(p), "label": name})
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="outputs/best.pt")
    p.add_argument("--texts", type=str, default="texts.json")  # a JSON list of strings
    a = p.parse_args()
    main(a.ckpt, a.texts)
