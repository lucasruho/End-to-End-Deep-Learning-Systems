
Track A — CV Classification • Dataset Setup
=======================================================
> Author : Badr TAJINI - Machine Learning & Deep learning 2 - ECE 2025/2026

This starter supports **two datasets** out of the box and a generic ImageFolder.

## A1) CIFAR-10 (primary)
Minimal Colab/Kaggle cell:
```python
import torchvision as tv, torch
from torch.utils.data import DataLoader
train_tf = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor()])
val_tf   = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.ToTensor()])
train = tv.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val   = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)
print(len(train), len(val))
```
Then train:
```bash
python src/train.py --config configs/cv_cifar10.yaml
```

## A2) Caltech-101 (secondary)
```python
import torchvision as tv
tf_train = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor()])
tf_val   = tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.ToTensor()])
train = tv.datasets.Caltech101(root="./data", download=True, transform=tf_train)
# Create a small validation split:
val_size = int(0.1*len(train)); train, val = torch.utils.data.random_split(train, [len(train)-val_size, val_size])
val.dataset.transform = tf_val
print(len(train), len(val))
```
Switch config to ImageFolder-like split if you exported folders, else keep CIFAR config but it will still work because loaders are built in `src/data.py`.

## A3) Your own ImageFolder
Directory layout:
```
data/
  class_a/xxx.jpg
  class_b/yyy.jpg
```
Config change (in `configs/cv_cifar10.yaml`):
```yaml
data:
  dataset: imagefolder
  root: ./data
  val_split: 0.1
```
Then:
```bash
python src/train.py --config configs/cv_cifar10.yaml
python src/evaluate.py --config configs/cv_cifar10.yaml --ckpt outputs/best.pt
```
Metrics saved to `outputs/metrics.json`, confusion matrix to `outputs/confusion_matrix.png`.
