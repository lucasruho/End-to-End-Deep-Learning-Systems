"""Data loading helpers used by the CV starter.

The functions here keep all dataset-specific logic in one place so the rest of
the codebase can stay agnostic to whether we are working with CIFAR-10 or a
custom ImageFolder.  Transforms mirror standard transfer-learning recipes: we
resize to the target input size, add basic augmentation for training, and apply
ImageNet normalization so pretrained weights behave as expected.
"""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet  # Added for Track A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ============================================================
# TRANSFORMS
# ============================================================

def _build_transforms(img_size: int, cfg=None):
    """
    Build torchvision transforms for training and validation.

    If cfg is provided and contains:

        augment:
          stronger: true

    then a stronger training augmentation is applied.
    """

    # ===== Default augmentation =====
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ===== Stronger augmentation if requested =====
    if cfg is not None:
        use_stronger_aug = cfg.get("augment", {}).get("stronger", False)
        if use_stronger_aug:
            train_tf = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),  # Stronger crop
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    # ===== Validation transforms =====
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf



# ============================================================
# DATA LOADER BUILDER
# ============================================================

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, int, list]:
    """
    Create dataloaders and metadata according to the configuration dictionary.

    Returns:
        train_loader, val_loader,
        num_classes,
        classes (list[str])
    """
    img_size = cfg["data"]["img_size"]

    # IMPORTANT → Pass cfg to _build_transforms so augment: stronger=True works
    train_tf, val_tf = _build_transforms(img_size, cfg)

    root = Path(cfg["data"]["root"])
    dataset_name = cfg["data"]["dataset"].lower()

    # ===== CIFAR-10 =====
    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
        val_set   = datasets.CIFAR10(root=str(root), train=False, download=True, transform=val_tf)
        classes = train_set.classes

    # ===== ImageFolder =====
    elif dataset_name == "imagefolder":
        full = datasets.ImageFolder(root=str(root), transform=train_tf)
        val_ratio = float(cfg["data"]["val_split"])
        n_val = max(1, int(len(full) * val_ratio))
        n_train = len(full) - n_val

        train_set, val_set = random_split(full, [n_train, n_val])
        val_set.dataset.transform = val_tf
        classes = full.classes

    # ===== Oxford-IIIT Pet =====
    elif dataset_name == "oxfordpet":
        full_trainval = OxfordIIITPet(
            root=str(root),
            split="trainval",
            target_types="category",
            download=True,
            transform=train_tf,
        )

        val_ratio = float(cfg["data"]["val_split"])
        n_val = max(1, int(len(full_trainval) * val_ratio))
        n_train = len(full_trainval) - n_val

        train_set, val_set = random_split(
            full_trainval,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        val_set.dataset.transform = val_tf
        classes = full_trainval.classes

    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}")

    # ===== Build DataLoaders =====
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    num_classes = len(classes)

    return train_loader, val_loader, num_classes, classes
