"""
Training script for CervicalClassifier (EfficientNetV2-S).

Handles class imbalance via TWO complementary strategies:

  Strategy A — Class-weighted loss
    Compute per-class weight = total_samples / (n_classes * class_count).
    Pass as `weight` tensor to nn.CrossEntropyLoss so the model penalises
    mistakes on rare classes (ASC-H, HSIL) more heavily.

  Strategy B — WeightedRandomSampler
    Each training batch is assembled by sampling images with probability
    inversely proportional to their class frequency.  This ensures every
    epoch sees each class represented roughly equally, regardless of how
    few images a class has.

Both strategies are active by default.  You can disable either via flags.

Usage
-----
    python scripts/train_cervical.py
    python scripts/train_cervical.py --epochs 30 --lr 1e-4 --no-weighted-sampler
    python scripts/train_cervical.py --freeze-epochs 5 --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Allow importing from models/
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from models.cervical_classifier import CervicalClassifier, CLASS_NAMES, NUM_CLASSES

# ---------------------------------------------------------------------------
# Config defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    manifest    = str(PROJ / "data" / "sipakmed_split.csv"),
    out_dir     = str(PROJ / "trained_models"),
    img_size    = 224,
    batch_size  = 16,
    epochs      = 25,
    freeze_epochs = 5,   # freeze backbone for this many epochs, then unfreeze
    lr          = 3e-4,
    lr_backbone = 1e-5,  # lower LR for backbone after unfreeze
    weight_decay = 1e-4,
    seed        = 42,
    weighted_sampler = True,
    weighted_loss    = True,
)


# ---------------------------------------------------------------------------
# Albumentations augmentation pipelines
# ---------------------------------------------------------------------------

def build_train_transform(img_size: int) -> A.Compose:
    """
    Augmentation pipeline for training.

    Why these transforms for cervical cytology:
    - HorizontalFlip / VerticalFlip / Rotate: cells have no canonical
      orientation; flipping/rotating are always valid.
    - ColorJitter (HSV shifts): staining variation between labs is common;
      making the model robust to hue/brightness shifts improves generalisation.
    - GaussianBlur / GaussNoise: simulates microscope focus artefacts.
    - RandomResizedCrop: forces the model to handle partial cells and
      varying zoom levels.
    - CoarseDropout: simulates obscured regions (debris, overlapping cells).
    - Normalize: ImageNet stats (matching pretrained EfficientNetV2-S weights).
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1,
            p=0.7,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 6), 
            hole_height_range=(1, 32), 
            hole_width_range=(1, 32), 
            fill_value=0, 
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def build_val_transform(img_size: int) -> A.Compose:
    """Deterministic pipeline for val/test: resize + centre crop + normalise."""
    return A.Compose([
        A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CervicalDataset(Dataset):
    """
    Reads images using the manifest CSV produced by ingest_unclean_data.py.
    CSV columns: filepath (relative to project root), class, split.
    """

    def __init__(self, manifest_path: str, split: str, transform: A.Compose):
        df = pd.read_csv(manifest_path)
        self.df        = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        # root is two levels up from scripts/
        self.root = Path(manifest_path).resolve().parent.parent

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        img   = Image.open(self.root / row["filepath"]).convert("RGB")
        img_np = np.array(img)
        aug   = self.transform(image=img_np)
        label = self.class_to_idx[row["class"]]
        return aug["image"], label

    def class_counts(self) -> dict:
        return self.df["class"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Strategy A: compute class weights for CrossEntropyLoss
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: CervicalDataset, device: torch.device) -> torch.Tensor:
    """
    Returns a weight tensor of shape (NUM_CLASSES,).

    weight[c] = total_samples / (n_classes * count[c])

    Example with our counts:
      LSIL(66) weight ~ 0.37   <- majority, penalised less
      ASC-H(4) weight ~ 6.1    <- rare, penalised heavily when missed

    This is sklearn's 'balanced' class_weight formula.
    """
    counts = dataset.class_counts()
    total  = sum(counts.values())
    weights = []
    for cls in CLASS_NAMES:
        n = counts.get(cls, 1)   # guard against 0
        weights.append(total / (NUM_CLASSES * n))
    w = torch.tensor(weights, dtype=torch.float32).to(device)
    return w


# ---------------------------------------------------------------------------
# Strategy B: WeightedRandomSampler weights (per sample)
# ---------------------------------------------------------------------------

def compute_sample_weights(dataset: CervicalDataset) -> list[float]:
    """
    Returns one weight per sample.  Each sample gets weight = 1/class_count,
    so the sampler draws from each class at equal frequency over an epoch.
    """
    counts = dataset.class_counts()
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    sample_weights = []
    for _, row in dataset.df.iterrows():
        cls = row["class"]
        sample_weights.append(1.0 / counts[cls])
    return sample_weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_tf = build_train_transform(cfg["img_size"])
    val_tf   = build_val_transform(cfg["img_size"])

    train_ds = CervicalDataset(cfg["manifest"], "train", train_tf)
    val_ds   = CervicalDataset(cfg["manifest"], "val",   val_tf)

    print(f"\nTrain: {len(train_ds)} samples")
    print(f"Val  : {len(val_ds)} samples")
    print("\nTrain class distribution:")
    for cls, n in sorted(train_ds.class_counts().items()):
        print(f"  {cls:<30}: {n}")

    # ── Strategy B: WeightedRandomSampler ────────────────────────────────────
    if cfg["weighted_sampler"]:
        print("\n[Strategy B] WeightedRandomSampler ON")
        sample_weights = compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(train_ds),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            sampler=sampler, num_workers=0, pin_memory=True,
        )
    else:
        print("\n[Strategy B] WeightedRandomSampler OFF (shuffle only)")
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True, num_workers=0, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CervicalClassifier(pretrained=True).to(device)

    # ── Strategy A: class-weighted loss ──────────────────────────────────────
    if cfg["weighted_loss"]:
        class_weights = compute_class_weights(train_ds, device)
        print(f"\n[Strategy A] Class-weighted CrossEntropyLoss:")
        for cls, w in zip(CLASS_NAMES, class_weights.tolist()):
            print(f"  {cls:<30}: weight = {w:.3f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\n[Strategy A] Standard CrossEntropyLoss (no class weights)")
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser — two param groups for freeze/unfreeze ─────────────────────
    # Initially freeze backbone; only train classifier head.
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_path    = out_dir / "cervical_best.pt"

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Notes'}")
    print("-" * 75)

    for epoch in range(1, cfg["epochs"] + 1):
        notes = ""

        # Unfreeze backbone after freeze_epochs
        if epoch == cfg["freeze_epochs"] + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Replace optimiser with two param groups
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.classifier.parameters(), "lr": cfg["lr"]},
                {"params": [
                    p for n, p in model.backbone.named_parameters()
                    if "classifier" not in n
                ], "lr": cfg["lr_backbone"]},
            ], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - epoch
            )
            notes = "backbone unfrozen"

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  "
              f"{va_loss:>8.4f}  {va_acc:>7.3f}  {current_lr:>8.2e}  {notes}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.backbone.state_dict(),
                "val_acc":          va_acc,
                "class_names":      CLASS_NAMES,
            }, best_path)
            print(f"         ** saved best checkpoint (val_acc={va_acc:.3f})")

    print(f"\nDone. Best val acc: {best_val_acc:.3f}  ->  {best_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",       default=DEFAULTS["manifest"])
    p.add_argument("--out-dir",        default=DEFAULTS["out_dir"])
    p.add_argument("--img-size",       type=int,   default=DEFAULTS["img_size"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--freeze-epochs",  type=int,   default=DEFAULTS["freeze_epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-backbone",    type=float, default=DEFAULTS["lr_backbone"])
    p.add_argument("--weight-decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    p.add_argument("--no-weighted-sampler", dest="weighted_sampler",
                   action="store_false", default=DEFAULTS["weighted_sampler"])
    p.add_argument("--no-weighted-loss",    dest="weighted_loss",
                   action="store_false", default=DEFAULTS["weighted_loss"])
    args = p.parse_args()
    main(vars(args))
