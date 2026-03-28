"""
Multi-Modal, Multi-Task Training Script
========================================
Trains CervicalMultiModal on cytology images + clinical tabular data.

Two simultaneous tasks:
  Task 1 - progression_risk   : BCEWithLogitsLoss
  Task 2 - cancer_probability : BCEWithLogitsLoss (up-weighted — rarer label)

Combined loss = w_prog * loss_prog + w_cancer * loss_cancer

Usage
-----
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --epochs 40 --cancer-weight 3.0
    python scripts/train_multimodal.py --use-colposcopy --colpo-dir data/colposcopy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, confusion_matrix
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from models.multimodal_classifier import CervicalMultiModal

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    manifest       = str(PROJ / "data" / "sipakmed_split.csv"),
    clinical_csv   = str(PROJ / "data" / "clinical" / "clinical_data.csv"),
    out_dir        = str(PROJ / "trained_models"),
    img_size       = 224,
    batch_size     = 16,
    epochs         = 30,
    freeze_epochs  = 5,
    lr             = 3e-4,
    lr_backbone    = 1e-5,
    weight_decay   = 1e-4,
    prog_weight    = 1.0,   # loss weight for progression task
    cancer_weight  = 2.0,   # up-weight cancer task (rarer positive label)
    patience       = 7,     # early stopping patience
    seed           = 42,
    use_colposcopy = False,
    colpo_dir      = None,
)


# ---------------------------------------------------------------------------
# Augmentation pipelines (reuse same approach as train_cervical.py)
# ---------------------------------------------------------------------------

def train_tf(img_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def val_tf(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(int(img_size * 1.1), int(img_size * 1.1)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """
    Joins the image manifest (filepath, class, split) with the clinical CSV
    (patient_id, age, HPV_status, HPV_type_code, previous_CIN_stage, persistence,
     progression_label, cancer_label) on the image filename stem = patient_id.

    Missing tabular rows: filled with zeros (safe default — model handles gracefully).
    """

    def __init__(
        self,
        manifest_path:  str,
        clinical_path:  str,
        split:          str,
        transform:      A.Compose,
        colpo_dir:      str | None = None,
        colpo_transform: A.Compose | None = None,
    ):
        manifest  = pd.read_csv(manifest_path)
        clinical  = pd.read_csv(clinical_path)

        df = manifest[manifest["split"] == split].copy()
        df["patient_id"] = df["filepath"].apply(lambda p: Path(p).stem)

        # Left-join: every image gets a row; missing clinical -> NaN -> fill 0
        df = df.merge(clinical, on="patient_id", how="left")
        df["progression_label"] = df["progression_label"].fillna(0).astype(int)
        df["cancer_label"]      = df["cancer_label"].fillna(0).astype(int)
        df["HPV_status"]        = df["HPV_status"].fillna(0).astype(float)
        df["HPV_type_code"]     = df["HPV_type_code"].fillna(0).astype(int)
        df["previous_CIN_stage"]= df["previous_CIN_stage"].fillna(0).astype(float)
        df["persistence"]       = df["persistence"].fillna(0).astype(float)
        df["age"]               = df["age"].fillna(35).astype(float)  # population mean fallback

        self.df              = df.reset_index(drop=True)
        self.transform       = transform
        self.root            = Path(manifest_path).resolve().parent.parent
        self.colpo_dir       = Path(colpo_dir) if colpo_dir else None
        self.colpo_transform = colpo_transform or val_tf(224)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # --- cytology image ---
        img = Image.open(self.root / row["filepath"]).convert("RGB")
        cyto = self.transform(image=np.array(img))["image"]

        # --- tabular features (normalised) ---
        tabular = {
            "hpv_type_code": torch.tensor(int(row["HPV_type_code"]), dtype=torch.long),
            "age_norm":      torch.tensor(float(row["age"]) / 100.0, dtype=torch.float32),
            "hpv_status":    torch.tensor(float(row["HPV_status"]), dtype=torch.float32),
            "prev_cin_norm": torch.tensor(float(row["previous_CIN_stage"]) / 4.0, dtype=torch.float32),
            "persistence":   torch.tensor(float(row["persistence"]), dtype=torch.float32),
        }

        # --- optional colposcopy image ---
        colpo = None
        if self.colpo_dir is not None:
            colpo_path = self.colpo_dir / (row["patient_id"] + ".png")
            if colpo_path.exists():
                cimg = Image.open(colpo_path).convert("RGB")
                colpo = self.colpo_transform(image=np.array(cimg))["image"]

        # --- labels ---
        labels = {
            "progression": torch.tensor(float(row["progression_label"]), dtype=torch.float32),
            "cancer":      torch.tensor(float(row["cancer_label"]),      dtype=torch.float32),
        }

        return cyto, tabular, colpo, labels

    def pos_weights(self) -> dict:
        """Return positive-class counts for optional BCEWithLogitsLoss pos_weight."""
        n_prog   = self.df["progression_label"].sum()
        n_cancer = self.df["cancer_label"].sum()
        n        = len(self.df)
        return {
            "progression": (n - n_prog) / max(n_prog, 1),
            "cancer":      (n - n_cancer) / max(n_cancer, 1),
        }

    def sampler_weights(self) -> list[float]:
        """WeightedRandomSampler weights — oversample minority classes."""
        # Use combined label: 0=neither, 1=progression only, 2=cancer
        combined = self.df["progression_label"] + self.df["cancer_label"]
        counts   = combined.value_counts().to_dict()
        return [1.0 / counts[v] for v in combined]


# ---------------------------------------------------------------------------
# Collate — handles None colposcopy images in a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    cytos, tabulars, colpos, labels = zip(*batch)

    cyto_batch = torch.stack(cytos)

    # Stack tabular dicts
    tab_batch = {
        k: torch.stack([t[k] for t in tabulars])
        for k in tabulars[0]
    }

    # Colposcopy: stack only if all present, else None
    if all(c is not None for c in colpos):
        colpo_batch = torch.stack(colpos)
    else:
        colpo_batch = None

    label_batch = {
        "progression": torch.stack([l["progression"] for l in labels]),
        "cancer":      torch.stack([l["cancer"]      for l in labels]),
    }

    return cyto_batch, tab_batch, colpo_batch, label_batch


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    binary = (preds >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, binary, average="binary", zero_division=0)
    acc = (binary == labels).mean()
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = float("nan")
    return dict(acc=acc, precision=p, recall=r, f1=f1, auc=auc)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion_prog, criterion_cancer,
              cfg, device, scaler, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_prog_preds, all_prog_labels   = [], []
    all_cancer_preds, all_cancer_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for cyto, tabular, colpo, labels in loader:
            cyto = cyto.to(device)
            tabular = {k: v.to(device) for k, v in tabular.items()}
            if colpo is not None:
                colpo = colpo.to(device)
            prog_lbl   = labels["progression"].to(device)
            cancer_lbl = labels["cancer"].to(device)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out  = model(cyto, tabular, colpo)
                l_prog   = criterion_prog(out["progression_risk"],   prog_lbl)
                l_cancer = criterion_cancer(out["cancer_probability"], cancer_lbl)
                loss = cfg["prog_weight"] * l_prog + cfg["cancer_weight"] * l_cancer

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * len(prog_lbl)

            prog_probs   = torch.sigmoid(out["progression_risk"]).detach().cpu().numpy()
            cancer_probs = torch.sigmoid(out["cancer_probability"]).detach().cpu().numpy()
            all_prog_preds.extend(prog_probs)
            all_prog_labels.extend(prog_lbl.cpu().numpy())
            all_cancer_preds.extend(cancer_probs)
            all_cancer_labels.extend(cancer_lbl.cpu().numpy())

    n    = len(all_prog_labels)
    loss = total_loss / n

    prog_metrics   = compute_metrics(np.array(all_prog_preds),   np.array(all_prog_labels))
    cancer_metrics = compute_metrics(np.array(all_cancer_preds), np.array(all_cancer_labels))

    return loss, prog_metrics, cancer_metrics, \
           np.array(all_prog_preds), np.array(all_prog_labels), \
           np.array(all_cancer_preds), np.array(all_cancer_labels)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_curves(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Combined Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_prog_f1"],   label="train prog F1")
    axes[1].plot(history["val_prog_f1"],     label="val prog F1")
    axes[1].plot(history["train_cancer_f1"], label="train cancer F1", linestyle="--")
    axes[1].plot(history["val_cancer_f1"],   label="val cancer F1",   linestyle="--")
    axes[1].set_title("F1 Scores"); axes[1].legend()

    axes[2].plot(history["val_prog_auc"],   label="prog AUC")
    axes[2].plot(history["val_cancer_auc"], label="cancer AUC")
    axes[2].set_title("Validation AUC"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close()


def plot_confusion(labels, preds, title: str, out_path: Path):
    cm = confusion_matrix(labels, (preds >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"], ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}")

    # --- Generate clinical data if missing ---
    clinical_path = Path(cfg["clinical_csv"])
    if not clinical_path.exists():
        print("Clinical CSV not found — generating synthetic data...")
        from scripts.generate_clinical_data import generate
        clinical_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate(Path(cfg["manifest"]))
        df.to_csv(clinical_path, index=False)
        print(f"  Saved {len(df)} rows -> {clinical_path}")

    # --- Datasets ---
    tr_tf = train_tf(cfg["img_size"])
    v_tf  = val_tf(cfg["img_size"])

    train_ds = MultiModalDataset(cfg["manifest"], cfg["clinical_csv"], "train", tr_tf,
                                 cfg.get("colpo_dir"), v_tf)
    val_ds   = MultiModalDataset(cfg["manifest"], cfg["clinical_csv"], "val",   v_tf,
                                 cfg.get("colpo_dir"), v_tf)
    test_ds  = MultiModalDataset(cfg["manifest"], cfg["clinical_csv"], "test",  v_tf,
                                 cfg.get("colpo_dir"), v_tf)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    pw = train_ds.pos_weights()
    print(f"Positive-class ratios  progression={pw['progression']:.1f}x  "
          f"cancer={pw['cancer']:.1f}x")

    # WeightedRandomSampler for balanced batches
    sampler      = WeightedRandomSampler(train_ds.sampler_weights(), len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- Model ---
    model = CervicalMultiModal(
        pretrained=True, use_colposcopy=cfg["use_colposcopy"]
    ).to(device)

    # --- Loss functions ---
    # BCEWithLogitsLoss — model outputs raw logits, sigmoid applied internally
    criterion_prog   = nn.BCEWithLogitsLoss()
    criterion_cancer = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw["cancer"]], device=device)
    )

    # --- Optimiser: freeze backbone initially ---
    for p in model.image_branch.features.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Training loop ---
    history   = {k: [] for k in [
        "train_loss", "val_loss",
        "train_prog_f1", "val_prog_f1",
        "train_cancer_f1", "val_cancer_f1",
        "val_prog_auc", "val_cancer_auc",
    ]}

    best_val_loss  = float("inf")
    best_path      = out_dir / "multimodal_best.pt"
    patience_count = 0

    print(f"\n{'Ep':>3}  {'TLoss':>7}  {'VLoss':>7}  "
          f"{'ProgF1':>7}  {'CancF1':>7}  {'ProgAUC':>8}  {'CancAUC':>8}  Notes")
    print("-" * 78)

    for epoch in range(1, cfg["epochs"] + 1):
        notes = ""

        # Unfreeze backbone after freeze_epochs
        if epoch == cfg["freeze_epochs"] + 1:
            for p in model.image_branch.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.image_branch.features.parameters(),
                 "lr": cfg["lr_backbone"]},
                {"params": [p for n, p in model.named_parameters()
                            if "image_branch.features" not in n],
                 "lr": cfg["lr"]},
            ], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - epoch
            )
            notes = "unfrozen"

        tr_loss, tr_pm, tr_cm, _, _, _, _ = run_epoch(
            model, train_loader, criterion_prog, criterion_cancer,
            cfg, device, scaler, optimizer
        )
        va_loss, va_pm, va_cm, vp_preds, vp_lbls, vc_preds, vc_lbls = run_epoch(
            model, val_loader, criterion_prog, criterion_cancer,
            cfg, device, scaler, optimizer=None
        )
        scheduler.step()

        # Record history
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_prog_f1"].append(tr_pm["f1"])
        history["val_prog_f1"].append(va_pm["f1"])
        history["train_cancer_f1"].append(tr_cm["f1"])
        history["val_cancer_f1"].append(va_cm["f1"])
        history["val_prog_auc"].append(va_pm["auc"])
        history["val_cancer_auc"].append(va_cm["auc"])

        print(f"{epoch:>3}  {tr_loss:>7.4f}  {va_loss:>7.4f}  "
              f"{va_pm['f1']:>7.3f}  {va_cm['f1']:>7.3f}  "
              f"{va_pm['auc']:>8.3f}  {va_cm['auc']:>8.3f}  {notes}")

        # Save best
        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            patience_count = 0
            model.save(str(best_path))
            print(f"     ** checkpoint saved (val_loss={va_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs)")
                break

    # Plot training curves
    plot_curves(history, out_dir)
    print(f"\nLoss curves -> {out_dir / 'training_curves.png'}")

    # --- Final evaluation on test set ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    best_model = CervicalMultiModal.load(str(best_path), device=str(device)).to(device)
    _, _, _, tp_preds, tp_lbls, tc_preds, tc_lbls = run_epoch(
        best_model, test_loader, criterion_prog, criterion_cancer,
        cfg, device, scaler, optimizer=None
    )

    for task, preds, lbls in [
        ("Progression", tp_preds, tp_lbls),
        ("Cancer",      tc_preds, tc_lbls),
    ]:
        m = compute_metrics(preds, lbls)
        print(f"\n  {task}:")
        print(f"    Accuracy : {m['acc']:.3f}")
        print(f"    Precision: {m['precision']:.3f}")
        print(f"    Recall   : {m['recall']:.3f}")
        print(f"    F1       : {m['f1']:.3f}")
        print(f"    AUC-ROC  : {m['auc']:.3f}")
        plot_confusion(lbls, preds, f"{task} — Test Set Confusion Matrix",
                       out_dir / f"confusion_{task.lower()}.png")

    print(f"\nConfusion matrices -> {out_dir}/confusion_*.png")
    print(f"Best checkpoint   -> {best_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",        default=DEFAULTS["manifest"])
    p.add_argument("--clinical-csv",    default=DEFAULTS["clinical_csv"])
    p.add_argument("--out-dir",         default=DEFAULTS["out_dir"])
    p.add_argument("--img-size",        type=int,   default=DEFAULTS["img_size"])
    p.add_argument("--batch-size",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",          type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--freeze-epochs",   type=int,   default=DEFAULTS["freeze_epochs"])
    p.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-backbone",     type=float, default=DEFAULTS["lr_backbone"])
    p.add_argument("--weight-decay",    type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--prog-weight",     type=float, default=DEFAULTS["prog_weight"])
    p.add_argument("--cancer-weight",   type=float, default=DEFAULTS["cancer_weight"])
    p.add_argument("--patience",        type=int,   default=DEFAULTS["patience"])
    p.add_argument("--seed",            type=int,   default=DEFAULTS["seed"])
    p.add_argument("--use-colposcopy",  action="store_true", default=False)
    p.add_argument("--colpo-dir",       default=None)
    args = p.parse_args()
    main(vars(args))
