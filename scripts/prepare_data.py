"""
Data preparation script for FemScan AI.

1. Audits + splits cervical cytology images (data/sipakmed/)
2. Generates synthetic endometriosis symptom dataset (data/endo/)
3. Generates synthetic HPV risk factor dataset (data/hpv/)
"""

import os
import shutil
import random
import math
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent  # femscan-ai/
SRC_DIR   = PROJ / "data" / "sipakmed"
SPLIT_DIR = PROJ / "data" / "sipakmed_split"
ENDO_DIR  = PROJ / "data" / "endo"
HPV_DIR   = PROJ / "data" / "hpv"

SEED = 42
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

# ── 1. AUDIT ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 - AUDIT SOURCE DATASET")
print("=" * 60)

class_dirs = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])

# Detect if there are nested CROPPED subfolders
# (classic SIPaKMeD layout: class/CROPPED/*.bmp)
all_images = {}
for cls_dir in class_dirs:
    imgs = []
    # Check for CROPPED subfolder first (real SIPaKMeD)
    cropped = cls_dir / "CROPPED"
    if cropped.exists():
        for f in cropped.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                imgs.append(f)
    # Also collect directly in class folder
    for f in cls_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            imgs.append(f)
    all_images[cls_dir.name] = sorted(imgs)

total = sum(len(v) for v in all_images.values())
print(f"\nSource directory : {SRC_DIR}")
print(f"Classes found    : {list(all_images.keys())}")
print(f"\nImages per class:")
for cls, imgs in all_images.items():
    # Sample one image for resolution info
    res_str = ""
    if imgs:
        try:
            with Image.open(imgs[0]) as im:
                res_str = f"  - sample res: {im.size[0]}x{im.size[1]} {im.mode}"
        except Exception:
            res_str = "  - (could not read)"
    print(f"  {cls:30s}: {len(imgs):5d} images{res_str}")
print(f"\nTotal images: {total}")

# ── ADVISORY ─────────────────────────────────────────────────────────────────
SIPAKMED_CLASSES = {
    "Dyskeratotic", "Koilocytotic", "Metaplastic",
    "Parabasal", "Superficial-Intermediate",
}
found_classes = set(all_images.keys())
if not found_classes & SIPAKMED_CLASSES:
    print("\n[ADVISORY] The classes found do NOT match the standard SIPaKMeD")
    print("  taxonomy (Dyskeratotic / Koilocytotic / Metaplastic / Parabasal /")
    print("  Superficial-Intermediate). The dataset appears to use Bethesda")
    print("  classification (ASC-H, ASC-US, LSIL, ca, Negative).")
    print("  If you intended to use the original SIPaKMeD dataset, please")
    print("  re-download from: https://www.cs.uoi.gr/~marina/sipakmed.html")
    print("  Proceeding with the classes that are present.\n")

if total < 50:
    print(f"[WARNING] Only {total} images found. This is very small for a")
    print("  70/15/15 train/val/test split - some classes will have <5 samples")
    print("  in val/test. Results should be treated as a structural demo only.")

# ── 2. STRATIFIED SPLIT ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 - STRATIFIED SPLIT  (70 / 15 / 15)")
print("=" * 60)

splits_meta = {"train": {}, "val": {}, "test": {}}

for cls, imgs in all_images.items():
    if not imgs:
        print(f"  {cls}: 0 images - skipping")
        continue

    n = len(imgs)
    random.seed(SEED)

    if n < 3:
        # Can't split - put everything in train
        splits_meta["train"][cls] = imgs
        splits_meta["val"][cls]   = []
        splits_meta["test"][cls]  = []
        print(f"  {cls}: only {n} image(s) - all assigned to train")
        continue

    # First split: train vs (val+test)
    train_imgs, temp = train_test_split(
        imgs, test_size=0.30, random_state=SEED
    )
    # Second split: val vs test (50/50 of the 30%)
    if len(temp) < 2:
        val_imgs  = temp
        test_imgs = []
    else:
        val_imgs, test_imgs = train_test_split(
            temp, test_size=0.50, random_state=SEED
        )

    splits_meta["train"][cls] = train_imgs
    splits_meta["val"][cls]   = val_imgs
    splits_meta["test"][cls]  = test_imgs

# ── 3. COPY FILES ─────────────────────────────────────────────────────────────
print("\nCopying files to:", SPLIT_DIR)

for split, cls_map in splits_meta.items():
    for cls, imgs in cls_map.items():
        dest = SPLIT_DIR / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            shutil.copy2(img_path, dest / img_path.name)

# ── 4. VERIFY NO LEAKAGE ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 - VERIFY NO DATA LEAKAGE")
print("=" * 60)

all_split_files = {}
for split in ("train", "val", "test"):
    files = set()
    for cls in splits_meta.get(split, {}):
        dest = SPLIT_DIR / split / cls
        if dest.exists():
            files.update(f.name for f in dest.iterdir() if f.is_file())
    all_split_files[split] = files

train_val  = all_split_files["train"] & all_split_files["val"]
train_test = all_split_files["train"] & all_split_files["test"]
val_test   = all_split_files["val"]   & all_split_files["test"]

leakage = bool(train_val or train_test or val_test)
if leakage:
    print("  [ERROR] Leakage detected!")
    if train_val:  print("  train ∩ val :", train_val)
    if train_test: print("  train ∩ test:", train_test)
    if val_test:   print("  val ∩ test  :", val_test)
else:
    print("  No leakage -- all splits are disjoint. OK")

# ── 5. FINAL STATISTICS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 - FINAL SPLIT STATISTICS")
print("=" * 60)
print(f"\n{'Class':<30} {'Total':>7} {'Train':>7} {'Val':>5} {'Test':>5}")
print("-" * 58)

for cls in all_images:
    n_train = len(splits_meta["train"].get(cls, []))
    n_val   = len(splits_meta["val"].get(cls, []))
    n_test  = len(splits_meta["test"].get(cls, []))
    n_total = n_train + n_val + n_test
    print(f"  {cls:<28} {n_total:>7} {n_train:>7} {n_val:>5} {n_test:>5}")

grand_train = sum(len(v) for v in splits_meta["train"].values())
grand_val   = sum(len(v) for v in splits_meta["val"].values())
grand_test  = sum(len(v) for v in splits_meta["test"].values())
grand_total = grand_train + grand_val + grand_test

print("-" * 58)
print(f"  {'TOTAL':<28} {grand_total:>7} {grand_train:>7} {grand_val:>5} {grand_test:>5}")
print(f"\n  Split ratios - train: {grand_train/grand_total:.1%}  "
      f"val: {grand_val/grand_total:.1%}  test: {grand_test/grand_total:.1%}")
print(f"\n  Split written to: {SPLIT_DIR}")

# ── 6. ENDOMETRIOSIS SYNTHETIC DATASET ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 - ENDOMETRIOSIS SYMPTOM DATASET")
print("=" * 60)

endo_csv = ENDO_DIR / "endo_symptoms.csv"
if endo_csv.exists():
    df_check = pd.read_csv(endo_csv)
    print(f"  Already exists: {endo_csv} ({len(df_check)} rows) - skipping generation.")
else:
    rng = np.random.default_rng(SEED)
    N = 1000

    age             = rng.integers(18, 51, size=N).astype(float)
    bmi             = rng.uniform(18, 35, size=N)
    family_history  = rng.binomial(1, 0.15, size=N)
    pelvic_pain     = rng.integers(0, 11, size=N).astype(float)
    dysmenorrhea    = rng.integers(0, 11, size=N).astype(float)
    pain_intercourse = rng.binomial(1, 0.35, size=N)
    infertility_months = rng.integers(0, 61, size=N).astype(float)
    heavy_periods   = rng.binomial(1, 0.40, size=N)
    bloating        = rng.binomial(1, 0.45, size=N)
    fatigue         = rng.integers(0, 11, size=N).astype(float)
    bowel_pain      = rng.binomial(1, 0.30, size=N)
    urinary_pain    = rng.binomial(1, 0.20, size=N)
    back_pain       = rng.binomial(1, 0.40, size=N)

    # Logistic score -> realistic ~20% prevalence
    log_odds = (
        -3.5
        + 0.30 * pelvic_pain
        + 0.25 * dysmenorrhea
        + 0.80 * pain_intercourse
        + 0.04 * infertility_months
        + 0.50 * heavy_periods
        + 0.40 * bowel_pain
        + 0.20 * fatigue
        + 1.20 * family_history
        + 0.10 * back_pain
        + rng.normal(0, 0.5, size=N)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    threshold = np.percentile(prob, 80)  # keep ~20% positive
    endometriosis = (prob >= threshold).astype(int)

    df_endo = pd.DataFrame({
        "pelvic_pain":          pelvic_pain,
        "dysmenorrhea":         dysmenorrhea,
        "pain_intercourse":     pain_intercourse,
        "infertility_months":   infertility_months,
        "heavy_periods":        heavy_periods,
        "bloating":             bloating,
        "fatigue":              fatigue,
        "bowel_pain":           bowel_pain,
        "urinary_pain":         urinary_pain,
        "back_pain":            back_pain,
        "age":                  age,
        "bmi":                  np.round(bmi, 1),
        "family_history":       family_history,
        "endometriosis":        endometriosis,
    })

    ENDO_DIR.mkdir(parents=True, exist_ok=True)
    df_endo.to_csv(endo_csv, index=False)
    prevalence = endometriosis.mean()
    print(f"  Generated {N} rows  ->  {endo_csv}")
    print(f"  Prevalence: {prevalence:.1%}  (target ~20%)")
    print(f"  Positive cases: {endometriosis.sum()}  /  Negative: {N - endometriosis.sum()}")
    print(f"  Columns: {list(df_endo.columns)}")

# ── 7. HPV RISK FACTOR SYNTHETIC DATASET ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 - HPV RISK FACTOR DATASET")
print("=" * 60)

hpv_csv = HPV_DIR / "hpv_risk_factors.csv"
if hpv_csv.exists():
    df_check = pd.read_csv(hpv_csv)
    print(f"  Already exists: {hpv_csv} ({len(df_check)} rows) - skipping generation.")
else:
    rng2 = np.random.default_rng(SEED)
    N2 = 1500

    age2              = rng2.integers(15, 66, size=N2).astype(float)
    num_partners      = rng2.integers(0, 16, size=N2).astype(float)
    first_intercourse = rng2.integers(13, 31, size=N2).astype(float)
    num_pregnancies   = rng2.integers(0, 11, size=N2).astype(float)
    smoking           = rng2.binomial(1, 0.25, size=N2)
    hormonal_years    = rng2.uniform(0, 20, size=N2)
    iud               = rng2.binomial(1, 0.20, size=N2)
    num_stds          = rng2.integers(0, 6, size=N2).astype(float)
    prev_abnormal_pap = rng2.binomial(1, 0.15, size=N2)
    family_hx_cc      = rng2.binomial(1, 0.08, size=N2)

    log_odds2 = (
        -3.8
        + 0.18 * num_partners
        - 0.10 * (first_intercourse - 13)  # younger = higher risk
        + 0.60 * smoking
        + 0.45 * num_stds
        + 0.70 * prev_abnormal_pap
        + 0.35 * family_hx_cc
        + 0.15 * num_pregnancies
        + rng2.normal(0, 0.5, size=N2)
    )
    prob2 = 1 / (1 + np.exp(-log_odds2))
    threshold2 = np.percentile(prob2, 85)  # keep ~15% positive
    hpv_positive = (prob2 >= threshold2).astype(int)

    df_hpv = pd.DataFrame({
        "age":                            age2,
        "num_sexual_partners":            num_partners,
        "first_intercourse_age":          first_intercourse,
        "num_pregnancies":                num_pregnancies,
        "smoking":                        smoking,
        "hormonal_contraceptive_years":   np.round(hormonal_years, 1),
        "iud":                            iud,
        "num_stds_diagnosed":             num_stds,
        "previous_abnormal_pap":          prev_abnormal_pap,
        "family_history_cervical_cancer": family_hx_cc,
        "hpv_positive":                   hpv_positive,
    })

    HPV_DIR.mkdir(parents=True, exist_ok=True)
    df_hpv.to_csv(hpv_csv, index=False)
    prevalence2 = hpv_positive.mean()
    print(f"  Generated {N2} rows  ->  {hpv_csv}")
    print(f"  Prevalence: {prevalence2:.1%}  (target ~15%)")
    print(f"  Positive cases: {hpv_positive.sum()}  /  Negative: {N2 - hpv_positive.sum()}")
    print(f"  Columns: {list(df_hpv.columns)}")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
