"""
Ingests unclean_data/ trials into data/sipakmed/ and rebuilds the train/val/test split.

Structure discovered:
  unclean_data/
    trial_XX/
      images/              <- .png cervical cell images
      fixation_maps/       <- saliency maps (not used for classification)
      fixation_locs/       <- .mat eye-tracking data (not used for classification)
      labels_trial_XX.txt  <- format: index,filename_no_ext,class_label

Steps:
  1. Parse every label file -> build master map {filename -> class}
  2. For each unique image, pick source from whichever trial has it
  3. Copy into data/sipakmed/<class>/
  4. Rebuild stratified 70/15/15 split in data/sipakmed_split/
  5. Verify no leakage + print statistics
"""

import csv
import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

PROJ          = Path(__file__).resolve().parent.parent
UNCLEAN_DIR   = PROJ / "unclean_data"
SIPAKMED_DIR  = PROJ / "data" / "sipakmed"
SPLIT_CSV     = PROJ / "data" / "sipakmed_split.csv"   # manifest instead of copied dirs
SEED          = 42


# ── 1. BUILD MASTER LABEL MAP ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 - PARSE ALL TRIAL LABEL FILES")
print("=" * 60)

trials = sorted([d for d in UNCLEAN_DIR.iterdir() if d.is_dir()])
print(f"Trials found: {[t.name for t in trials]}")

master_labels = {}   # filename.png -> class
master_source = {}   # filename.png -> Path to actual image file

for trial in trials:
    lbl_file = trial / f"labels_{trial.name}.txt"
    img_dir  = trial / "images"
    if not lbl_file.exists():
        print(f"  [WARN] No label file in {trial.name}, skipping")
        continue
    count = 0
    for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        fname = parts[1] + ".png"
        cls   = parts[2]
        img_path = img_dir / fname
        if fname not in master_labels:
            master_labels[fname] = cls
            if img_path.exists():
                master_source[fname] = img_path
        count += 1
    print(f"  {trial.name}: {count} labels parsed, images available: {sum(1 for f in master_labels if (trial/'images'/f).exists())}")

total_unique = len(master_labels)
sourced      = len(master_source)
print(f"\nUnique labeled images  : {total_unique}")
print(f"With image file found  : {sourced}")
if sourced < total_unique:
    missing = set(master_labels) - set(master_source)
    print(f"[WARN] {len(missing)} label entries have no matching image file.")

class_counts = Counter(master_labels.values())
print("\nClass distribution:")
for cls, n in sorted(class_counts.items()):
    print(f"  {cls:30s}: {n:5d} images")

# ── 2. COPY IMAGES INTO data/sipakmed/<class>/ ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 - POPULATE data/sipakmed/ BY CLASS")
print("=" * 60)

# Wipe existing class folders (keep the root dir)
for cls_dir in SIPAKMED_DIR.iterdir():
    if cls_dir.is_dir():
        shutil.rmtree(cls_dir)

copied = Counter()
skipped = 0
for fname, cls in sorted(master_labels.items()):
    src = master_source.get(fname)
    if src is None:
        skipped += 1
        continue
    dest_dir = SIPAKMED_DIR / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / fname)
    copied[cls] += 1

print(f"Copied {sum(copied.values())} images across {len(copied)} classes:")
for cls, n in sorted(copied.items()):
    print(f"  {cls:30s}: {n:5d}")
if skipped:
    print(f"[WARN] Skipped {skipped} images with no source file found.")

# ── 3. SAMPLE RESOLUTION AUDIT ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 - RESOLUTION AUDIT (one sample per class)")
print("=" * 60)
for cls_dir in sorted(SIPAKMED_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    samples = list(cls_dir.glob("*.png"))
    if not samples:
        continue
    try:
        with Image.open(samples[0]) as im:
            print(f"  {cls_dir.name:30s}: {len(samples):5d} imgs  {im.size[0]}x{im.size[1]} {im.mode}")
    except Exception as e:
        print(f"  {cls_dir.name:30s}: {len(samples):5d} imgs  (could not read: {e})")

# ── 4. STRATIFIED 70 / 15 / 15 SPLIT (manifest only - no copy) ───────────────
print("\n" + "=" * 60)
print("STEP 4 - STRATIFIED SPLIT  (70 / 15 / 15,  seed=42)")
print("NOTE: Writing manifest CSV only -- images stay in data/sipakmed/<class>/")
print("=" * 60)

splits_meta = {"train": {}, "val": {}, "test": {}}

for cls_dir in sorted(SIPAKMED_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls = cls_dir.name
    imgs = sorted(cls_dir.glob("*.png"))
    n = len(imgs)

    if n == 0:
        print(f"  {cls}: 0 images - skipping")
        continue

    if n < 3:
        splits_meta["train"][cls] = imgs
        splits_meta["val"][cls]   = []
        splits_meta["test"][cls]  = []
        print(f"  {cls}: only {n} images - all -> train")
        continue

    train_imgs, temp = train_test_split(imgs, test_size=0.30, random_state=SEED)

    if len(temp) < 2:
        val_imgs, test_imgs = temp, []
    else:
        val_imgs, test_imgs = train_test_split(temp, test_size=0.50, random_state=SEED)

    splits_meta["train"][cls] = train_imgs
    splits_meta["val"][cls]   = val_imgs
    splits_meta["test"][cls]  = test_imgs

# Write manifest CSV
rows = []
for split, cls_map in splits_meta.items():
    for cls, imgs in cls_map.items():
        for img_path in imgs:
            rows.append({
                "filepath": str(img_path.relative_to(PROJ)),
                "class":    cls,
                "split":    split,
            })

with open(SPLIT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath", "class", "split"])
    writer.writeheader()
    writer.writerows(rows)

print(f"  Manifest written: {SPLIT_CSV}  ({len(rows)} rows)")

# ── 5. LEAKAGE CHECK ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 - LEAKAGE CHECK")
print("=" * 60)

split_files = {"train": set(), "val": set(), "test": set()}
for row in rows:
    split_files[row["split"]].add(row["filepath"])

tv = split_files["train"] & split_files["val"]
tt = split_files["train"] & split_files["test"]
vt = split_files["val"]   & split_files["test"]

if tv or tt or vt:
    print("  [ERROR] Leakage detected!")
    for label, overlap in [("train/val", tv), ("train/test", tt), ("val/test", vt)]:
        if overlap:
            print(f"  {label}: {list(overlap)[:3]}")
else:
    print("  No leakage -- all splits are disjoint. OK")

# ── 6. FINAL STATISTICS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 - FINAL STATISTICS")
print("=" * 60)
print(f"\n{'Class':<30} {'Total':>7} {'Train':>7} {'Val':>5} {'Test':>5}")
print("-" * 58)

grand = [0, 0, 0, 0]
all_cls = sorted(set(splits_meta["train"]) | set(splits_meta["val"]) | set(splits_meta["test"]))
for cls in all_cls:
    n_tr = len(splits_meta["train"].get(cls, []))
    n_va = len(splits_meta["val"].get(cls, []))
    n_te = len(splits_meta["test"].get(cls, []))
    n_to = n_tr + n_va + n_te
    grand[0] += n_to; grand[1] += n_tr; grand[2] += n_va; grand[3] += n_te
    print(f"  {cls:<28} {n_to:>7} {n_tr:>7} {n_va:>5} {n_te:>5}")

print("-" * 58)
print(f"  {'TOTAL':<28} {grand[0]:>7} {grand[1]:>7} {grand[2]:>5} {grand[3]:>5}")
print(f"\n  Ratios -- train: {grand[1]/grand[0]:.1%}  "
      f"val: {grand[2]/grand[0]:.1%}  test: {grand[3]/grand[0]:.1%}")
print(f"\n  Clean dataset   : {SIPAKMED_DIR}")
print(f"  Split manifest  : {SPLIT_CSV}")
print("  (Training code should read filepath column to locate images)")

print("\n" + "=" * 60)
print("INGEST COMPLETE")
print("=" * 60)
