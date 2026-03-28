"""
Generate synthetic clinical CSV for multi-modal training.

Each row represents one patient and links to a cytology image via patient_id
(the image filename stem).  Labels are derived deterministically from the
Bethesda class so image + tabular data are self-consistent:

  Bethesda class  ->  CIN grade  ->  progression_label  cancer_label
  Negative            CIN0            0                   0
  ASC-US              CIN0/1          0                   0
  LSIL                CIN1            0                   0
  ASC-H               CIN2 suspect    1                   0
  HSIL                CIN2/3          1                   0
  ca                  carcinoma       1                   1

Output: data/clinical/clinical_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJ     = Path(__file__).resolve().parent.parent
MANIFEST = PROJ / "data" / "sipakmed_split.csv"
OUT_DIR  = PROJ / "data" / "clinical"
OUT_CSV  = OUT_DIR / "clinical_data.csv"
SEED     = 42

# Bethesda -> (cin_grade, progression_label, cancer_label)
CLASS_TO_LABELS = {
    "Negative": (0, 0, 0),
    "ASC-US":   (1, 0, 0),
    "LSIL":     (1, 0, 0),
    "ASC-H":    (2, 1, 0),
    "HSIL":     (3, 1, 0),
    "ca":       (4, 1, 1),
}

HPV_TYPE_MAP = {"none": 0, "other": 1, "18": 2, "16": 3}


def generate(manifest_path: Path, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(manifest_path)

    rows = []
    for _, row in df.iterrows():
        cls                                  = row["class"]
        cin_grade, prog_label, cancer_label  = CLASS_TO_LABELS[cls]
        patient_id = Path(row["filepath"]).stem   # MD5 filename without .png

        # Age: skew older for higher-grade lesions
        age_mean = 25 + cin_grade * 6
        age = int(np.clip(rng.normal(age_mean, 7), 18, 70))

        # HPV status: more likely positive for higher grades
        hpv_prob   = [0.15, 0.35, 0.55, 0.80, 0.90, 0.98][min(cin_grade, 5)]
        hpv_status = int(rng.binomial(1, hpv_prob))

        # HPV type: none if hpv_status=0; high-risk strains more common in HSIL/ca
        if not hpv_status:
            hpv_type = "none"
        elif cin_grade >= 3:
            hpv_type = rng.choice(["16", "18", "other"], p=[0.55, 0.25, 0.20])
        elif cin_grade >= 1:
            hpv_type = rng.choice(["16", "18", "other"], p=[0.30, 0.20, 0.50])
        else:
            hpv_type = "other"

        # Previous CIN stage: 0 for new patients, may be elevated for progression
        if prog_label == 1:
            prev_cin = int(rng.choice([0, 1, 2, 3], p=[0.20, 0.25, 0.35, 0.20]))
        else:
            prev_cin = int(rng.choice([0, 1, 2, 3], p=[0.70, 0.20, 0.08, 0.02]))

        # Persistence: prior abnormal smear that hasn't resolved
        persistence_prob = 0.1 + 0.15 * prog_label + 0.1 * (prev_cin > 0)
        persistence = int(rng.binomial(1, min(persistence_prob, 0.95)))

        rows.append({
            "patient_id":         patient_id,
            "split":              row["split"],
            "bethesda_class":     cls,
            "cin_grade":          cin_grade,
            "age":                age,
            "HPV_status":         hpv_status,
            "HPV_type":           hpv_type,
            "HPV_type_code":      HPV_TYPE_MAP[hpv_type],
            "previous_CIN_stage": prev_cin,
            "persistence":        persistence,
            "progression_label":  prog_label,
            "cancer_label":       cancer_label,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate(MANIFEST)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} rows -> {OUT_CSV}")
    print(f"\nClass distribution:")
    print(df["bethesda_class"].value_counts().to_string())
    print(f"\nProgression label  (1=high risk): {df['progression_label'].sum()} / {len(df)}")
    print(f"Cancer label       (1=cancer):    {df['cancer_label'].sum()} / {len(df)}")
    print(f"\nHPV type breakdown:")
    print(df["HPV_type"].value_counts().to_string())
