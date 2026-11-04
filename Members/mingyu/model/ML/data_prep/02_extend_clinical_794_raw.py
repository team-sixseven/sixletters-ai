"""Unfilled copy of clinical_clean_4year_holter_max.csv (missing values left blank) plus per-patient flags.

Same 794 patients and values as 01_extend_clinical_794.py; the added 349 rows are rebuilt from MUSIC
subject-info without the train median/mode fill. Values that were present must match the filled file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 경로: 본인 환경에 맞게 바꿀 줄은 MUSIC_DIR 하나. 데이터 폴더(D)는 model/basic.yaml의 drive_root를 따름
MUSIC_DIR = Path("/mnt/c/data/music")  # ← MUSIC 원본 폴더 (subject-info.csv가 있는 곳)
BASIC_YAML = Path(__file__).resolve().parents[2] / "basic.yaml"
D = Path(yaml.safe_load(BASIC_YAML.read_text(encoding="utf-8"))["drive_root"])
SUBJECT_CSV = MUSIC_DIR / "subject-info.csv"
FILLED = D / "clincal/clinical_clean_4year_holter_max.csv"
OUT_RAW = D / "clincal/clinical_clean_4year_holter_max_raw.csv"
OUT_FLAGS = D / "clincal/ml_max_patient_flags.csv"
for path in (OUT_RAW, OUT_FLAGS):
    if path.exists():
        raise FileExistsError(path)

filled = pd.read_csv(FILLED, dtype={"Patient ID": str})
base = filled.iloc[:445]
added_ids = filled["Patient ID"].iloc[445:].to_numpy()

raw = pd.read_csv(SUBJECT_CSV, sep=";", dtype=str, keep_default_na=False)
raw.columns = [c.strip() for c in raw.columns]
raw["pid"] = raw["Patient ID"].str.strip()
raw = raw.set_index("pid")


def num(column, frame):
    return pd.to_numeric(frame[column].str.strip().str.replace(",", "."), errors="coerce")


# Rebuild the added rows exactly as 01_extend_clinical_794.py does, stopping before the fill.
sub = raw.loc[added_ids]
rows = pd.DataFrame({"Patient ID": added_ids, "SCD": filled["SCD"].iloc[445:].to_numpy()})
for column in base.columns[2:]:
    if column.strip() in sub.columns:
        rows[column] = num(column.strip(), sub).to_numpy()
rows["Age"] = np.where(sub["Age"].str.strip().eq(">89").to_numpy(), 90.0, rows["Age"])
age, male = rows["Age"].to_numpy(dtype=float), rows["Gender (male=1)"].to_numpy() == 1
creatinine = num("Creatinine (?mol/L)", sub).to_numpy() / 88.4
rows["eGFR"] = 186 * creatinine**-1.154 * age**-0.203 * np.where(male, 1.0, 0.742)
etiology = num("HF etiology - Diagnosis", sub).to_numpy()
rows["Ischemic etiology"] = np.where(np.isnan(etiology), np.nan, (etiology == 2).astype(float))
holter_available = num("Holter available", sub).eq(1).to_numpy()
holter = {
    "Holter ventricular premature beats in 24h": num(
        "Number of ventricular premature beats in 24h", sub
    ),
    "Holter ventricular couplets": num("Extrasystole couplets", sub),
    "Holter NSVT (yes=1)": num("Non-sustained ventricular tachycardia (CH>10)", sub),
}
ve = num("Ventricular Extrasystole", sub).to_numpy()
holter["Holter polymorphic ventricular extrasystole (yes=1)"] = pd.Series(
    np.where(np.isnan(ve), np.nan, (ve == 2).astype(float))
)
for column, values in holter.items():
    values = np.asarray(values, dtype=float)
    values[(values == 999) | ~holter_available] = np.nan
    rows[column] = values
rows = rows[base.columns]

# Every observed value must equal the filled file; only blanks may differ.
filled_added = filled.iloc[445:].reset_index(drop=True)
for column in base.columns[2:]:
    observed = rows[column].notna().to_numpy()
    if not np.allclose(
        rows[column].to_numpy(dtype=float)[observed],
        filled_added[column].to_numpy(dtype=float)[observed],
    ):
        raise ValueError(f"{column}: observed values differ from {FILLED.name}")
if base.isna().any().any():
    raise ValueError("original 445 rows unexpectedly contain blanks")

out = pd.concat([base, rows], ignore_index=True)
out.to_csv(OUT_RAW, index=False)

device = num("Prior implantable device", raw.loc[out["Patient ID"]]).to_numpy()
flags = pd.DataFrame(
    {
        "patient_id": out["Patient ID"],
        "added": np.r_[np.zeros(445, int), np.ones(len(rows), int)],
        "device": (device != 0).astype(int),
        "holter_available": num("Holter available", raw.loc[out["Patient ID"]])
        .eq(1)
        .astype(int)
        .to_numpy(),
        "any_missing": out[base.columns[2:]].isna().any(axis=1).astype(int).to_numpy(),
    }
)
flags.to_csv(OUT_FLAGS, index=False)

print("wrote", OUT_RAW, out.shape)
print("blanks per column:", {k: int(v) for k, v in out.isna().sum().items() if v})
print("patients with any blank:", int(flags.any_missing.sum()))
print(flags.drop(columns="patient_id").groupby("added").sum())
