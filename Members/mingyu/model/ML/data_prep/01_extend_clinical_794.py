"""ML-only cohort with every 4-year SCD case in MUSIC; current val/test kept, all additions go to train."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 경로: 본인 환경에 맞게 바꿀 줄은 MUSIC_DIR 하나. 데이터 폴더(D)는 model/basic.yaml의 drive_root를 따름
MUSIC_DIR = Path("/mnt/c/data/music")  # ← MUSIC 원본 폴더 (subject-info.csv가 있는 곳)
BASIC_YAML = Path(__file__).resolve().parents[2] / "basic.yaml"
D = Path(yaml.safe_load(BASIC_YAML.read_text(encoding="utf-8"))["drive_root"])
SUBJECT_CSV = MUSIC_DIR / "subject-info.csv"
BASE = D / "clincal/clinical_clean_4year_holter.csv"
OUT_CLIN = D / "clincal/clinical_clean_4year_holter_max.csv"
OUT_SPLIT = D / "final_split_ml_max.csv"
for path in (OUT_CLIN, OUT_SPLIT):
    if path.exists():
        raise FileExistsError(path)

base_text = BASE.read_text(encoding="utf-8")
base = pd.read_csv(BASE, dtype={"Patient ID": str})
split445 = pd.read_csv(D / "final_split_ecg_445.csv", dtype=str)

raw = pd.read_csv(SUBJECT_CSV, sep=";", dtype=str, keep_default_na=False)
raw.columns = [c.strip() for c in raw.columns]
raw["pid"] = raw["Patient ID"].str.strip()
raw = raw.set_index("pid")


def num(column, frame=raw):
    return pd.to_numeric(frame[column].str.strip().str.replace(",", "."), errors="coerce")


follow_up = num("Follow-up period from enrollment (days)")
cause = num("Cause of death")
exit_code = num("Exit of the study")
scd_4y = cause.eq(3) & follow_up.le(1460)
negative_4y = follow_up.ge(1460) & ~scd_4y & exit_code.ne(2)  # 4-year outcome known, no transplant
eligible = (scd_4y | negative_4y) & ~raw.index.isin(split445.pid)
new_ids = raw.index[eligible]
label = scd_4y.loc[new_ids].astype(int)

sub = raw.loc[new_ids]
rows = pd.DataFrame({"Patient ID": new_ids, "SCD": label.to_numpy()})
for column in base.columns[2:]:
    if column.strip() in sub.columns:
        rows[column] = num(column.strip(), sub).to_numpy()
# MUSIC codes ages of 90 or older as ">89".
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

missing_before = rows.isna().sum()
missing_before = missing_before[missing_before > 0].to_dict()
patients_with_missing = int(rows.isna().any(axis=1).sum())

# Fill from training patients only (current train rows + the new train rows, observed values).
train_ids = set(split445.loc[split445.split == "train", "pid"])
train_pool = pd.concat([base[base["Patient ID"].isin(train_ids)], rows], ignore_index=True)
modal = ["NYHA class", "Mitral valve insufficiency "] + [
    c for c in base.columns[2:] if set(pd.concat([base[c], rows[c]]).dropna().unique()) <= {0, 1}
]
fill = {
    column: (
        train_pool[column].dropna().mode().iloc[0]
        if column in modal
        else train_pool[column].dropna().median()
    )
    for column in missing_before
}
rows = rows.fillna(fill)
assert not rows.isna().any().any()
for column in base.columns:
    if pd.api.types.is_integer_dtype(base[column]):
        rows[column] = rows[column].round().astype(int)

buffer = io.StringIO()
rows.to_csv(buffer, index=False, header=False)
OUT_CLIN.write_text(base_text + buffer.getvalue(), encoding="utf-8")
split = pd.concat(
    [
        split445,
        pd.DataFrame({"pid": rows["Patient ID"], "split": "train", "SCD": rows["SCD"].astype(str)}),
    ],
    ignore_index=True,
)
split.to_csv(OUT_SPLIT, index=False)

device = num("Prior implantable device", sub).to_numpy()
print("added:", len(rows), "SCD", int(rows.SCD.sum()))
print(
    "  device implanted:",
    int((device != 0).sum()),
    "SCD",
    int(rows.SCD[device != 0].sum()),
    "| ICD:",
    int((device == 5).sum()),
    "SCD",
    int(rows.SCD[device == 5].sum()),
)
print("  no Holter:", int((~holter_available).sum()), "SCD", int(rows.SCD[~holter_available].sum()))
print("  ischemic etiology code 6 (set 0 by team rule):", int((etiology == 6).sum()))
print("missing before fill:", missing_before)
print("patients with any missing:", patients_with_missing)
print("fill values:", {k: round(float(v), 3) for k, v in fill.items()})
print(
    split.groupby("split").SCD.apply(lambda s: f"{len(s)} (SCD {int((s == '1').sum())})").to_dict()
)
