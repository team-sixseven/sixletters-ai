"""445-patient ML table: team clinical 23 variables + 4 Holter variables (27 inputs, 29 columns).

Rebuilds clincal/clinical_clean_4year_holter.csv, the input of 01_extend_clinical_794.py.
The original file was made outside this repository; this script reproduces every value of it.

Steps
1. Team Drive file clincal/clinical_clean.csv (552 patients). Its SCD column means "died of SCD at any
   time"; the 4-year label is rebuilt from MUSIC (SCD = cause of death 3 and follow-up <= 1460 days),
   which changes exactly 3 patients (P0043, P0346, P0539) from 1 to 0. Keep the 445 patients of
   final_split_ecg_445.csv, in the original file's row order (split file order).
2. Add 4 Holter variables from MUSIC subject-info.csv. They were chosen on the train split with
   scripts/data_prep/holter_assoc.py (Mann-Whitney / Fisher, Benjamini-Hochberg FDR q < 0.05, VT dropped
   as a duplicate of NSVT). 999 = missing; "Ventricular Extrasystole" code >= 2 (polymorphic) -> 1.
3. Fill Holter blanks with train-patient values: median for counts, mode for 0/1 variables.

If the output already exists, nothing is written: the rebuilt table is compared with it instead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 경로: 본인 환경에 맞게 바꿀 줄은 MUSIC_DIR 하나. 데이터 폴더(D)는 model/basic.yaml의 drive_root를 따름
MUSIC_DIR = Path("/mnt/c/data/music")  # ← MUSIC 원본 폴더 (subject-info.csv가 있는 곳)
BASIC_YAML = Path(__file__).resolve().parents[2] / "basic.yaml"
D = Path(yaml.safe_load(BASIC_YAML.read_text(encoding="utf-8"))["drive_root"])
SUBJECT_CSV = MUSIC_DIR / "subject-info.csv"
TEAM_CLINICAL = D / "clincal/clinical_clean.csv"  # 팀 Drive 원본 임상 552명
SPLIT_445 = D / "final_split_ecg_445.csv"  # 팀 Drive
OUT = D / "clincal/clinical_clean_4year_holter.csv"

team = pd.read_csv(
    TEAM_CLINICAL, dtype={"Patient ID": str}, float_precision="round_trip"
)  # keep eGFR digits exact
split = pd.read_csv(SPLIT_445, dtype=str)
raw = pd.read_csv(SUBJECT_CSV, sep=";", dtype=str, keep_default_na=False)
raw.columns = [c.strip() for c in raw.columns]
raw["pid"] = raw["Patient ID"].str.strip()
raw = raw.set_index("pid")

# 4-year SCD label (Swee 2026 rule, same as 01_extend_clinical_794.py): SCD within 1,460 days after enrollment.
follow_up = pd.to_numeric(
    raw["Follow-up period from enrollment (days)"].str.strip().str.replace(",", "."),
    errors="coerce",
)
cause = pd.to_numeric(raw["Cause of death"].str.strip().str.replace(",", "."), errors="coerce")
scd_4year = (cause.eq(3) & follow_up.le(1460)).astype(int)
unknown = sorted(set(team["Patient ID"]) - set(scd_4year.index))
if unknown:
    raise ValueError(f"MUSIC 원본에 없는 환자 {len(unknown)}명: {unknown[:5]}")
relabeled = team.loc[
    team["SCD"].to_numpy() != scd_4year.reindex(team["Patient ID"]).to_numpy(), "Patient ID"
]
team["SCD"] = scd_4year.reindex(team["Patient ID"]).to_numpy()
print(f"4년 기준으로 정답을 고친 환자 {len(relabeled)}명: {sorted(relabeled)}")


def num(column):
    value = pd.to_numeric(
        raw[column].str.strip().str.replace(",", ".").str.strip("'"), errors="coerce"
    )
    return value.mask(value == 999)


extrasystole = num("Ventricular Extrasystole")
holter = pd.DataFrame(
    {
        "Holter ventricular premature beats in 24h": num(
            "Number of ventricular premature beats in 24h"
        ),
        "Holter ventricular couplets": num("Extrasystole couplets"),
        "Holter NSVT (yes=1)": num("Non-sustained ventricular tachycardia (CH>10)"),
        "Holter polymorphic ventricular extrasystole (yes=1)": (extrasystole >= 2)
        .astype(float)
        .where(extrasystole.notna()),
    }
)

missing_ids = sorted(set(split.pid) - set(team["Patient ID"]))
if missing_ids:
    raise ValueError(f"팀 임상 파일에 없는 분할 환자 {len(missing_ids)}명: {missing_ids[:5]}")
table = team.set_index("Patient ID").loc[split.pid].join(holter, how="left")
table.index.name = "Patient ID"

train = split.set_index("pid").loc[table.index, "split"].eq("train").to_numpy()
for column in holter.columns:
    values = table[column]
    binary = set(values.dropna().unique()) <= {0.0, 1.0}
    fill = values[train].mode().iloc[0] if binary else values[train].median()
    print(
        f"{column}: 빈칸 {int(values.isna().sum())}명 → train {'최빈값' if binary else '중앙값'} {fill:g}"
    )
    filled = values.fillna(fill)
    table[column] = (
        filled.astype(int) if np.allclose(filled, filled.round()) else filled
    )  # counts / flags as integers
table = table.reset_index()
print(f"결과: {len(table)}명 (SCD {int(table['SCD'].sum())}), {table.shape[1]}열")

if not OUT.exists():
    table.to_csv(OUT, index=False)
    print(f"저장: {OUT}")
    sys.exit(0)

# Output exists: compare instead of overwriting.
original = pd.read_csv(OUT, dtype={"Patient ID": str})
problems = []
if list(original.columns) != list(table.columns):
    problems.append("열 이름 또는 순서가 다름")
elif not original["Patient ID"].equals(table["Patient ID"]):
    problems.append("환자 ID 또는 순서가 다름")
else:
    for column in table.columns[1:]:
        a = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(original[column], errors="coerce").to_numpy(dtype=float)
        n_diff = int((~np.isclose(a, b, equal_nan=True)).sum())
        if n_diff:
            problems.append(f"{column}: {n_diff}명 값이 다름")
if problems:
    print(f"기존 파일과 다름 ({OUT}):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"기존 파일과 모든 값이 같음 ({OUT}). 새로 쓰지 않음")
