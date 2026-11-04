"""MOMENT split file: the 445-patient split plus augmented copies (-1, -2) of the train SCD patients only.

Input  (drive_root in model/basic.yaml): final_split_ecg_445.csv  (pid, split, SCD; ML·DL common split)
Output (basic.yaml data.raw_ecg.split_path): ECG_tem/final_split_augmented.csv  (pid, SCD, split)

445 original rows keep their split. For each train patient with SCD = 1, two rows are added (pid-1 = first 15 h,
pid-2 = last 15 h, made by preprocessing/02), split = train, SCD = 1: 32 patients x 2 = 64 rows, 509 rows in total.
If the output already exists, it is not overwritten; the script only reports whether it matches.
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from Members.mingyu.model.config import load_basic_config  # noqa: E402

basic = load_basic_config(REPO / "Members/mingyu/model/basic.yaml")
split_445 = Path(basic["drive_root"]) / "final_split_ecg_445.csv"
out_path = Path(basic["data"]["raw_ecg"]["split_path"])

base = pd.read_csv(split_445, dtype=str)[["pid", "SCD", "split"]]
assert len(base) == 445 and base.pid.is_unique, f"{split_445}: 445 patients expected"
positives = base[base.split.eq("train") & base.SCD.eq("1")]
augmented = pd.DataFrame(
    [{"pid": f"{pid}-{k}", "SCD": "1", "split": "train"} for pid in positives.pid for k in (1, 2)]
)
split = pd.concat([base, augmented], ignore_index=True)
print(
    f"원본 {len(base)} {base.split.value_counts().to_dict()} + 증강 {len(augmented)} (train SCD {len(positives)}명 x 2) = {len(split)}줄"
)

if out_path.exists():
    existing = pd.read_csv(out_path, dtype=str)
    print("이미 있음:", out_path, "| 같은 내용:", existing.equals(split))
else:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_path, index=False)
    print("저장:", out_path)
