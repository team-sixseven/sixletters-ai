"""Univariate association of each ML input variable with 4-year SCD (train split only, reference only).

Same method as holter_assoc.py (which chose the 4 Holter variables): Mann-Whitney U for continuous
variables, Fisher exact for 0/1 variables, Benjamini-Hochberg FDR over all tested variables.
Here it is run on all 27 inputs of clincal/clinical_clean_4year_holter.csv (team clinical 23 + Holter 4).

These q-values were NOT used to choose the 12- or 5-variable sets (those come from literature and
guidelines); they are the "train q" reference column in the variable-selection slides.

    uv run --project Global python Members/mingyu/scripts/data_prep/clinical_assoc.py
"""

from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score

# 데이터 폴더는 model/basic.yaml의 drive_root를 따름
BASIC_YAML = Path(__file__).resolve().parents[2] / "model" / "basic.yaml"
D = Path(yaml.safe_load(BASIC_YAML.read_text(encoding="utf-8"))["drive_root"])
TABLE = (
    D / "clincal/clinical_clean_4year_holter.csv"
)  # model/ML/data_prep/00_build_clinical_445_holter.py
SPLIT = D / "final_split_ecg_445.csv"

# The 12-variable set (model/ML/kfold/settings.py FEATURE_SETS["core"]), marked in the output.
CORE = {
    "Age",
    "Gender (male=1)",
    "NYHA class",
    "LVEF (%)",
    "Left atrial size (mm)",
    "eGFR",
    "Pro-BNP (ng/L)",
    "Ischemic etiology",
    "Prior Myocardial Infarction (yes=1)",
    "Holter ventricular premature beats in 24h",
    "Holter ventricular couplets",
    "Holter NSVT (yes=1)",
}

table = pd.read_csv(TABLE, dtype={"Patient ID": str})
split = pd.read_csv(SPLIT, dtype=str)
d = split.loc[split.split == "train", ["pid"]].merge(
    table.rename(columns={"Patient ID": "pid"}), on="pid", how="left", validate="one_to_one"
)
y = d.SCD.astype(int)

rows = []
for column in [c for c in d.columns if c not in ("pid", "SCD")]:
    values = pd.to_numeric(d[column], errors="coerce")
    ok = values.notna()
    v, t = values[ok], y[ok]
    if v.nunique() == 2:  # 0/1 variable: share of "1" per group, Fisher exact test
        one = v == v.max()
        table2x2 = [
            [(one & (t == 1)).sum(), (~one & (t == 1)).sum()],
            [(one & (t == 0)).sum(), (~one & (t == 0)).sum()],
        ]
        p = fisher_exact(table2x2)[1]
        scd, non_scd = f"{one[t == 1].mean():.0%}", f"{one[t == 0].mean():.0%}"
    else:  # continuous / ordinal: medians, Mann-Whitney U
        p = mannwhitneyu(v[t == 1], v[t == 0]).pvalue
        scd, non_scd = f"{v[t == 1].median():.1f}", f"{v[t == 0].median():.1f}"
    auc = roc_auc_score(t, v)
    rows.append(
        {
            "변수": column.strip(),
            "12개": "●" if column in CORE else "",
            "SCD군": scd,
            "비SCD군": non_scd,
            "방향": "SCD에서 높음" if auc >= 0.5 else "SCD에서 낮음",
            "단변량 AUROC": round(max(auc, 1 - auc), 3),
            "p": p,
        }
    )

out = pd.DataFrame(rows)
out["FDR q"] = false_discovery_control(out.p.to_numpy(), method="bh")
out = out.sort_values("p")
out["p"] = out.p.map(lambda x: f"{x:.4f}")
out["FDR q"] = out["FDR q"].map(lambda x: f"{x:.3f}")
print(f"분석 대상: train {len(d)}명 (SCD {int(y.sum())}명), 변수 {len(out)}개")
print(out.to_string(index=False))
