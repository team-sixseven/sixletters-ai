"""Univariate association of Holter variables with 4-year SCD (train split only)."""

import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

SUBJECT = "/mnt/c/data/music/subject-info.csv"
SPLIT = "/home/mingyu/sixletters-poc/Members/mingyu/datas/DATA/final_split_ecg_445.csv"
SPLITS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["train"]

s = pd.read_csv(SUBJECT, sep=";", decimal=",", engine="python", dtype={"Patient ID": str})
# "Average RR (ms)" appears twice: 12-lead ECG first, Holter second.
names = [c.strip() for c in s.columns]
s.columns = [n + "__holter" if names[:i].count(n) else n for i, n in enumerate(names)]
s["pid"] = s["Patient ID"].str.strip()
sp = pd.read_csv(SPLIT, dtype=str)
if SPLITS == ["expanded"]:
    # Swee 4-year SCD cohort (PFD excluded) minus our validation/test patients.
    cohort = pd.read_csv(
        "/home/mingyu/poc-test/results/final_holter_ecg/cohort_4year.csv", dtype=str
    )
    cohort = cohort[cohort.SCD_4year_label.notna()]
    held_out = set(sp.loc[sp.split.isin(["val", "test"]), "pid"])
    cohort = cohort[~cohort.pid.isin(held_out)].rename(columns={"SCD_4year_label": "SCD"})
    d = cohort[["pid", "SCD"]].merge(s, on="pid", how="left", validate="one_to_one")
    in_445 = d.pid.isin(sp.pid)
    print(f"확장 대상 = 우리 train {int(in_445.sum())}명 + 445 밖 {int((~in_445).sum())}명")
else:
    d = sp[sp.split.isin(SPLITS)].merge(s, on="pid", how="left", validate="one_to_one")
y = d.SCD.astype(int)


def num(col):
    v = pd.to_numeric(d[col].astype(str).str.replace(",", ".").str.strip("'"), errors="coerce")
    return v.mask(v == 999)


continuous = {
    "최소 RR (ms)": num("minimum RR (ms)"),
    "평균 RR (ms)": num("Average RR (ms)__holter"),
    "최대 RR (ms)": num("maximum RR (ms)"),
    "RR 범위 (ms)": num("RR range (ms)"),
    "24h 심실조기수축(VPB) 수": num("Number of ventricular premature beats in 24h"),
    "시간당 VPC 수": num("Number of ventricular premature contractions per hour"),
    "심실 커플렛 수": num("Extrasystole couplets"),
    "24h 상심실조기수축(SVPB) 수": num("Number of supraventricular premature beats in 24h"),
    "최장 RR pause (ms)": num("Longest RR pause (ms)"),
    "SDNN (ms)": num("SDNN (ms)"),
    "SDANN (ms)": num("SDANN (ms)"),
    "RMSSD (ms)": num("RMSSD (ms)"),
    "pNN50 (%)": num("pNN50 (%)"),
}
rhythm = num("Holter  rhythm")
vt = num("Ventricular Tachycardia")
ve = num("Ventricular Extrasystole")
psvt = num("Paroxysmal supraventricular tachyarrhythmia")
brady = num("Bradycardia")
binary = {
    "Holter 리듬 = 심방세동": (rhythm == 1).where(rhythm.notna()),
    "NSVT (CH>10)": num("Non-sustained ventricular tachycardia (CH>10)")
    .eq(1)
    .where(num("Non-sustained ventricular tachycardia (CH>10)").notna()),
    "심실빈맥(VT) 있음": (vt >= 1).where(vt.notna()),
    "다형성 이상 심실기외수축": (ve >= 2).where(ve.notna()),
    "발작성 상심실빈맥 있음 (만성AF 제외)": psvt.isin([1, 2, 3, 4]).where(
        psvt.notna() & psvt.ne(9)
    ),
    "서맥/전도장애 있음": (brady >= 1).where(brady.notna()),
}

rows = []
for name, v in continuous.items():
    ok = v.notna()
    a, b = v[ok & (y == 1)], v[ok & (y == 0)]
    auc = roc_auc_score(y[ok], v[ok])
    rho = spearmanr(v[ok], y[ok]).statistic
    rows.append(
        {
            "변수": name,
            "유형": "연속",
            "결측": f"{1 - ok.mean():.0%}",
            "SCD군": f"{a.median():.0f}",
            "비SCD군": f"{b.median():.0f}",
            "방향": "SCD에서 높음" if auc >= 0.5 else "SCD에서 낮음",
            "단변량 AUROC": round(max(auc, 1 - auc), 3),
            "Spearman r": round(rho, 3),
            "p": mannwhitneyu(a, b).pvalue,
        }
    )
for name, v in binary.items():
    ok = v.notna()
    vv, yy = v[ok].astype(bool), y[ok]
    table = [
        [(vv & (yy == 1)).sum(), (~vv & (yy == 1)).sum()],
        [(vv & (yy == 0)).sum(), (~vv & (yy == 0)).sum()],
    ]
    odds, p = fisher_exact(table)
    rho = spearmanr(vv.astype(int), yy).statistic
    rows.append(
        {
            "변수": name,
            "유형": "이진",
            "결측": f"{1 - ok.mean():.0%}",
            "SCD군": f"{vv[yy == 1].mean():.0%}",
            "비SCD군": f"{vv[yy == 0].mean():.0%}",
            "방향": f"OR {odds:.2f}",
            "단변량 AUROC": round(roc_auc_score(yy, vv), 3) if vv.nunique() > 1 else np.nan,
            "Spearman r": round(rho, 3),
            "p": p,
        }
    )

out = pd.DataFrame(rows).sort_values("p")
from scipy.stats import false_discovery_control  # noqa: E402

out["FDR q"] = false_discovery_control(out.p.to_numpy(), method="bh")
out["p"] = out.p.map(lambda x: f"{x:.3f}")
out["FDR q"] = out["FDR q"].map(lambda x: f"{x:.2f}")
print(f"분석 대상: {'+'.join(SPLITS)} {len(d)}명 (SCD {int(y.sum())}명)")
print(out.to_string(index=False))
