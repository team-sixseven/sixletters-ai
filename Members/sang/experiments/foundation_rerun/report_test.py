"""Add test metrics to the existing three slides and the accompanying report."""

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "Members/sang/results/foundation_rerun_20261103"
PPT = ROOT / "Members/sang/docs/모델 정리.pptx"
DOC = ROOT / "Members/sang/docs/foundation_models_e100_p10_20261103.md"
specs = [
    (14, "ecg_fm", "ECG-FM"),
    (15, "deepecg", "DeepECG"),
    (16, "ecgfounder", "ECGFounder"),
]
prs = Presentation(PPT)
lines = [
    "",
    "## Test 재평가 (기존 사용 이력 있음)",
    "",
    "최적 validation AP epoch의 저장된 가중치로 동일한 test 환자 89명(양성 10명·음성 79명)을 평가했다. 민감도·특이도는 점수 0.5 기준이다. 이 분할은 과거 실험에서도 사용됐으므로 독립적인 최종 검증이 아니다.",
    "",
    "| 모델 | AUROC | AP | 민감도 | 특이도 | TN / FP / FN / TP |",
    "|---|---:|---:|---:|---:|---|",
]
for n, name, title in specs:
    d = BASE / name
    m = json.loads((d / "test_metrics.json").read_text())
    data = json.loads((d / "metrics.json").read_text())
    v = data["splits"]["validation"] if name == "ecg_fm" else data["validation"]
    rows = list(csv.DictReader((d / "validation.csv").open()))
    positives = sum(int(r["scd_4year_label"]) for r in rows)
    vtp = sum(int(r["scd_4year_label"]) and float(r["model_score"]) >= 0.5 for r in rows)
    vtn = sum(not int(r["scd_4year_label"]) and float(r["model_score"]) < 0.5 for r in rows)
    c = m["confusion"]
    lines.append(
        f"| {title} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['sensitivity']:.1%} | {m['specificity']:.1%} | {c['tn']} / {c['fp']} / {c['fn']} / {c['tp']} |"
    )
    slide = prs.slides[n - 1]
    for shape in list(slide.shapes):
        if shape.has_table:
            shape._element.getparent().remove(shape._element)
    tf = slide.shapes[6].text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = "Validation / Test 비교 (기준 0.5)"
    p.font.name = "NanumGothic"
    p.font.size = Pt(15)
    slide.shapes[7].text_frame.clear()
    vals = [
        ("지표", "Validation", "Test"),
        ("환자 수 / 양성", f"{len(rows)} / {positives}", f"{m['n']} / {m['positive']}"),
        (
            "AUROC / AP",
            f"{v['auroc']:.3f} / {v['average_precision']:.3f}",
            f"{m['auroc']:.3f} / {m['average_precision']:.3f}",
        ),
        (
            "민감도",
            f"{vtp}/{positives} ({vtp / positives:.1%})",
            f"{c['tp']}/{m['positive']} ({m['sensitivity']:.1%})",
        ),
        (
            "특이도",
            f"{vtn}/{len(rows) - positives} ({vtn / (len(rows) - positives):.1%})",
            f"{c['tn']}/{m['n'] - m['positive']} ({m['specificity']:.1%})",
        ),
    ]
    table = slide.shapes.add_table(
        len(vals), 3, Inches(0.65), Inches(5.02), Inches(5.8), Inches(1.6)
    ).table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(2.0)
    table.columns[2].width = Inches(2.0)
    for i, row in enumerate(vals):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.03)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.01)
            for p in cell.text_frame.paragraphs:
                p.font.name = "NanumGothic"
                p.font.size = Pt(11)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor.from_string("202936")
    note = slide.shapes.add_textbox(Inches(0.7), Inches(6.72), Inches(5.75), Inches(0.38))
    p = note.text_frame.paragraphs[0]
    p.text = "Test: 이전 실험에 사용됨 · 독립 최종 검증 아님"
    p.font.name = "NanumGothic"
    p.font.size = Pt(10)
text = DOC.read_text()
text = text.replace(
    "ECG-FM 코드가 기존 test 89명에 대해서도 평가하지만 이 test는 과거 실험에서 이미 사용됐으므로 독립 최종 성능으로 취급하지 않는다. DeepECG와 ECGFounder는 이번 실행에서 test를 평가하지 않았다.",
    "세 모델 모두 과거 실험에서 사용된 test 89명을 재평가했다. 이 결과는 독립 최종 성능으로 취급하지 않는다.",
)
if "## Test 재평가 (기존 사용 이력 있음)" in text:
    raise SystemExit("Test report already exists")
DOC.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n")
prs.save(PPT)
print("Updated", PPT, "and", DOC)
