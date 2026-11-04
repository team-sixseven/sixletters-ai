import copy
import csv
import json
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE

base = Path("/workspace/dl_six")
state = json.loads((base / "six_search/status.json").read_text())
prs = Presentation(base / "source_deck.pptx")
template = prs.slides[4]
out = base / "presentation"
out.mkdir(exist_ok=True)
names = {
    "tcn_transformer": "TCN + Transformer",
    "resnet_attention_lstm_fcn": "ResNet + Attention LSTM",
    "tcn_attention_lstm_fcn": "TCN + Attention LSTM",
    "resnet_bilstm": "ResNet + BiLSTM",
    "tcn_bilstm": "TCN + BiLSTM",
    "resnet_transformer": "ResNet + Transformer",
}


def settext(sh, text, size):
    sh.text_frame.clear()
    for i, line in enumerate(text.split("\n")):
        p = sh.text_frame.paragraphs[0] if i == 0 else sh.text_frame.add_paragraph()
        p.text = line
        p.font.name = "맑은 고딕"
        p.font.size = Pt(size)
        p.font.color.rgb = RGBColor(45, 45, 45)


for name, title in names.items():
    st = state["combinations"].get(name, {})
    run_dir = st.get("run_dir")
    if not run_dir:
        continue
    run = Path(run_dir)
    metrics = json.loads((run / "metrics.json").read_text())["validation"]
    meta = json.loads((run / "metadata.json").read_text())
    with (run / "training_history.csv").open() as f:
        hist = list(csv.DictReader(f))
    with (run / "validation_predictions.csv").open() as f:
        pred = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 3))
    for label, color in [(0, "#4779a8"), (1, "#e8934d")]:
        values = [float(x["p_positive"]) for x in pred if int(x["y_true"]) == label]
        ax.hist(
            values,
            bins=[i / 10 for i in range(11)],
            alpha=0.6,
            label=f"Actual {label}",
            color=color,
        )
    ax.set_xlabel("Predicted SCD score")
    ax.set_ylabel("Patients")
    ax.legend()
    fig.tight_layout()
    dist = out / (name + "_val.png")
    fig.savefig(dist, dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(
        [int(x["epoch"]) for x in hist],
        [float(x["training_loss"]) for x in hist],
        label="Train",
    )
    ax.plot(
        [int(x["epoch"]) for x in hist],
        [float(x["validation_loss"]) for x in hist],
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    loss = out / (name + "_loss.png")
    fig.savefig(loss, dpi=130)
    plt.close(fig)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    for sh in list(slide.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in template.shapes:
        if sh.shape_type != MSO_SHAPE_TYPE.PICTURE:
            slide.shapes._spTree.insert_element_before(
                copy.deepcopy(sh._element), "p:extLst"
            )
    ts = [x for x in slide.shapes if x.has_text_frame]
    settext(ts[0], title + " · 추가 실험", 28)
    settext(
        ts[1],
        f"Patience 10 · 최대 100 epoch\n종료 {meta['completed_epochs']} / 선택 {meta['best_epoch']} epoch\nValidation {metrics['n']}명 (양성 {metrics['scd']}명)\nAUROC {metrics['auroc']:.3f} · AP {metrics['average_precision']:.3f}\n최대 128구간 · 3채널 원본 선택값",
        17,
    )
    settext(ts[2], "", 14)
    settext(ts[3], "", 14)
    settext(ts[4], "Validation 예측 분포", 16)
    settext(ts[5], "Train / validation loss", 16)
    if st.get("test_status") == "complete":
        t = json.loads((Path(st["test_dir"]) / "metrics.json").read_text())
        cm = t["confusion_matrix"]
        summary = f"Test {t['n']}명 · AUROC {t['auroc']:.3f} · AP {t['average_precision']:.3f}\nTN {cm[0][0]} · FP {cm[0][1]} · FN {cm[1][0]} · TP {cm[1][1]}\n이전 사용 이력 있음 · 독립 test 아님"
    else:
        summary = "Test 보류: 60 epoch 조건 미충족\nTest는 과거 사용 이력이 있음"
    settext(ts[6], summary, 14)
    settext(ts[7], "", 14)
    slide.shapes.add_picture(
        str(dist), Inches(6.8), Inches(2.2), width=Inches(5.6), height=Inches(2.3)
    )
    slide.shapes.add_picture(
        str(loss), Inches(6.8), Inches(4.8), width=Inches(5.6), height=Inches(2.2)
    )
result = out / "model_summary_dl_six.pptx"
tmp = out / "model_summary_dl_six.tmp.pptx"
prs.save(tmp)
os.replace(tmp, result)
print("PPT_UPDATED", len(prs.slides), flush=True)
