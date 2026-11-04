"""Summarize completed runs and update the three foundation-model slides."""
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

root = Path(__file__).resolve().parents[4]
base = root/'Members/sang/results/foundation_rerun_20261103'
slides = root/'Members/sang/docs/모델 정리.pptx'
out = root/'Members/sang/docs/foundation_models_e100_p10_20261103.md'
specs = [(14,'ecg_fm','ECG-FM','5초 · 500Hz · 본체 동결, 입력/출력 학습'),
         (15,'deepecg','DeepECG','10초 · 250Hz · 대체 인코더 동결, 어댑터 학습'),
         (16,'ecgfounder','ECGFounder','10초 · 500Hz · 본체 동결, 어댑터 학습')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
def graph(rows):
    y=np.array([int(r['scd_4year_label']) for r in rows]); p=np.array([float(r['model_score']) for r in rows]); bins=np.linspace(0,1,21)
    assert len(y)==71 and sum(y)==8 and np.all((0<=p)&(p<=1))
    fig,ax=plt.subplots(figsize=(7.2,2.9),layout='constrained')
    for label,color in [(0,'#3178ad'),(1,'#df6658')]:
        counts,_=np.histogram(p[y==label],bins=bins); assert sum(counts)==sum(y==label)
        ax.bar(bins[:-1]*100,counts*100/sum(y==label),width=4.6,align='edge',alpha=.55,color=color,label=f'SCD={label} (n={sum(y==label)})')
    ax.axvline(50,ls='--',lw=.8,color='#555');ax.set(xlim=(0,100),xlabel='Model score (%)',ylabel='Within-label (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=160);plt.close(fig);b.seek(0);return b

def loss(history):
    fig,ax=plt.subplots(figsize=(7.2,2.9),layout='constrained');e=[h['epoch'] for h in history]
    ax.plot(e,[h['train_loss'] for h in history],label='Train loss')
    ax.plot(e,[h['validation']['validation_loss'] for h in history],label='Validation loss')
    ax.set(xlabel='Epoch',ylabel='Weighted BCE');ax.legend(fontsize=8);ax.grid(alpha=.2)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=160);plt.close(fig);b.seek(0);return b

def put(shape,text,size=15):
    tf=shape.text_frame;tf.clear();tf.word_wrap=True
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='NanumGothic';p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string('252D39');p.space_after=Pt(3)

prs=Presentation(slides)
lines=['# ECG 파운데이션 모델 재실행 — 최대 100 epoch, patience 10','',
'공통 조건: seed 42, batch size 4, AdamW, 환자 단위 split (train 285명·양성 32명, validation 71명·양성 8명), train 증강 기록 포함. 검증 AP로 checkpoint를 선택하고 10 epoch 연속 개선이 없으면 중단했다. 검증 결과는 모델 선택에 사용한 탐색 결과다.','',
'각 환자별 최대 8개 품질 통과 구간을 사용하고, train에서는 환자별 한 구간을 무작위로 사용한다. 검증에서는 구간별 로짓을 기록 및 환자 단위로 평균한다. 세 모델은 사전학습 요구사항에 따라 입력 길이·샘플링·전처리가 다르다.','',
'Loss는 train 양성 가중치를 적용한 BCE다. Train은 학습 중 무작위 구간의 batch 평균, validation은 평가 모드에서 환자별 평균 로짓으로 계산한다. 두 곡선은 집계 방식이 달라 차이를 과적합의 정량값으로 해석할 수 없다.','',
'| 모델 | 입력 | 종료 / 선택 epoch | validation AUROC | validation AP | 양성 탐지 (0.5) |', '|---|---|---:|---:|---:|---:|']
for slide,name,title,details in specs:
    d=base/name; data=json.loads((d/'metrics.json').read_text()); history=data['history']; rows=list(csv.DictReader((d/'validation.csv').open()))
    assert history and [h['epoch'] for h in history]==list(range(1,len(history)+1)) and len(history)<=100
    m=data['splits']['validation'] if name=='ecg_fm' else data['validation']
    assert len(rows)==m['n'] and sum(int(r['scd_4year_label']) for r in rows)==m['positive']
    c=m.get('confusion'); tp=int(c['tp']) if c else sum(int(r['scd_4year_label']) and float(r['model_score'])>=.5 for r in rows)
    lines.append(f"| {title} | {details} | {len(history)} / {data['best_epoch']} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {tp}/8 |")
    s=prs.slides[slide-1]
    for shape in list(s.shapes):
        if shape.shape_type==13: shape._element.getparent().remove(shape._element)
    put(s.shapes[1],f"최대 100 epoch · patience 10 · seed 42\n{details}\n종료 {len(history)} / 선택 {data['best_epoch']} epoch\nValidation 71명 (양성 8명)\nAUROC {m['auroc']:.3f} · AP {m['average_precision']:.3f}",15)
    put(s.shapes[4],'Validation 예측 분포',15);put(s.shapes[5],'Train / validation loss',15)
    put(s.shapes[6],'Test: 기존 재사용 분할 · 독립 검증 아님' if name=='ecg_fm' else 'Test: 이번 실행에서 평가하지 않음',14)
    put(s.shapes[7],f"0.5 기준 양성 탐지 {tp}/8\n검증 AP로 epoch 선택: 탐색 결과\nLoss의 train/val 집계 단위 다름",14)
    s.shapes.add_picture(graph(rows),Inches(6.75),Inches(2.16),width=Inches(5.64),height=Inches(2.28))
    s.shapes.add_picture(loss(history),Inches(6.75),Inches(4.80),width=Inches(5.64),height=Inches(2.25))
lines += ['', 'DeepECG는 공식 모델이 요구하는 인코더 대신 공개 대체 인코더를 사용했다. 공개 기준 예측과 차이가 있어 공식 DeepECG 재현 결과가 아니다. XYZ를 표준 12유도로 복원했다고 볼 수 없다.','',
'ECG-FM 코드가 기존 test 89명에 대해서도 평가하지만 이 test는 과거 실험에서 이미 사용됐으므로 독립 최종 성능으로 취급하지 않는다. DeepECG와 ECGFounder는 이번 실행에서 test를 평가하지 않았다. 모델별 validation 양성은 8명뿐이므로 결과의 불확실성이 크다.','',
'히스토그램은 5%p 구간별 인원을 해당 실제 라벨의 전체 인원으로 나눈 비율이다. 환자별 예측 CSV에는 라벨 등이 포함되므로 비공개로 관리한다.','',
'실행 코드: `Members/sang/experiments/foundation_rerun/run.py`; 각 모델의 원본 코드와 체크포인트는 프로젝트 기존 설정을 사용한다. 환자별 예측과 이력은 `Members/sang/results/foundation_rerun_20261103/`에 저장한다.']
out.write_text('\n'.join(lines)+'\n');prs.save(slides)
print(out, 'and', slides)
