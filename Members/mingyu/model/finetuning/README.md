# MOMENT: Holter ECG foundation model (445명, 3유도)

사전학습된 시계열 foundation model **MOMENT-1-base**(AutonLab)로 Holter 30초 구간을 임베딩하고, 그 위에 MIL head 또는 선형 probe를 학습해 4년 내 SCD를 예측합니다. backbone은 고정하고 새로 붙인 부분만 학습합니다. 평가는 ML과 같은 5-fold × 3회 교차검증이고, 고른 후보 1개를 test에서 한 번 평가합니다.

## 전체 순서

```
preprocessing/01 → 02            445명 Holter 전처리 + 증강  →  <drive_root>/ECG_tem/
model/ML/data_prep/01 → 02       ML 임상 데이터 (fold 파일을 만들 때 필요)
model/finetuning/01              MOMENT 분할 파일            →  <drive_root>/ECG_tem/final_split_augmented.csv
model/finetuning/moment_cv       임베딩 → 교차검증 14개 후보 → test
```

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `01_make_augmented_split.py` | 445명 분할(`final_split_ecg_445.csv`) + **train SCD 환자 32명의 증강본 64개**(`-1`, `-2`) = 509줄. 증강본은 SCD 환자만 학습에 넣음 |
| 2 | `moment_cv.ipynb` | MOMENT 임베딩(처음 한 번) → 후보 14개 교차검증 → OOF 가중 log loss 1등을 test 85명에서 1회 평가 |
| - | `moment_probe.py` | MOMENT 불러오기, 임베딩 추출·캐시, MIL head 학습·예측 함수 |

발표 자료의 MOMENT 결과는 `moment_cv.ipynb`의 결과입니다 (선택: MIL_v2 = hidden_dim 64, dropout 0.5, learning_rate 1e-4, weight_decay 0.1, batch 16, class weight 학습 데이터 SCD 비율).

## 준비

### 데이터

`model/basic.yaml`의 `drive_root`를 본인 데이터 폴더로 바꾸고, 아래처럼 둡니다.

| 위치 (`drive_root` 기준) | 만드는 곳 |
|---|---|
| `ECG_tem/clean_signal/*.npy`, `ECG_tem/segment_rpeaks.csv`, `ECG_tem/lead_info.csv` | `preprocessing/02` (**`OUT_DIR`을 `<drive_root>/ECG_tem`으로 지정**하면 그대로 이 위치) |
| `final_split_ecg_445.csv` | 팀 Drive (445명 분할, ML·DL 공통) |
| `ECG_tem/final_split_augmented.csv` | `finetuning/01_make_augmented_split.py` |
| ML 임상 데이터 4개 (`clincal/…max.csv` 등) | `model/ML/data_prep/01`, `02` (fold 파일이 없을 때 fold를 만들려고 읽음) |

- 팀 Drive의 `ECG_tem/`을 그대로 받았다면 `lead_info.csv` 대신 `ECG_tem_clean.csv`가 있습니다 (내용 같음). 이 경우 `basic.yaml`의 `lead_metadata_path`를 `drive://ECG_tem/ECG_tem_clean.csv`로 바꿉니다.
- fold 파일 `Members/mingyu/results/ml_kfold/kfold_logloss_balanced/folds.csv`는 `ml_1_main`을 실행하면 생깁니다. 없으면 `moment_cv`가 ML과 같은 규칙(5-fold × 3, seed 2026)으로 fold만 만듭니다.

### 환경

- 패키지: `./dev env sync Members/mingyu` (`torch`, `momentfm` 포함). torch는 GPU의 CUDA 버전에 맞게 설치합니다.
- GPU 권장: 임베딩 추출(파일 445 + 64개 × 64구간)이 가장 오래 걸립니다. 한 번 계산하면 `Members/mingyu/datas/moment_cache/`에 저장되고 이후에는 다시 계산하지 않습니다.
- 인터넷: 처음 실행할 때 Hugging Face에서 `AutonLab/MOMENT-1-base` 가중치를 받습니다.

### 실행

```bash
uv run --project Global python Members/mingyu/model/finetuning/01_make_augmented_split.py   # 이미 있으면 같은지만 확인
# 노트북: moment_cv.ipynb 위에서부터
```

## 규칙

| 항목 | 설정 |
|---|---|
| 환자 | 445명 중 **Z 없는 2유도 22명은 학습·평가 모두 제외** (증강본 포함) → 교차검증 338명 (SCD 38), test 85명 (SCD 9) |
| 입력 | X·Y·Z 3유도, 환자당 30초 구간 64개 (제외 표시 조각 빼고 24시간에 고르게), 200 Hz → 100 Hz, 구간마다 512 샘플 조각 5개 임베딩 평균 |
| fold | ML `only445`와 같은 fold를 338명으로 제한. 증강본은 원본 환자의 fold를 따름 |
| 학습 fold | 나머지 원본 + 그 환자들의 SCD 증강본. 평가 fold: 원본만 |
| 후보 14개 | MIL head 2개 (`MIL_v1`: hidden 128·dropout 0.3·lr 1e-3·wd 1e-2, `MIL_v2`: hidden 64·dropout 0.5·lr 1e-4·wd 0.1) + 선형 probe 12개 (임베딩 mean / mean+std → PCA 16 / 64 → 로지스틱 C 0.01 / 0.1 / 1) |
| MIL 학습 | batch 16, 최대 200 epoch, 30 epoch 동안 개선이 없으면 중단. best epoch은 학습 fold 원본 20%를 떼어 그 가중 log loss로 고름 |
| class weight | 학습 데이터의 SCD 비율로 고정 (MIL `pos_weight`, 로지스틱 `class_weight="balanced"`) |
| 선택 | OOF **가중 log loss** 최소 (ML과 같음), threshold 0.5 고정 |
| 최종 | 고른 후보로 개발 338명 전체(증강 포함)를 재학습, test 85명 1회 평가 |

## 결과

`Members/mingyu/results/moment_cv/<run_tag>/` (Git 제외, 기본 `run_tag = moment_cv_445_xyz`)

```
settings.json            실행 설정 (다르면 실행을 멈춤 → run_tag를 새로 정함)
oof/<후보>.csv           후보별 OOF 예측 (끝난 후보는 다시 실행해도 건너뜀)
loss_curve/<후보>.csv    fold별 train / validation loss
test_predictions.csv     최종 후보의 test 예측
test_loss_curve.csv      최종 모델 학습 loss
```
