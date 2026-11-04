# 임상 ML: 반복 k-fold 교차검증

심부전 환자(MUSIC 코호트)의 4년 내 심장 급사(SCD)를 임상 변수와 24시간 Holter 요약값으로 예측하는 모델을, 같은 교차검증 fold에서 학습하고 비교합니다. 모델은 로지스틱 회귀(LR), XGBoost, Random Forest, CatBoost 4종 × 변수 묶음 4종입니다.

## 전체 순서

```
데이터 준비 (data_prep)          학습·평가 (노트북)
00 → 445명 임상 23 + Holter 4
01 → 445명 + 349명 = 794명   →   ml_0 → ml_1 → ml_2 → ml_3
02 → 결측 빈칸 + 환자 표시
```

| 순서 | 파일 | 내용 | 결과 |
|---|---|---|---|
| 데이터 0 | `data_prep/00_build_clinical_445_holter.py` | 팀 임상 파일 445명에 Holter 4개(`scripts/data_prep/holter_assoc.py`로 train에서 FDR q < 0.05 선정, VT는 NSVT와 중복이라 제외)를 붙이고 train 기준 중앙값·최빈값으로 결측 채움. 팀 Drive 파일과 같은 파일을 만듦 (이미 있으면 비교만) | 445명 임상 표 (29열) |
| 데이터 1 | `data_prep/01_extend_clinical_794.py` | 445명에 4년 결과를 아는 MUSIC 환자 349명을 더함 (추가분은 전부 train, val/test는 그대로). train 기준 중앙값·최빈값으로 결측 채움 | 794명 임상 표, 분할 |
| 데이터 2 | `data_prep/02_extend_clinical_794_raw.py` | 같은 794명을 결측 빈칸으로 다시 만들고 환자 표시 저장 (fold마다 결측 대체용) | 빈칸 임상 표, 환자 표시 |
| 0 | `ml_0_dataset.ipynb` | 학습 데이터셋 결정: 705명 vs 원래 445명 출신으로 각각 학습해 비교 → **445명 출신**. class weight를 Optuna로 함께 튜닝 | `kfold_logloss/` |
| 1 | `ml_1_main.ipynb` | class weight를 고정하고 445명 출신으로 학습, 모델 비교. 6번에서 최종 모델 1개 선택 | `kfold_logloss_balanced/` |
| 2 | `ml_2_test_all_models.ipynb` | 참고: 0·1의 445명 모델 전체를 test 89명에서 평가 (**모델 선택에 쓰지 않음**) | `test_only445_all.csv` |
| 3 | `ml_3_final_test.ipynb` | 최종 모델 1개 재학습과 test 1회 평가. **최종 모델을 고른 뒤 `final_name`을 적고 실행** (비어 있으면 멈춤) | `<결과 폴더>/test/` |

발표 자료의 ML 결과는 0(Optuna 튜닝, 445명 모델 16개), 1(class weight 고정, 16개), 2(두 묶음 test)에서 나왔습니다.

## 준비

### 데이터 위치

`model/basic.yaml`의 `drive_root` 한 줄을 본인 데이터 폴더로 바꿉니다. 아래 경로는 모두 이 폴더 기준입니다.

| 파일 | 출처 | 쓰는 곳 |
|---|---|---|
| `clincal/clinical_clean.csv` | 팀 Drive (원본 임상 552명, 변수 23개). 4년 정답은 데이터 0이 MUSIC 원본으로 다시 계산 (원본과 3명이 다름) | 데이터 0 |
| `final_split_ecg_445.csv` | 팀 Drive (445명 분할, ML·DL 공통) | 데이터 0, 1 |
| `clincal/clinical_clean_4year_holter.csv` | 데이터 0이 만듦 | 데이터 1 |
| `clincal/clinical_clean_4year_holter_max.csv`, `final_split_ml_max.csv` | 데이터 1이 만듦 | 데이터 2, 노트북 (`basic.yaml`) |
| `clincal/clinical_clean_4year_holter_max_raw.csv`, `clincal/ml_max_patient_flags.csv` | 데이터 2가 만듦 | 노트북 (`kfold/settings.py`) |

MUSIC 원본 `subject-info.csv`도 필요합니다. 데이터 0·1·2 스크립트 맨 위 `MUSIC_DIR` 한 줄을 그 폴더로 바꿉니다.

### 실행

```bash
# 데이터 준비 (0은 결과 파일이 있으면 비교만, 1·2는 덮어쓰지 않고 멈춤)
uv run --project Global python Members/mingyu/model/ML/data_prep/00_build_clinical_445_holter.py
uv run --project Global python Members/mingyu/model/ML/data_prep/01_extend_clinical_794.py
uv run --project Global python Members/mingyu/model/ML/data_prep/02_extend_clinical_794_raw.py
# 노트북: ml_0 → ml_1 → ml_2 (→ 최종 모델을 고른 뒤 ml_3)
```

필요 패키지: `scikit-learn`, `xgboost`, `catboost`, `optuna`, `joblib`, `pyyaml`, `matplotlib`, `pandas`, `numpy`. 트리 모델은 Optuna 100 trial × 15 fold라 모델·변수 묶음마다 시간이 걸립니다. 이미 학습된 방법은 다시 실행해도 건너뜁니다.

## 데이터

| 구분 | 환자 | SCD | 용도 |
|---|---|---|---|
| 개발 (`ml_0`의 `all`) | 705명 (원래 445명 출신 356 + 추가 349) | 81명 | 데이터셋 비교에만 사용 |
| 개발 (`only445`, 사용) | 원래 445명 출신 356명 | 40명 | 교차검증으로 학습, 튜닝, 모델 비교 |
| test | 89명 (모두 원래 445명 출신) | 10명 | `ml_2`(참고)와 `ml_3`(최종 1회)에서만 평가 |

- 결측은 추가 349명에게만 있습니다. 445명 줄은 팀 파일이 이미 결측을 채운 상태입니다.
- 학습 데이터셋 결정 (`ml_0`): 445명 출신 356명에서 705명 학습이 유의하게 나은 조합 0 / 16 (사전 규칙: 8개 이상이면 705명) → **445명 출신**

## 변수 묶음

| 이름 | 개수 | 내용 |
|---|---|---|
| `v5` | 27 | 팀 임상 변수 23 + Holter 4 |
| `v4` | 23 | 팀 기본 임상 변수 |
| `core` | 12 | 주요 임상 변수 + Holter 3 |
| `MUSIC_SCD` | 5 | MUSIC SCD 위험 점수 항목 (Pro-BNP, 좌심방 크기, NSVT, 24시간 심실조기수축 수, 이전 심근경색) |
| `ProBNP` | 1 | 기준 모델 (Pro-BNP 단독 로지스틱). 비교표의 Δ는 이 모델 대비 |

## 규칙 (모두 `settings.py`에 있음)

| 항목 | 설정 |
|---|---|
| 교차검증 | 층화 5-fold × 3반복 (seed 2026), 모든 방법이 같은 fold 사용. `only445`는 같은 fold에서 356명만 사용 |
| 하이퍼파라미터 | 방법당 1세트. OOF 예측의 log loss(3반복 평균)가 가장 낮은 설정 |
| 탐색 | 트리 모델: Optuna TPE 100 trial (팀 공통 탐색 범위 `xgboost/`·`catboost/`·`random_forest/parameter.yaml`). 로지스틱: C 0.001~10 중 9개 |
| class weight | `ml_0` (`kfold_logloss`): 고정하지 않음, 트리 모델은 Optuna가 함께 탐색. `ml_1` (`kfold_logloss_balanced`): fold마다 학습 환자의 SCD 비율로 고정 (XGBoost `scale_pos_weight` = 비SCD/SCD, RF·LR `class_weight="balanced"`, CatBoost `auto_class_weights="Balanced"`), log loss도 같은 가중치로 계산 |
| 결측 | iterative 대체. fold마다 학습 환자로만 fit, 27개 변수 전체로 대체, SCD는 사용하지 않음 |
| 전처리 | 로지스틱: 치우친 변수 log(x+1), 표준화, 3개 이상 범주 원-핫. 트리: 팀 공통 전처리 (`ML/models.py`) |
| threshold | 0.5 고정, 확률 보정 없음 |
| 모델 비교 | 같은 환자를 묶은 paired bootstrap (1000회)으로 ΔAUROC, ΔAP의 95% CI |
| 최종 모델 선택 (`ml_1` 6번) | OOF AUROC 순. 상위와의 ΔAUROC 95% CI가 0을 포함하면 더 단순한 모델 |
| 최종 모델 (`ml_3`) | 고른 설정으로 445명 출신 356명 전체를 재학습 (재튜닝 없음), test 1회 평가 |

교차검증 점수는 **모델을 고르는 데** 쓰고, 성능은 test로 보고합니다. 튜닝과 같은 fold로 계산한 점수라 약간 낙관적일 수 있습니다.

## 코드 구성

| 파일 | 역할 |
|---|---|
| `settings.py` | 모든 규칙과 상수 (변수 묶음, fold, trial 수, threshold, 결과 폴더) |
| `cohort.py` | 개발/test 환자, 원본 입력(빈칸 유지), fold 생성과 고정 |
| `imputation.py` | iterative 대체 (학습 환자로 fit → 학습/평가 환자 채우기) |
| `estimators.py` | 모델 4종 생성, 학습, 예측과 Optuna 탐색 범위 |
| `experiment.py` | 방법 하나(모델 × 변수 묶음 × 학습 환자) 튜닝과 OOF 예측 |
| `results.py` | 결과 폴더 저장·불러오기 |
| `report.py` | 결과 요약, AUROC 표, 비교표 |
| `curves.py` | 선택된 설정의 fold별 train / validation loss 기록 |
| `plots.py` | 예측 확률 분포 히스토그램, loss 곡선 |
| `final.py` | 최종 모델 학습, test 평가 |

팀 공통 코드를 그대로 사용합니다: `ML/models.py`, `ML/feature_versions.py`, `ML/model_choice.yaml`, `model/config.py`, `model/data.py`, `model/evaluation/core.py`, `model/basic.yaml`.

## 결과 폴더

`Members/mingyu/results/ml_kfold/` (Git 제외)

```
kfold_logloss/                    ml_0 결과 (class weight Optuna 튜닝)
kfold_logloss_balanced/           ml_1 결과 (class weight 고정)
  folds.csv, folds_info.json      고정된 fold와 데이터 해시 (데이터가 바뀌면 실행을 멈춤)
  <방법 이름>/oof.csv             평가 환자 × 3반복의 OOF 예측
  <방법 이름>/info.json           실행 내용과 고른 설정
  <방법 이름>/trials.csv          시험한 모든 설정과 OOF log loss
  <방법 이름>/probability_hist.png, loss_curve.png, loss_curve.csv
  comparison_*.csv                비교표
  test/<시각>_<이름>/             test 평가 결과
test_only445_all.csv              ml_2: 두 폴더 445명 모델 전체 test 지표
```

방법 이름은 `모델_변수묶음_iterative[_학습환자필터]` 형식입니다 (예: `LR_MUSIC_SCD_iterative_only445`, `XGBoost_v5_iterative_only445`).
