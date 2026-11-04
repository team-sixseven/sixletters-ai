# DL 전처리 + 증강: 445명 Holter ECG

MUSIC 원본에서 SCD 4년 예측 DL 모델용 Holter ECG 데이터셋(445명)을 만드는 코드입니다. 데이터는 저장소에 없으므로 아래 **준비** 파일을 각자 받아 두고 실행합니다.

```
01 코호트·4년 정답 + 팀 분할 (445명)  →  02 Holter ECG 전처리 + 증강 (1,335개)
```

| 순서 | 파일 | 하는 일 | 결과 |
|---|---|---|---|
| 01 | `01_cohort_445.ipynb` | MUSIC 730명 4년 SCD/PFD 정답(Swee 2026 규칙) → 팀 445명 분할에 붙임 | **445명** (SCD 50), train 285 · val 71 · test 89 |
| 02 | `02_ecg_preprocess_augment_445.ipynb` | 445명 Holter: 앞 30초 제거, 0.5–40 Hz 필터, X·Y·Z 3칸, R파 수·30초 조각 제외 표시, 증강(앞 15시간 `-1` / 뒤 15시간 `-2`) | `.npy` **1,335개** (원본 445 + 증강 890) |

## 445명이 되기까지

| 단계 | 남은 환자 (SCD) | 뺀 환자 (SCD) | 뺀 이유 |
|---|---|---|---|
| MUSIC 전체 | 992 | | |
| 1. Holter 없음 | 936 (85) | 56 (6) | Holter 기록이 없어 ECG를 쓸 수 없음 |
| 2. 비심장사 등 | 879 (85) | 57 (0) | 생존·SCD·PFD가 아닌 사망 원인 |
| 3. 등록 전 이식형 심장기기 | 758 (71) | 121 (14) | 페이스메이커·CRT는 심전도 리듬을, ICD는 돌연사 발생을 바꿈 |
| 4. 심장이식 | 746 (71) | 12 (0) | 이식으로 추적이 끝나 4년 결과를 알 수 없음 |
| 5. 4년 결과 모름 | **730** (71) | 16 (0) | 4년 전에 추적이 끝났고 사건도 없음 |
| 6. 4년 안 PFD 사망 | 648 (71) | 82 (0) | SCD와 경쟁하는 사건이라 SCD 정답을 정할 수 없음 |
| 7. 4년 뒤 PFD 사망 | 643 (71) | 5 (0) | 4년 정답은 0이지만, 팀 데이터는 PFD 사망을 시점과 상관없이 모두 뺌 |
| 8. 병인 코드 6·9, 나이 `>89` | 617 (71) | 26 (0) | 팀 임상 데이터에 이 값의 환자가 한 명도 없음 |
| 9. 결측치 | **445** (50) | 172 (21) | 팀 전처리에서 결측치가 있는 환자를 뺌 (임상 변수 빈칸 156명: 좌심실 수축기말 직경, 우심실 이완기말 직경, Pro-BNP 등 / 팀 분할 단계에서 결측으로 빠진 16명) |

- 1~5는 Swee 2026 코호트 규칙, 6은 Swee 4년 정답 규칙, 7~9는 팀 임상 전처리·분할에서 빠진 환자입니다. 자세한 내용은 01 노트북 맨 위 설명에 있습니다.
- 최종 445명: SCD 50 · 4년 안 심장사 없음 395. split은 팀 분할 train 285 · val 71 · test 89 (ML·DL 공통)

## 준비

| 파일 | 출처 | 쓰는 곳 |
|---|---|---|
| `subject-info.csv` | MUSIC 원본 (PhysioNet `music-sudden-cardiac-death`) | 01 |
| `RECORDS` | MUSIC 원본 | 02 |
| `Holter_ECG/` (`.hea`, `.dat`) | MUSIC 원본 | 02 |
| `final_split_ecg_445.csv` | 팀 Drive `DATA` (445명 분할 `pid`, `split`, `SCD`, ML·DL 공통) | 01 |

- MUSIC 원본은 PhysioNet 폴더 구조 그대로 둡니다. `subject-info.csv`, `RECORDS`, `Holter_ECG/`가 한 폴더 안에 있어야 합니다.
- 팀 파일 `final_split_ecg_445.csv`는 `RESULT_DIR`(아래)에 미리 넣어 둡니다.

## 경로 설정

**바꿀 줄은 각 노트북 ① 셀 맨 위에만 있습니다.** 나머지 경로는 모두 이 변수에서 만들어집니다.

| 파일 | 바꿀 줄 | 뜻 |
|---|---|---|
| 01 | `MUSIC_DIR` | MUSIC 원본 폴더 (`subject-info.csv`가 있는 곳) |
| 02 | `MUSIC_DIR` | 01과 같은 폴더 (`RECORDS`, `Holter_ECG/`가 있는 곳) |
| 02 | `OUT_DIR` | 전처리 결과 폴더 (약 105 GB, 여유 공간이 있는 디스크) |

`RESULT_DIR = ../results/final_holter_ecg`는 노트북 폴더 기준 상대경로라 바꾸지 않아도 됩니다 (이 저장소에서는 `Members/mingyu/results/final_holter_ecg/`, Git 제외 폴더). 01 결과와 02 처리 기록이 여기에 저장됩니다.

필요 패키지: `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, `tqdm`, `wfdb`, `neurokit2` (`wfdb`·`neurokit2`는 `Global`의 `ecg` 옵션, `scipy`는 `Members/mingyu/requirements.in`). 그림 글꼴은 `NanumGothic`이며, 없으면 한글만 깨지고 결과는 같습니다.

## 실행 순서

1. **01** 셀 ①→⑤ → `cohort_final.csv`, `labels_445.csv`
2. **02** 셀 ①→③ (시험 환자 2명, 처리 시간·메모리 확인) → ④ 전체 445명 → ⑤ 확인 → ⑥ 결과 폴더 정리

## 결과 확인 (기대값)

| 단계 | 확인할 값 |
|---|---|
| 01 ② | 992 → Holter 있음 936 → 생존·SCD·PFD 879 → 기기 이식 없음 758 → 심장이식 아님 746 → 4년 결과 확인 **730** |
| 01 ③ | 심장사 없음 577 · SCD 71 · PFD 82, SCD 분석 대상 648명 |
| 01 ⑤ | 분할 445명 · 기기 이식 제외 0명 · 정답이 바뀐 환자 없음 → **445명, SCD 50** (train 285/32 · val 71/8 · test 89/10) |
| 02 ① | Holter 기록 936개 중 처리 대상 445명 |
| 02 ⑤ | 유도 구성 XYZ 423 · XY 22 / `.npy` 1,335개, 빠진 것 없음 / Z 없는 22명의 Z 칸 전부 0 |
| 02 ⑥ | `lead_info.csv` 1,335줄, `segment_rpeaks.csv` 파일 1,335개 |

## 결과 파일

| 단계 | 파일 | 내용 |
|---|---|---|
| 01 | `results/final_holter_ecg/cohort_4year.csv`, `flow_cohort.csv` | 730명 4년 정답, 단계별 인원 |
| 01 | `results/final_holter_ecg/cohort_final.csv` | 445명: `pid`, `split`, `SCD_4year_label`, `outcome_code` |
| 01 | `results/final_holter_ecg/labels_445.csv` | 445명 정답: `pid`, `SCD_4year_label` |
| 02 | `OUT_DIR/clean_signal/*.npy` | 정제 신호 1,335개, `(샘플 수, 3)` X·Y·Z, float16, mV, 200 Hz |
| 02 | `OUT_DIR/segment_rpeaks.csv` | 30초 조각마다 유도별 R파 수와 `exclude` 표시 |
| 02 | `OUT_DIR/lead_info.csv` | 파일마다 원본 유도, `has_z`, 샘플 수, 시간 |
| 02 | `OUT_DIR/labels_445.csv`, `README.md` | 정답표 복사본, 데이터 설명 |

## 주의

- 445명 = 팀 분할 508명에서 등록 전 기기 이식 63명을 뺀 환자이고, 분할은 이 445명 기준으로 팀이 다시 나눈 것입니다 (ML·DL 공통, test 89명). 01은 분할을 새로 만들지 않고 이 파일을 그대로 씁니다.
- 02는 split을 쓰지 않습니다 (445명 목록과 정답만 사용). 증강본(`P0001-1`, `P0001-2`)은 학습할 때 원본 환자와 같은 split에 넣습니다.
- **02**는 처리 기록(`RESULT_DIR/record_log.csv`)에 있는 환자를 건너뜁니다. 처음부터 다시 하려면 이 파일을 지웁니다. `N_JOBS`는 ③ 시험의 `peak_mem_gb`를 보고 정합니다.
