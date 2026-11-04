# 학습·사용 모델 위치와 실행 환경 정리 (2026-11-04 기준)

모델 가중치와 체크포인트는 용량과 환자 데이터 정책 때문에 Git에 올리지 않는다(`results/`, `model/*/models/`, `datas/`는 `.gitignore` 대상). 아래 경로는 작업자 PC 기준이며, 저장소에는 이 문서와 학습·평가 코드만 있다. 실행 환경은 결과 파일의 `device` 기록과 코드·폴더 이름으로 확인했고, 확인하지 못한 항목은 "추정"으로 표시했다.

## 1. 로컬 PC에서 학습 (RTX 4060 Laptop)

| 모델 | 위치 | 규모 | 근거 |
| --- | --- | --- | --- |
| 임상 ML (CatBoost, Random Forest, XGBoost) | `Members/sang/model/ML/models/`, `results/clinical_445_20261030/`, `results/clinical_445_parameter_cases/`, `results/clinical_445_versions_parameter_cases/` | 118개 파일, 70개 폴더, 약 56MB | 로컬 실험 폴더 기준, 2026-10-28~30 (장치 기록은 확인하지 못함) |
| DL 초기·스모크 모델 | `Members/sang/model/DL/models/` | 3개, 약 1.4MB | 로컬 스모크 실행, 2026-10-29~30 |
| ECG-FM 어댑터·파인튜닝 | `results/ecg_fm_adapter/`, `ecg_fm_finetune*/`, `foundation_rerun_20261103/ecg_fm/` | 결과 `metrics.json`의 `device`가 RTX 4060 | 결과 파일 기록 |
| ECGFounder 어댑터 | `results/ECGFounder_result/` | README에 "로컬 RTX 4060" 명시 | 결과 README |
| DeepECG 어댑터 | `results/DeepECG_result/` | 위 파운데이션 모델 어댑터 합계 19개 파일, 약 114MB | 로컬 환경(`deepecg-3channel`) 사용, GPU 기록은 없어 추정 |

사전학습 가중치(우리가 학습한 것이 아님): `deepecg-3channel/`의 DeepECG 가중치 약 2.9GB, `.poc-cache/ecg-fm/finetuned.pt`(공식 파인튜닝 체크포인트) 약 1.1GB, `.poc-cache/ecgfounder/12_lead_ECGFounder.pth` 약 0.4GB. 공개 가중치를 내려받은 것이다.

## 2. RunPod에서 학습

| 모델 | 위치 | 규모 | 근거 |
| --- | --- | --- | --- |
| ECG-only DL 6조합 Optuna 60 trial | `results/runpod_tcn_20261101/output/study*/models/trial_NNN/best_model.pt` | 60개, 약 26MB | `runpod/dl_optuna_search/`로 실행 |
| DL 6조합 재시도·test 평가용 모델 | `results/dl_six_retry_20261104/dl_six_retry_results.tar.gz`, `results/dl_six_test_20261104/`, `server_backups/runpod_dl_six/best_previous/` | 12개, 약 5MB | `runpod/dl_six/`로 실행, 서버 `/workspace/dl_six` |
| MedGemma 27B QLoRA 어댑터 | `/home/sang/medgemma_runpod/results*/checkpoint-*/adapter_model.safetensors` | 52개 파일(어댑터 + optimizer 상태), 약 8.9GB. 어댑터 1개당 약 454MB | 폴더·스크립트 이름과 학습 기록 기준(GPU 이름은 결과에 기록되지 않아 추정) |

MedGemma는 `results/`(1 epoch), `results_5epoch/`(5 epoch), `results_balanced*/`(균형 표본), `results_class_weight_b8/`(클래스 가중, batch 8) 네 가지 설정이 있고, 각 폴더의 `checkpoint-35/70/105/140/175`가 epoch별 어댑터다. 기반 모델은 `google/medgemma-27b-text-it`이며 별도로 내려받아야 한다. 이 폴더는 저장소 밖(`/home/sang/medgemma_runpod`)에 있다.

## 3. Furiosa 서버(std3)에서 실행 — 학습 없음

Furiosa에서는 모델을 **학습하지 않고** 미세조정 없는 LLM 추론만 했다. 따라서 우리가 만든 가중치가 없다.

| 사용 모델 | 위치 | 비고 |
| --- | --- | --- |
| Qwen3-32B(FP8), Qwen3-30B-A3B-Instruct-2507(FP8), EXAONE-4.0-32B(FP8), GPT-OSS-20B | 서버의 사전 컴파일 모델 | 로컬에 가중치 없음 |
| 추론 결과(예측·지표) | 서버 `/root/.cache/std3_holter/llm_compare/`, 로컬 복사본 `results/furiosa_llm_compare/`, `results/furiosa_holter/`, `results/furiosa_summary30_20261103/` | 환자별 예측은 Git 제외 |

## 4. 모델 파일이 없는 실험 (외부 API)

Claude Haiku 5.5, GPT-6-Luna, GPT-6.1-Terra, Jev는 외부 API를 호출했을 뿐이라 가중치가 없다. 결과는 `results/monorouter_holter/`, `results/openai_holter/`, `results/jev_clinical/`에 있다.

## 5. Git에 올리는 것과 올리지 않는 것

| 구분 | 내용 |
| --- | --- |
| 올림 | 학습·평가 코드(`model/`, `experiments/`, `runpod/`, `furiosa/`, `scripts/`), 설정 YAML, 문서(`docs/`), 출력을 비운 노트북 |
| 올리지 않음 | 환자 데이터(`datas/`, `split_datas/`), 모델·체크포인트(`model/*/models/`, `results/`), 환자별 예측, 생성 차트, 발표 자료(`*.pptx`와 백업), API 키·SSH 키, 서버 접속·자동 중지 스크립트(`.git/info/exclude`로 로컬 제외) |
