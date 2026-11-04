# 실험 코드 구조

로컬에서 실행하는 실험을 모았다. RunPod와 Furiosa 서버에서 실행하는 코드는 별도 폴더에 있다.

| 경로 | 내용 |
| --- | --- |
| `ecg_fm/`, `ecgfounder/` | ECG 파운데이션 모델 실험 |
| `foundation_rerun/` | 파운데이션 모델 어댑터를 같은 분할·epoch 조건으로 다시 실행하고 validation loss 기록 |
| `monorouter_holter/` | 외부 API LLM의 Holter 특징 추출, 호출 보조 및 검증 |
| `holter_viewer/` | 실제 30초 파형 확인 및 연구용 데모 계약 (Furiosa 연동 스크립트 포함) |
| `clinical_445/` | 임상 ML 실험과 파라미터 탐색 |
| `experiment_inventory/` | 로컬 실험 결과 정리 유틸리티 |

| 서버별 폴더 | 내용 |
| --- | --- |
| [`../runpod/`](../runpod/) | RunPod GPU 서버 실행 코드 (Optuna 탐색, dl_six 서버 스크립트, 모니터) |
| [`../furiosa/`](../furiosa/) | Furiosa NPU 서버 LLM 비교 코드와 결과 확인 노트북 |

실행 진입점과 데이터 위치는 상위 [README](../README.md)를 참고한다. 환자 데이터, API 비밀값, 모델 가중치와 실행 결과는 저장소에 포함하지 않는다. 과거 실행 기록은 해당 환경과 시점의 상태이며 현재 실행 가능 여부를 보장하지 않는다.
