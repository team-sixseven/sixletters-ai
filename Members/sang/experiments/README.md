# 실험 코드 구조

| 경로 | 내용 |
| --- | --- |
| `runpod_tcn_optuna/` | ECG-only DL 모델 조합 및 Optuna 탐색 |
| `ecg_fm/`, `ecgfounder/` | ECG 파운데이션 모델 실험 |
| `monorouter_holter/` | 외부 API LLM의 Holter 특징 추출, 호출 보조 및 검증 |
| `furiosa_llm_compare/` | Furiosa LLM 전처리, 환자 단위 추론·평가·분포도 |
| `holter_viewer/` | 실제 30초 파형 확인 및 연구용 데모 계약 |
| `clinical_445/` | 임상 ML 실험과 파라미터 탐색 |
| `experiment_inventory/` | 로컬 실험 결과 정리 유틸리티 |

실행 진입점과 데이터 위치는 상위 [README](../README.md)를 참고한다. 환자 데이터, API 비밀값, 모델 가중치와 실행 결과는 저장소에 포함하지 않는다. 과거 실행 기록은 해당 환경과 시점의 상태이며 현재 실행 가능 여부를 보장하지 않는다.
