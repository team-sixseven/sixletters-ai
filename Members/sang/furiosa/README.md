# Furiosa 서버 실행 코드

Furiosa NPU 서버(`std3`)에서 LLM을 돌리려고 만든 코드와 결과 확인 노트북을 모았다. 사전 컴파일 모델과 Furiosa 런타임이 필요하다.

| 경로 | 내용 |
| --- | --- |
| [`furiosa_llm_compare/`](furiosa_llm_compare/) | Holter·임상 입력 생성, 환자 단위 추론·집계, 분포도, 임계값·진단 비교 |
| [`notebooks/`](notebooks/) | 서버 커널에서 여는 `furiosa_*.ipynb`. 대부분 서버의 `/root/.cache/std3_holter/llm_compare` 결과를 읽기만 하며 모델을 다시 실행하지 않는다 |

30초 요약 비교는 `prepare_all_30s.py`, `prepare_summary30.py`, `compare_models.py`, `report_summary30.py` 순서로 실행한다. 노트북이 읽는 서버 경로는 로컬에 없으므로 서버 밖에서는 동작하지 않는다.

30초 파형 뷰어의 Furiosa 연동 스크립트(`furiosa_30s.py`, `furiosa_one_patient.py`)는 `holter_viewer/demo_adapter.py`가 이 폴더의 로컬 경로 기준으로 서버에 올리고 `llm_contract.py`와 함께 쓰므로 [`../experiments/holter_viewer/`](../experiments/holter_viewer/)에 남겼다. 결과 요약은 [`../docs/`](../docs/)의 `furiosa_*` 문서를 참고한다. 환자 데이터와 환자별 예측은 저장소에 포함하지 않는다.
