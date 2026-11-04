# 남상기 모델 실험

## 제출 코드 안내

- **DL 학습·평가:** [`model/DL/`](model/DL/)과 [`model/evaluation/`](model/evaluation/). RunPod에서 수행한 여섯 조합의 탐색 코드는 [`runpod/dl_optuna_search/`](runpod/dl_optuna_search/), 서버 재시도·test 평가 스크립트는 [`runpod/dl_six/`](runpod/dl_six/)(전체 안내는 [`runpod/README.md`](runpod/README.md)), 설정·조합 기록은 [`docs/runpod_60_trial_parameters_20261103.md`](docs/runpod_60_trial_parameters_20261103.md)에 있다.
- **ECG 파운데이션 모델:** [`experiments/ecg_fm/`](experiments/ecg_fm/)과 [`experiments/ecgfounder/`](experiments/ecgfounder/).
- **외부 API LLM:** [`notebooks/monorouter_holter_experiment.ipynb`](notebooks/monorouter_holter_experiment.ipynb), [`experiments/monorouter_holter/`](experiments/monorouter_holter/). 설정의 `run_live`는 실제 외부 API 호출 및 데이터 전송을 수행하므로 실행 전 설정과 승인을 확인한다. 키는 환경 변수로 제공한다.
- **LLM 전체 445명 결과:** [`docs/llm_full445_results_20261104.md`](docs/llm_full445_results_20261104.md)(Furiosa 4모델·Haiku·Terra). 집계 코드는 [`scripts/summarize_llm_full445.py`](scripts/summarize_llm_full445.py), 모델·실행 환경별 위치는 [`docs/trained_models_inventory.md`](docs/trained_models_inventory.md).
- **Furiosa LLM 비교:** [`furiosa/furiosa_llm_compare/`](furiosa/furiosa_llm_compare/)와 `furiosa/notebooks/furiosa_*.ipynb`. 30초 요약 비교는 `prepare_all_30s.py`, `prepare_summary30.py`, `compare_models.py`, `report_summary30.py` 순서로 실행한다. 사전 컴파일 모델과 Furiosa 런타임이 필요하다.
- **30초 파형 뷰어:** [`experiments/holter_viewer/`](experiments/holter_viewer/). 연구용 확인 구간은 검증된 SCD 위험 구간이나 QT 판정이 아니다.

환자 데이터·분할·체크포인트·환자별 예측과 생성 차트는 Git에서 제외한다. 노트북 출력은 환자 정보가 포함될 수 있어 제출본에서 비웠다. `model/basic.yaml`의 `drive_root`는 해당 YAML을 기준으로 `../datas`이며, 다른 데이터 위치를 사용하는 환경에서는 이를 수정한다. 모델·결과 폴더가 없는 새 환경에서는 해당 아티팩트를 별도로 준비해야 한다. 일부 과거 실행 노트북과 기록은 당시 서버 경로를 포함한다.

Holter DL과 공통 평가 코드는 [`model/README.md`](model/README.md)에 있습니다. 임상 ML 모델 코드와 ML·DL 결합 실행은 이 제출에 포함하지 않습니다. 입력 원본 CSV·NPY는 수정하거나 복사하지 않습니다.

저장소 루트에서 실행:

```bash
uv run --project Global python Members/sang/scripts/audit_model_data.py
uv run --project Global --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
uv run --project Global python -m Members.sang.model.evaluation.evaluation
```

공통 설정은 `model/basic.yaml`이 관리합니다. Holter 입력과 데이터 감사에 관한 설명은 [`model/README.md`](model/README.md)에 기록합니다.

P1030은 NPY 입력·전처리·CNN+LSTM/Transformer forward 확인에 사용합니다. 임상 라벨이 없으므로 supervised 학습·성능 계산에는 사용하지 않습니다. 과거 더미 실험과 이전 split 비교는 `docs/`의 이력이며, 현재 `final_split.csv` 기반 성능 결과로 간주하지 않습니다.
