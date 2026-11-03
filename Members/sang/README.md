# 남상기 모델 실험

## 제출 코드 안내

- **DL 학습·평가:** [`model/DL/`](model/DL/)과 [`model/evaluation/`](model/evaluation/). RunPod에서 수행한 여섯 조합의 탐색 코드는 [`experiments/runpod_tcn_optuna/`](experiments/runpod_tcn_optuna/), 설정·조합 기록은 [`docs/runpod_60_trial_parameters_20261103.md`](docs/runpod_60_trial_parameters_20261103.md)에 있다.
- **ECG 파운데이션 모델:** [`experiments/ecg_fm/`](experiments/ecg_fm/)과 [`experiments/ecgfounder/`](experiments/ecgfounder/).
- **외부 API LLM:** [`notebooks/monorouter_holter_experiment.ipynb`](notebooks/monorouter_holter_experiment.ipynb), [`experiments/monorouter_holter/`](experiments/monorouter_holter/). 설정의 `run_live`는 실제 외부 API 호출 및 데이터 전송을 수행하므로 실행 전 설정과 승인을 확인한다. 키는 환경 변수로 제공한다.
- **Furiosa LLM 비교:** [`experiments/furiosa_llm_compare/`](experiments/furiosa_llm_compare/)와 `notebooks/furiosa_*.ipynb`. 30초 요약 비교는 `prepare_all_30s.py`, `prepare_summary30.py`, `compare_models.py`, `report_summary30.py` 순서로 실행한다. 사전 컴파일 모델과 Furiosa 런타임이 필요하다.
- **30초 파형 뷰어:** [`experiments/holter_viewer/`](experiments/holter_viewer/). 연구용 확인 구간은 검증된 SCD 위험 구간이나 QT 판정이 아니다.

환자 데이터·분할·체크포인트·환자별 예측과 생성 차트는 Git에서 제외한다. 노트북 출력은 환자 정보가 포함될 수 있어 제출본에서 비웠다. `model/basic.yaml`의 `drive_root`는 해당 YAML을 기준으로 `../datas`이며, 다른 데이터 위치를 사용하는 환경에서는 이를 수정한다. 모델·결과 폴더가 없는 새 환경에서는 해당 아티팩트를 별도로 준비해야 한다. 일부 과거 실행 노트북과 기록은 당시 서버 경로를 포함한다.

임상 ML, Holter DL, 저장 모델 확률 결합과 공통 평가 코드는 [`model/README.md`](model/README.md)에 있습니다. 입력 원본 CSV·NPY는 수정하거나 복사하지 않습니다.

저장소 루트에서 실행:

```bash
uv run --project Global --extra ml python Members/sang/scripts/audit_model_data.py
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py
uv run --project Global --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
uv run --project Global python -m Members.sang.model.evaluation.evaluation
```

공통 설정은 `model/basic.yaml`이 관리합니다. 임상은 `model/ML/data/clinical_clean.csv`와 `model/ML/data/final_split.csv`를 환자 ID로 join합니다 (`final_split.csv`의 `val`은 validation으로 매핑). Holter는 `datas/Holter/windows/{pid}.npy`와 `datas/Holter/index/windows.csv.gz`의 row/SQI 정보를 사용합니다. 현재 확인된 환자 수, modality별 사용 환자, 라벨 분포, 제외 사유는 audit 결과와 [`model/README.md`](model/README.md)에 기록합니다.

P1030은 NPY 입력·전처리·CNN+LSTM/Transformer forward 확인에 사용합니다. 임상 라벨이 없으므로 supervised 학습·성능 계산에는 사용하지 않습니다. 과거 더미 실험과 이전 split 비교는 `docs/`의 이력이며, 현재 `final_split.csv` 기반 성능 결과로 간주하지 않습니다.
