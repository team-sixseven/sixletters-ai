# RunPod 실행 코드

RunPod GPU 서버에서 실행하려고 만든 코드를 모았다. 로컬에서 실행하는 실험은 [`../experiments/`](../experiments/)에 있다.

| 경로 | 내용 |
| --- | --- |
| [`runpod_tcn_optuna/`](runpod_tcn_optuna/) | ECG-only DL 여섯 조합의 Optuna 탐색, 전송·동기화·자동 중지 보조 도구 |
| [`dl_six/`](dl_six/) | 서버(`/workspace/dl_six`)에서 실행한 재시도·test 평가·결과 보고 스크립트 |

## 사용 시 주의

- `dl_six/` 스크립트는 서버 경로(`/workspace/dl_six`, `/tmp/dl_six_venv`)를 그대로 쓴다. 서버의 `repo/` 아래에 이 저장소 코드가 풀려 있어야 하며, 새 Pod에서는 `/tmp`가 비어 있으므로 스냅샷을 다시 올려야 한다.
- 파형 스냅샷(`dl_six_snapshot.tar`, 약 2.3GB)은 저장소에 없다. 학습은 이 스냅샷의 128구간 표본에서만 동작한다.
- Pod 접속, 디스크, 종료 절차는 [`../docs/runpod_guide.md`](../docs/runpod_guide.md)를 따른다. SSH 개인키와 Jupyter 토큰은 저장소에 두지 않는다.
- 실행 기록과 설정은 [`../docs/runpod_ecg_60_experiments_20261101.md`](../docs/runpod_ecg_60_experiments_20261101.md), [`../docs/runpod_60_trial_parameters_20261103.md`](../docs/runpod_60_trial_parameters_20261103.md)에 있다.
