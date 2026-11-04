# RunPod TCN + transformer Optuna 준비

현재 상태: **준비만 됨. RunPod SSH 공개키 인증 거부로 실험·GPU 검증·평가 CSV 미실행.** 모델 버전이 임상 feature v1~v4인지 여부도 확인 필요하다. 이 스크립트는 우선 ECG-only TCN+transformer_optuna이며 임상 변수 v1~v4를 적용하거나 fusion하지 않는다.

기존 사용자 `basic.yaml`, DL YAML, 노트북을 수정하지 않고 trial마다 별도 `basic.yaml`·`model_choice.yaml`·`training.yaml`·TCN/Transformer YAML을 생성한다. `train_raw_ecg`는 각 trial의 최적 epoch를 **validation loss**로 정하며, Optuna는 선택 checkpoint의 **validation AP**를 비교한다. 두 기준이 다르므로 결과 해석 시 구분한다. 각 trial의 validation predictions, Optuna trial CSV와 선택 모델의 test evaluation prediction CSV를 저장하도록 구성했다. 이전에 이미 사용한 test cohort는 독립적인 최종 holdout이 아니다.

[search_space.yaml](search_space.yaml)에서 탐색 범위를 관리한다. 탐색은 learning_rate, weight_decay, classifier dropout, TCN hidden dimension/kernel, Transformer dropout/layer 수를 바꾸고 `training.yaml`의 나머지 설정·분할·구간 선택을 유지한다. 기존 raw 학습기는 원본/증강 기록을 각각 train 기록으로 사용하며 양성 class weight도 기록 수 기준이라는 점을 다른 환자 단위 Foundation Model 실험과 구분한다.

실행 전 원격 GPU와 Python/Optuna, 지정된 Holter 디렉터리 접근 및 저장 공간을 확인해야 한다. 다음은 **실행 예시이지 실행 결과가 아니다**. 예산과 버전 해석을 확정한 뒤 고유 output 경로를 지정한다. 환자별 CSV와 audit이 생성되므로 비공개 경로를 사용하고 외부 전송 여부를 확인한다.

```bash
python Members/sang/runpod/runpod_tcn_optuna/run.py \
  --data-root /path/to/datas --output /private/new/run-directory \
  --trials 8 --epochs 100 --patience 10
```

`--data-root` 아래 `Holter/final_split_augmented.csv`, `windows/`, `segment_rpeaks.csv`, `ECG_tem_clean.csv`가 필요하다. output 디렉터리는 기존 파일을 덮어쓰지 않도록 새 경로여야 한다. GPU가 없거나 저장 실패 시 종료되며 작업 중단 후 현재 실행 스크립트는 자동 재개하지 않는다. Optuna SQLite와 개별 trial 결과는 output에 보존된다.

로컬 확인: 합성 입력의 단일 trial YAML 생성과 TCN+Transformer CPU forward 통과. 실제 데이터·GPU·원격 다운로드/학습·test 및 환경 버전 호환성은 미확인.

## A100 실행 중 배치 설정 변경

2026-11-01 실행에서 다음 trial부터 `batch_size: 6`, `num_workers: 4`를 사용한다. 현재 trial의 설정은 변경하지 않는다. 앞선 batch 1 결과와 optimizer 업데이트 횟수가 다르므로 validation 비교에서 배치 설정을 함께 확인한다. 모든 조합은 epoch 100, patience 10을 유지한다.

A100 80GB, PyTorch 2.8.0+cu128에서 실제 파형 128구간×3유도×6000 samples를 이용한 단일 forward/backward 점검 결과, TCN hidden 64/kernel 5 + Transformer 2 layers의 batch 6 최대 할당 메모리는 약 56.2 GiB였다. 이 단일 단계 측정은 전체 학습 속도나 모든 데이터 배치의 메모리 상한을 보장하지 않는다. 데이터로더 worker와 실제 학습 속도는 다음 trial에서 확인한다. 더 큰 배치는 메모리 여유와 실행 중인 작업을 고려해 시험하지 않았다.
