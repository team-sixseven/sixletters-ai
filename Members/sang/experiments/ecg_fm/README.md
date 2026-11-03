# ECG-FM Holter pilot

This is an exploratory 4-year SCD classifier using the frozen MIMIC-finetuned ECG-FM checkpoint, not a validated 12-lead ECG model for the Holter inputs. The raw cohort loader validates the existing patient split, lead availability and 30-second quality sidecar. Up to eight usable 30-second intervals are selected evenly per record; the first five seconds of each is resampled from 200 Hz to 500 Hz and standardized per channel. XYZ channels are placed arbitrarily into the first three ECG-FM input slots, with the remaining slots zeroed and missing Z masked. XYZ are not standard leads and this does not reconstruct 12-lead ECG. Compare alternative input strategies before interpreting this pilot as evidence about ECG-FM's potential.

The model's encoder outputs are averaged over time and segments, then over augmented records with the same patient ID. A standardized logistic regression is trained only on training patients. Five C values are chosen by validation average precision; test is then evaluated once. Input representations and selection differ from prior pilots; their metrics cannot be directly compared. The embeddings and data contain patient-linked information and should remain private.

Local preparation, from repository root:

```bash
Global/.venv/bin/python Members/sang/experiments/ecg_fm/prepare.py /tmp/sang_ecg_fm_pilot
```

On the configured RunPod, with `fairseq-signals`, compatible PyTorch and `transformers==4.45.2` installed and the official finetuned checkpoint downloaded:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv/bin/python run_pilot.py sang_ecg_fm_pilot ckpts/finetuned.pt results_pilot
```

PyTorch's weights-only restriction is disabled for the official checkpoint, which includes serialized model configuration; do this only with a trusted checkpoint. Dependencies in the Pod's venv use the Pod's system PyTorch. `/workspace` was container storage, not a mounted persistent volume, at execution time.

2026-10-30 A100 pilot: 509 records correspond to 285 train, 71 validation and 89 test patients. Validation AP 0.264 and AUROC 0.607; test AP 0.156 and AUROC 0.604. Training AP and AUROC were 1.0, consistent with substantial overfitting. Validation selected C=1.0. Test had only 10 positive cases, and 0.5-threshold sensitivity was 0.20; these exploratory scores are not clinical estimates. Results: `Members/sang/results/ecg_fm_pilot/metrics.json`.

## Local 3-channel adapter experiment

`train_adapter.py` trains a bias-free 1x1 convolution from XYZ to the ECG-FM model's twelve internal input slots, initialized to the earlier three-slot/zero mapping. The pretrained finetuned encoder is frozen; gradients pass through it to the adapter. An SCD linear head is trained alongside the adapter after per-example feature layer normalization. This is a learned feature mapping, not reconstruction of measured standard ECG leads. For training, each patient contributes one randomly selected five-second segment per epoch; for evaluation, logits are averaged across all selected segments and records for that patient. The available lead mask sets an unmeasured Z to zero.

The pilot uses seed 42, AdamW (learning rate 0.0003, weight decay 0.001), class-weighted BCE, batch size 4, maximum 10 epochs, and early stopping after four epochs without validation AP improvement. Only validation AP chooses the checkpoint. This has a different classifier and training procedure than the previous logistic-probe experiment, so the difference cannot be attributed only to the adapter. The test cohort was previously viewed and is not an untouched final holdout.

Execution in the local machine's isolated experiment environment, with the official checkpoint and fairseq-signals commit `f8f0ff1c788a82c2059cb452cd5462898867489e` available:

```bash
PYTHONPATH=.poc-cache/ecg-fm-fairseq TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .poc-cache/ecg-fm/venv/bin/python Members/sang/experiments/ecg_fm/train_adapter.py \
  --data /tmp/sang_ecg_fm_pilot --checkpoint .poc-cache/ecg-fm/finetuned.pt \
  --output Members/sang/results/ecg_fm_adapter --epochs 10 --patience 4
Global/.venv/bin/python Members/sang/experiments/ecg_fm/make_report.py
```

The local environment uses project PyTorch `2.13.0+cu126` and `transformers==4.45.2`, with fairseq-signals and other experiment dependencies isolated under `.poc-cache/ecg-fm/venv`. The prepared `/tmp` data is temporary; regenerate it using `prepare.py` before rerunning if absent.

2026-10-30 RTX 4060 pilot: validation-selected epoch 4, stopped after epoch 8. Test AUROC 0.611, AP 0.152, sensitivity 0/10 at threshold 0.5; no patient was predicted positive, so precision is undefined. Validation AP 0.285 with 8 positives. These scores are exploratory; 10 test positives and one training seed do not support a definitive performance claim. The JSON, learned adapter/head checkpoint and HTML report are stored in `Members/sang/results/ecg_fm_adapter/`.

## Partial encoder fine-tuning with final layers frozen

`train_finetune.py` starts from the preceding adapter/head checkpoint and the same official pretrained encoder. The encoder has 12 transformer layers: only indices 8 and 9 are unfrozen, while the final layers 10 and 11 and all other encoder layers stay frozen. The adapter and SCD head are trained too. AdamW uses learning rate 1e-5 for those two blocks and 1e-4 for the adapter/head. The other cohort and window policies remain as above. The code asserts that the final two layers remain bitwise unchanged and verifies that the selected trainable layers changed.

The best epoch is selected by validation AP. Afterwards, a threshold maximizing validation F1 is selected using validation patients only; both that threshold and 0.5 are evaluated. The small validation set makes the chosen threshold especially unstable, and it should not be treated as a calibrated probability cutoff. This previously used test set is not an independent final holdout.

```bash
PYTHONPATH=.poc-cache/ecg-fm-fairseq TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .poc-cache/ecg-fm/venv/bin/python Members/sang/experiments/ecg_fm/train_finetune.py \
  --data /tmp/sang_ecg_fm_pilot --checkpoint .poc-cache/ecg-fm/finetuned.pt \
  --adapter Members/sang/results/ecg_fm_adapter/best_adapter.pt \
  --output Members/sang/results/ecg_fm_finetune
Global/.venv/bin/python Members/sang/experiments/ecg_fm/report_finetune.py
```

2026-10-30 single-seed local run: selected epoch 1; final layers were verified unchanged. Test AUROC 0.663, AP 0.221; at 0.5 no positives were predicted. At validation-selected F1 threshold 0.018455, test precision was 5/36 (0.139), recall 5/10 (0.5), and specificity 48/79 (0.608). Validation contains 8 positive patients and test 10; these numbers are exploratory. Model checkpoint, metrics and visual report are in `Members/sang/results/ecg_fm_finetune/`.

환자별 예측은 `Members/sang/results/ecg_fm_finetune/patient_predictions.csv`에 저장한다. 환자 ID와 라벨이 포함되므로 비공개로 관리한다. `model_score`는 모델의 sigmoid 점수이며, 두 예측 열은 각각 0.5와 validation F1에서 선택한 기준을 사용한다. 점수를 재산출할 때는 다음을 실행한다. 스크립트는 각 분할의 집계 결과가 저장된 지표와 일치하는지 검사한다.

```bash
PYTHONPATH=.poc-cache/ecg-fm-fairseq TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .poc-cache/ecg-fm/venv/bin/python Members/sang/experiments/ecg_fm/export_predictions.py \
  --data /tmp/sang_ecg_fm_pilot --checkpoint .poc-cache/ecg-fm/finetuned.pt \
  --output Members/sang/results/ecg_fm_finetune
```

2026-10-30 재실행: `--epochs 100 --patience 10`, 동일 seed·동결 정책·초기 adapter를 사용했다. 1 epoch 이후 검증 AP가 10회 연속 개선되지 않아 11 epoch에서 종료하고 1 epoch 가중치를 선택했다. 검증 71명 예측은 `Members/sang/results/ecg_fm_finetune_e100_p10/validation.csv`에 별도 저장했고, 전체 분할 예측 및 시각화 보고서도 같은 디렉터리에 있다. 동일 설정의 기존 단일 seed 실행과 사실상 동일한 최종 성능이며 추가 반복의 개선 근거가 아니다.
