# Foundation model rerun

Runs the existing ECG-FM adapter, DeepECG adapter, and ECGFounder adapter experiments with the same patient split and maximum epochs/patience. `run.py` wraps their original training code to add patient-level, train-class-weighted validation BCE to each epoch's metrics. Training loss is a mean of batch losses on randomly sampled patient windows; validation loss uses the sigmoid of each patient's mean record/segment logit, and is therefore not the same aggregation. It does not alter checkpoint selection, which remains validation average precision. ECG-FM validation labels and scores are written without patient IDs. Original experiment code, environments, data and checkpoints remain required; outputs must be new, private directories.

Commands run from the repository root after preparing the ECG-FM input with `Members/sang/experiments/ecg_fm/prepare.py`:

```bash
PYTHONPATH=/home/sang/deepecg-3channel/fairseq-signals /home/sang/deepecg-3channel/.venv/bin/python Members/sang/experiments/foundation_rerun/run.py deep --base /home/sang/deepecg-3channel/checkpoints/SSL_pretrained.pt --public-encoder --weights /home/sang/deepecg-3channel/weights/deepecg_ssl_clsa_cvd_fold0.safetensors --output /NEW/PRIVATE/deepecg --epochs 100 --patience 10 --batch-size 4 --seed 42
Global/.venv/bin/python Members/sang/experiments/foundation_rerun/run.py founder --checkpoint .poc-cache/ecgfounder/12_lead_ECGFounder.pth --output /NEW/PRIVATE/ecgfounder --epochs 100 --patience 10 --batch-size 4 --seed 42
PYTHONPATH=.poc-cache/ecg-fm-fairseq TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .poc-cache/ecg-fm/venv/bin/python Members/sang/experiments/foundation_rerun/run.py fm --data /PRIVATE/PREPARED/DATA --checkpoint .poc-cache/ecg-fm/finetuned.pt --output /NEW/PRIVATE/ecg_fm --epochs 100 --patience 10 --batch-size 4
```

DeepECG uses the existing separate `/home/sang/deepecg-3channel` project and a public substitute encoder, whose reference outputs do not reproduce the official model. ECG-FM internally evaluates the previously used test set; it is not an independent final evaluation. Model preprocessing and frozen backbones differ; see the resulting report before comparing results. Do not publish patient-level files.
