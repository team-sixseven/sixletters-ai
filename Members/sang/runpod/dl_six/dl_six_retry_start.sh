#!/bin/bash
set -euo pipefail
cd /workspace/dl_six
while [ ! -f files_ready ] || [ ! -f dependencies_ready ] || [ ! -f /tmp/dl_six_snapshot.tar ] || [ "$(stat -c %s /tmp/dl_six_snapshot.tar)" != "2346362880" ] || [ ! -f dl_six_support.tar.gz ]; do sleep 15; done
printf 'Input transfer and dependencies ready.\n'
tar -xzf dl_six_code.tar.gz --no-same-owner --no-same-permissions -C repo
tar -xzf dl_six_support.tar.gz --no-same-owner --no-same-permissions -C repo
/tmp/dl_six_venv/bin/python -u /workspace/dl_six/reconstruct_dl_snapshot.py
printf 'Starting six new combination trials.\n'
cd /workspace/dl_six/repo
PYTHONPATH=/workspace/dl_six/repo:/workspace/dl_six/repo/Members/sang/runpod/runpod_tcn_optuna /tmp/dl_six_venv/bin/python -u /workspace/dl_six/run_dl_six_60.py
