"""RunPod Serverless handler.  Request body: {"input": {12 feature names: values}}

Local test:  python handler.py --test_input "$(python -c 'import json;print(json.dumps({"input":json.load(open("sample_input.json"))}))')"
"""

from __future__ import annotations

import runpod

from model_total_process.total_flow import run_model


def handler(job: dict):
    x = job.get("input") or {}
    try:
        result = run_model(x)
    except (ValueError, TypeError, KeyError) as e:
        return {"error": str(e)}
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
