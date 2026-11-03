"""LLM input/output contract for an experimental 30-second RR-summary demo.

The output is not an SCD probability, diagnostic interpretation or validated
priority scale. No patient identifiers, labels, raw waveform or free text are
included in the model prompt.
"""

import json
import math

CONTRACT_VERSION = "holter-rr30-review-v1"
SCHEMA = {
    "type": "object",
    "properties": {"review_score": {"type": "number", "minimum": 0, "maximum": 100}},
    "required": ["review_score"],
    "additionalProperties": False,
}
SYSTEM = (
    "Research demonstration only, not clinical care. Given a 30-second summary of experimental RR intervals, "
    "return a numeric 0-100 exploratory review score for this segment. The score is not a disease probability, "
    "SCD risk, diagnosis or validated priority. ECG waveform shape, T waves, rhythm annotations and patient "
    "outcomes are unavailable. Do not infer missing information. Return JSON with only review_score."
)
FIELDS = (
    "duration_seconds",
    "sample_rate_hz",
    "lead",
    "r_peak_count",
    "rr_valid_count",
    "rr_mean_ms",
    "rr_sd_ms",
    "rr_rmssd_ms",
    "mean_heart_rate_bpm",
)


def sanitize_features(features):
    if not isinstance(features, dict):
        raise ValueError("features must be a mapping")
    missing = set(FIELDS) - features.keys()
    allowed = set(FIELDS) | {"note"}
    if missing or features.keys() - allowed:
        raise ValueError("unsupported or missing feature fields")
    result = {key: features[key] for key in FIELDS}
    if result["duration_seconds"] != 30 or result["sample_rate_hz"] != 200:
        raise ValueError("expected a 30-second segment at 200 Hz")
    if result["lead"] not in ("X", "Y", "Z"):
        raise ValueError("unsupported lead")
    for field in ("r_peak_count", "rr_valid_count"):
        value = result[field]
        if type(value) is not int or not 0 <= value <= 6000:
            raise ValueError("invalid count")
    if (
        result["r_peak_count"] < 3
        or result["rr_valid_count"] < 2
        or result["rr_valid_count"] >= result["r_peak_count"]
    ):
        raise ValueError("insufficient or inconsistent RR counts")
    for field in ("rr_mean_ms", "rr_sd_ms", "rr_rmssd_ms", "mean_heart_rate_bpm"):
        value = result[field]
        if value is None and field in ("rr_sd_ms", "rr_rmssd_ms"):
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10000:
            # Zero variation is a valid numerical observation.
            if field in ("rr_sd_ms", "rr_rmssd_ms") and type(value) in (int, float) and value == 0:
                continue
            raise ValueError("invalid numeric feature")
    return result


def build_messages(features):
    user_data = {"segment_summary": sanitize_features(features)}
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                user_data, ensure_ascii=True, allow_nan=False, separators=(",", ":")
            ),
        },
    ]


def parse_model_output(text, finish_reason):
    if finish_reason != "stop":
        raise ValueError("model response did not finish")
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("model response must be JSON") from exc
    if not isinstance(value, dict) or set(value) != {"review_score"}:
        raise ValueError("model output schema mismatch")
    score = value["review_score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("invalid review_score")
    return float(score) / 100
