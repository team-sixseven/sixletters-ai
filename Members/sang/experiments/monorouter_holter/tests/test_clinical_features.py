import csv

import neurokit2 as nk
import numpy as np
import pytest
from Members.sang.experiments.monorouter_holter.clinical_features import (
    detect_peaks_by_lead,
    filtered_rr,
    interval_features,
    load_quality_index,
    prsa_dc,
    select_development,
    validate_llm_features,
)


def test_time_metrics_match_neurokit_on_continuous_clean_rr():
    rr = np.array([800, 840, 910, 880, 920, 950, 900, 940, 1010, 970], dtype=int)
    peaks = np.r_[0, np.cumsum(rr)]
    result = interval_features(peaks, np.zeros(len(peaks), dtype=int), 1000, "0", 10, 30, "X")
    reference = nk.hrv_time({"ECG_R_Peaks": peaks}, sampling_rate=1000)
    for own, reference_key in [
        ("SDNN_ms", "HRV_SDNN"),
        ("RMSSD_ms", "HRV_RMSSD"),
        ("pNN50_percent", "HRV_pNN50"),
    ]:
        assert result["hrv_time_domain_estimates"][own] == pytest.approx(
            reference[reference_key].iloc[0]
        )
    assert result["quality"]["hrv_status"] == "experimental_rr_surrogate"


def test_rejected_and_excluded_gaps_are_not_joined():
    peaks = np.array([0, 1000, 2000, 10000, 11000, 12000])
    blocks = np.array([0, 0, 0, 1, 1, 1])
    rr, good, same = filtered_rr(peaks, blocks, 1000)
    assert same.tolist() == [True, True, False, True, True]
    assert good.tolist() == [True, True, False, True, True]
    result = interval_features(peaks, blocks, 1000, "0", 100, 30, "X")
    assert result["quality"]["consecutive_pair_count"] == 2
    assert result["hrv_time_domain_estimates"]["RMSSD_ms"] == 0
    assert result["deceleration_capacity_estimate_ms"] is None
    assert prsa_dc(rr, good, same) == (None, 0)


def test_prsa_uses_aligned_four_intervals():
    rr = np.array([1000, 1000, 1100, 1100, 1000, 1000, 1100, 1100], dtype=float)
    value, count = prsa_dc(rr, np.ones(len(rr), dtype=bool), np.ones(len(rr), dtype=bool))
    assert count == 2
    assert value == pytest.approx(50.0)


def test_rhythm_gates_hrv_and_relative_filter():
    peaks = np.cumsum([0, 1000, 1010, 1500, 1000, 1010])
    rr, good, same = filtered_rr(peaks, np.zeros(len(peaks), dtype=int), 1000)
    assert good.tolist() == [True, True, False, False, True]
    for rhythm in ("1", "2", "3", "", "4"):
        output = interval_features(
            peaks, np.zeros(len(peaks), dtype=int), 1000, rhythm, 20, 30, "X"
        )
        assert output["hrv_time_domain_estimates"] is None
        assert output["deceleration_capacity_estimate_ms"] is None
        assert output["quality"]["hrv_status"] == "not_available_non_sinus_or_unknown"
        assert validate_llm_features(output) is output


def test_payload_rejects_patient_identifiers_and_nonfinite_values():
    peaks = np.arange(8) * 1000
    features = interval_features(peaks, np.zeros(len(peaks), dtype=int), 1000, "0", 20, 30, "X")
    assert validate_llm_features(features) is features
    features["recording"]["patient_id"] = "must-not-leave-local"
    with pytest.raises(ValueError):
        validate_llm_features(features)
    del features["recording"]["patient_id"]
    features["heart_rate_bpm"]["mean_from_filtered_rr"] = float("nan")
    with pytest.raises(ValueError):
        validate_llm_features(features)


def test_quality_excludes_augmented_and_test_and_requires_alignment(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text("pid,split,SCD\nP0001,train,0\nP0001-1,train,0\nP0002,val,1\nP0003,test,1\n")
    wave_dir = tmp_path / "waves"
    wave_dir.mkdir()
    for pid in ("P0001", "P0002"):
        np.save(wave_dir / f"{pid}.npy", np.ones((120, 3), dtype=np.float32))
    ref = tmp_path / "reference.csv"
    ref.write_text("patient_id,holter_rhythm\nP0001,0\nP0002,1\n")
    records = select_development(split, wave_dir, ref, None, 42)
    assert [r["patient_id"] for r in records] == ["P0001", "P0002"]
    quality = tmp_path / "quality.csv"
    with quality.open("w", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=["pid", "file", "window_idx", "start_idx", "exclude"]
        )
        writer.writeheader()
        for record in records:
            for i in range(4):
                pid = record["patient_id"]
                writer.writerow(
                    {
                        "pid": pid,
                        "file": f"{pid}.npy",
                        "window_idx": i,
                        "start_idx": i * 30,
                        "exclude": "true" if i == 1 else "false",
                    }
                )
    masks = load_quality_index(quality, records, 1, 30)
    assert masks["P0001"].tolist() == [True, False, True, True]
    quality.write_text("pid,file,window_idx,start_idx,exclude\nP0001,P0001.npy,0,0,false\n")
    with pytest.raises(ValueError, match="누락"):
        load_quality_index(quality, records, 1, 30)


def test_detector_uses_full_recording_segments_and_missing_lead_mask():
    sr = 200
    signal = nk.ecg_simulate(duration=60, sampling_rate=sr, heart_rate=72, random_state=42)
    wave = np.column_stack([signal, signal * 0.8, np.zeros_like(signal)])
    out = detect_peaks_by_lead(
        wave,
        np.array([True, True]),
        sr,
        30,
        [True, True, False],
        chunk_seconds=30,
        context_seconds=2,
    )
    assert out["lead"] in ("X", "Y")
    assert 60 <= len(out["peaks"]) <= 85
    assert out["good"].tolist() == [True, True]
