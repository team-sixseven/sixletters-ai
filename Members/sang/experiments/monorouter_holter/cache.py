"""Private, invalidated cache of locally prepared MonoRouter ECG features."""

from __future__ import annotations

import hashlib
import json
import marshal
import math
import os
import stat
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from Members.sang.model.config import resolve_data_path
from Members.sang.model.DL.dataloader.segmented_ecg import read_patient_split

FEATURE_KEYS = {
    "sample_rate_hz",
    "segment_seconds",
    "segments_sampled",
    "measured_leads",
    "signal_units",
}
METRIC_KEYS = {"mean", "std", "rms", "p95_minus_p05", "successive_difference_rms"}
CACHE_VERSION = 1


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_code(code: types.CodeType) -> types.CodeType:
    constants = tuple(
        _normalized_code(c) if isinstance(c, types.CodeType) else c for c in code.co_consts
    )
    return code.replace(co_filename="", co_firstlineno=1, co_linetable=b"", co_consts=constants)


def cache_identity(repo: Path, basic: dict, basic_path: Path, settings: dict, feature_fn):
    """Fingerprint inputs and implementations without reading all waveform values."""
    cfg = basic["data"]["raw_ecg"]
    split_path = resolve_data_path(cfg["split_path"], basic_path)
    wave_dir = resolve_data_path(cfg["wave_directory"], basic_path)
    split = read_patient_split(split_path)
    sources = {}
    for key in ("split_path", "quality_index_path", "lead_metadata_path"):
        if cfg.get(key):
            path = resolve_data_path(cfg[key], basic_path)
            sources[key] = {"path": str(path), "sha256": _hash_file(path)}
        else:
            sources[key] = None
    wave_hash = hashlib.sha256()
    for pid, _, _ in split:
        path = wave_dir / f"{pid}.npy"
        try:
            s = path.stat()
            fields = (pid, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
        except FileNotFoundError:
            fields = (pid, None)
        wave_hash.update(json.dumps(fields, separators=(",", ":")).encode() + b"\n")
    code_paths = [
        repo / "Members/sang/model/config.py",
        repo / "Members/sang/model/DL/dataloader/raw_waveform.py",
        repo / "Members/sang/model/DL/dataloader/segmented_ecg.py",
        Path(__file__),
    ]
    descriptor = {
        "cache_version": CACHE_VERSION,
        "settings": settings,
        "raw": cfg,
        "basic_path": str(basic_path.resolve()),
        "sources": sources,
        "wave_stat_sha256": wave_hash.hexdigest(),
        "feature_code_sha256": hashlib.sha256(
            marshal.dumps(_normalized_code(feature_fn.__code__))
        ).hexdigest(),
        "implementation_sha256": [_hash_file(p) for p in code_paths],
        "python": list(sys.version_info[:3]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    encoded = json.dumps(
        descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), split


def _check_features(value: dict, basic: dict, settings: dict) -> bool:
    if not isinstance(value, dict) or set(value) != FEATURE_KEYS:
        return False
    raw = basic["data"]["raw_ecg"]
    if (
        value["sample_rate_hz"] != int(raw["sample_rate_hz"])
        or value["segment_seconds"] != int(raw["segment_seconds"])
        or type(value["segments_sampled"]) is not int
        or not 1 <= value["segments_sampled"] <= settings["max_segments"]
        or value["signal_units"] != "same units as the input waveform; physical units unconfirmed"
    ):
        return False
    leads = value["measured_leads"]
    if not isinstance(leads, list) or not 1 <= len(leads) <= 3:
        return False
    names = []
    for lead in leads:
        if not isinstance(lead, dict) or set(lead) != {"lead", "statistics_across_segments"}:
            return False
        names.append(lead["lead"])
        metrics = lead["statistics_across_segments"]
        if (
            lead["lead"] not in {"X", "Y", "Z"}
            or not isinstance(metrics, dict)
            or set(metrics) != METRIC_KEYS
        ):
            return False
        for summary in metrics.values():
            if not isinstance(summary, dict) or set(summary) != {"median", "p10", "p90"}:
                return False
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in summary.values()):
                return False
            if not summary["p10"] <= summary["median"] <= summary["p90"]:
                return False
    return len(names) == len(set(names))


def _validate_records(records, split, basic, settings):
    if not isinstance(records, list) or not records:
        raise ValueError("캐시 기록 형식 오류")
    expected = {pid: (group, label) for pid, group, label in split}
    if settings["sample_size"] is not None and len(records) != settings["sample_size"]:
        raise ValueError("캐시 표본 크기 불일치")
    ids = set()
    prepared = []
    for item in records:
        if not isinstance(item, dict) or set(item) != {"pid", "target", "features"}:
            raise ValueError("캐시 기록 필드 오류")
        pid, target = item["pid"], item["target"]
        if (
            not isinstance(pid, str)
            or type(target) is not int
            or expected.get(pid) != (settings["split"], target)
            or pid in ids
        ):
            raise ValueError("캐시 환자 분할 또는 라벨 불일치")
        if not _check_features(item["features"], basic, settings):
            raise ValueError("캐시 특징 형식 오류")
        ids.add(pid)
        prepared.append((pid, target, item["features"]))
    return prepared


def _cache_dir(repo: Path) -> Path:
    path = repo / "Members/sang/results/monorouter_holter/cache_v1"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PermissionError("캐시 폴더는 본인 소유이고 다른 사용자에게 닫혀 있어야 합니다.")
    return path


def load_prepared(repo: Path, identity: str, split, basic: dict, settings: dict):
    path = _cache_dir(repo) / f"{identity}.json"
    if not path.exists():
        return None
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PermissionError("캐시 파일은 본인 소유의 비공개 파일이어야 합니다.")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(obj, dict)
            or obj.get("identity") != identity
            or obj.get("version") != CACHE_VERSION
        ):
            return None
        return _validate_records(obj.get("records"), split, basic, settings)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        print("기존 캐시를 읽을 수 없어 입력 데이터를 다시 검사합니다.")
        return None


def save_prepared(repo: Path, identity: str, records, split, basic: dict, settings: dict):
    items = [
        {"pid": pid, "target": target, "features": features} for pid, target, features in records
    ]
    _validate_records(items, split, basic, settings)
    directory = _cache_dir(repo)
    path = directory / f"{identity}.json"
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(
                {"version": CACHE_VERSION, "identity": identity, "records": items},
                out,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path
