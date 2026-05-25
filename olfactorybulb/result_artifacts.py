"""Utilities for saving and loading standard OBGPU result artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

SOMA_TRACE_FORMAT_VERSION = "obgpu_soma_vs_v2"
DEFAULT_SOMA_TRACE_FORMAT = "npz"
DEFAULT_SOMA_TRACE_DTYPE = "float32"
SOMA_TRACE_FILENAME_NPZ = "soma_vs.npz"
SOMA_TRACE_FILENAME_PKL = "soma_vs.pkl"


def soma_trace_artifact_candidates(*, preferred_format: str | None = None) -> tuple[str, ...]:
    """Return candidate soma-trace artifact names in preferred lookup order."""
    preferred = str(preferred_format or DEFAULT_SOMA_TRACE_FORMAT).strip().lower()
    if preferred == "pkl":
        return (SOMA_TRACE_FILENAME_PKL, SOMA_TRACE_FILENAME_NPZ)
    return (SOMA_TRACE_FILENAME_NPZ, SOMA_TRACE_FILENAME_PKL)


def preferred_soma_trace_artifact_name(trace_format: str | None = None) -> str:
    """Return the preferred soma-trace artifact filename for one configured format."""
    return soma_trace_artifact_candidates(preferred_format=trace_format)[0]


def find_soma_trace_artifact(path_or_dir: str | Path, *, preferred_format: str | None = None) -> Path | None:
    """Return the first existing soma-trace artifact under one directory, or the file itself."""
    path = Path(path_or_dir)
    if path.is_file():
        return path if path.exists() else None
    if path.suffix in {".npz", ".pkl"}:
        return path if path.exists() else None
    for filename in soma_trace_artifact_candidates(preferred_format=preferred_format):
        candidate = path / filename
        if candidate.exists():
            return candidate
    return None


def _normalize_trace_dtype(dtype: str | np.dtype | None) -> np.dtype:
    """Resolve one configured trace dtype to a concrete NumPy dtype."""
    dtype_name = str(dtype or DEFAULT_SOMA_TRACE_DTYPE).strip().lower()
    if dtype_name in {"f4", "float32", "single"}:
        return np.dtype(np.float32)
    if dtype_name in {"f8", "float64", "double"}:
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported soma trace dtype {dtype!r}")


def _npz_scalar_to_text(value: Any) -> str:
    """Convert one loaded NPZ scalar/string payload to plain text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return str(value)


def save_soma_trace_artifact(
    traces: list[tuple[str, Any, Any]],
    path_or_dir: str | Path,
    *,
    trace_format: str | None = None,
    trace_dtype: str | np.dtype | None = None,
) -> Path:
    """Save soma traces in the configured artifact format and return the written path."""
    trace_format = str(trace_format or DEFAULT_SOMA_TRACE_FORMAT).strip().lower()
    result_path = Path(path_or_dir)
    if result_path.is_dir():
        result_path = result_path / preferred_soma_trace_artifact_name(trace_format)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if trace_format == "pkl":
        with open(result_path, "wb") as handle:
            pickle.dump(traces, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result_path

    if trace_format != "npz":
        raise ValueError(f"Unsupported soma trace format {trace_format!r}")

    dtype = _normalize_trace_dtype(trace_dtype)
    labels: list[str] = []
    time_arrays: list[np.ndarray] = []
    value_arrays: list[np.ndarray] = []
    shared_t = True

    for label, times, values in traces:
        label_text = str(label)
        time_array = np.asarray(times, dtype=dtype)
        value_array = np.asarray(values, dtype=dtype)
        labels.append(label_text)
        time_arrays.append(time_array)
        value_arrays.append(value_array)
        if len(time_arrays) > 1 and (
            time_array.shape != time_arrays[0].shape or not np.array_equal(time_array, time_arrays[0])
        ):
            shared_t = False

    labels_array = np.asarray(labels, dtype=str)
    if shared_t:
        time_payload = time_arrays[0] if time_arrays else np.asarray([], dtype=dtype)
        value_payload = (
            np.stack(value_arrays).astype(dtype, copy=False)
            if value_arrays
            else np.empty((0, 0), dtype=dtype)
        )
        np.savez_compressed(
            result_path,
            format_version=np.asarray(SOMA_TRACE_FORMAT_VERSION),
            layout=np.asarray("shared_t"),
            labels=labels_array,
            t=time_payload,
            v=value_payload,
        )
        return result_path

    lengths = np.asarray([len(values) for values in value_arrays], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    time_payload = np.zeros((len(time_arrays), max_len), dtype=dtype)
    value_payload = np.zeros((len(value_arrays), max_len), dtype=dtype)
    for row, (time_array, value_array) in enumerate(zip(time_arrays, value_arrays)):
        count = min(len(time_array), len(value_array))
        if count <= 0:
            continue
        time_payload[row, :count] = time_array[:count]
        value_payload[row, :count] = value_array[:count]

    np.savez_compressed(
        result_path,
        format_version=np.asarray(SOMA_TRACE_FORMAT_VERSION),
        layout=np.asarray("ragged"),
        labels=labels_array,
        lengths=lengths,
        t=time_payload,
        v=value_payload,
    )
    return result_path


def load_soma_trace_artifact(path_or_dir: str | Path) -> list[tuple[str, Any, Any]]:
    """Load one soma-trace artifact and return the legacy notebook tuple structure."""
    path = find_soma_trace_artifact(path_or_dir)
    if path is None:
        raise FileNotFoundError(f"No soma trace artifact found near {path_or_dir}")
    if path.suffix != ".npz":
        with open(path, "rb") as handle:
            return pickle.load(handle)

    with np.load(path, allow_pickle=False) as payload:
        layout = _npz_scalar_to_text(payload["layout"])
        labels = [str(label) for label in payload["labels"].tolist()]
        if layout == "shared_t":
            shared_t = payload["t"]
            values = payload["v"]
            return [(label, shared_t, values[index]) for index, label in enumerate(labels)]
        if layout == "ragged":
            lengths = payload["lengths"]
            times = payload["t"]
            values = payload["v"]
            return [
                (label, times[index, : int(lengths[index])], values[index, : int(lengths[index])])
                for index, label in enumerate(labels)
            ]
    raise ValueError(f"Unsupported soma trace layout in {path}")


def load_saved_result_artifact(path: str | Path) -> Any:
    """Load one saved result artifact, supporting both NPZ and pickle encodings."""
    path = Path(path)
    if path.name in {SOMA_TRACE_FILENAME_NPZ, SOMA_TRACE_FILENAME_PKL}:
        return load_soma_trace_artifact(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)
