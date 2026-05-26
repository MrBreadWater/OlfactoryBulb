"""Utilities for saving and loading standard OBGPU result artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
try:  # pragma: no cover - optional speed/quality dependency
    from scipy.signal import find_peaks as _scipy_find_peaks
except ImportError:  # pragma: no cover - setup smoke tests may not have scipy
    _scipy_find_peaks = None

SOMA_TRACE_FORMAT_VERSION = "obgpu_soma_vs_v2"
SOMA_TRACE_QUANTIZED_FORMAT_VERSION = "obgpu_soma_vs_v3"
SOMA_SPIKES_FORMAT_VERSION = "obgpu_soma_spikes_v1"
VOLTAGE_SUMMARY_FORMAT_VERSION = "obgpu_voltage_summary_v1"
DEFAULT_SOMA_TRACE_FORMAT = "npz"
DEFAULT_SOMA_TRACE_DTYPE = "float32"
DEFAULT_SOMA_SPIKE_THRESHOLD_MV = None
DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV = 3.0
DEFAULT_SOMA_SPIKE_REFRACTORY_MS = 1.0
SOMA_TRACE_FILENAME_NPZ = "soma_vs.npz"
SOMA_TRACE_FILENAME_PKL = "soma_vs.pkl"
SOMA_SPIKES_FILENAME_NPZ = "soma_spikes.npz"
VOLTAGE_SUMMARY_FILENAME_NPZ = "voltage_summary.npz"


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


def _normalize_trace_dtype_name(dtype: str | np.dtype | None) -> str:
    """Resolve one configured trace dtype to a canonical storage name."""
    dtype_name = str(dtype or DEFAULT_SOMA_TRACE_DTYPE).strip().lower()
    if dtype_name in {"f4", "float32", "single"}:
        return "float32"
    if dtype_name in {"f8", "float64", "double"}:
        return "float64"
    if dtype_name in {"i2", "int16", "quantized_int16", "linear_int16"}:
        return "int16"
    raise ValueError(f"Unsupported soma trace dtype {dtype!r}")


def _npz_scalar_to_text(value: Any) -> str:
    """Convert one loaded NPZ scalar/string payload to plain text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return str(value)


def adaptive_soma_spike_peak_floor(v: np.ndarray | list[float]) -> float:
    """Estimate a conservative voltage floor for soma spike peaks."""
    finite_v = np.asarray(v, dtype=float)
    finite_v = finite_v[np.isfinite(finite_v)]
    if finite_v.size == 0:
        return np.inf
    baseline = float(np.percentile(finite_v, 5.0))
    upper = float(np.percentile(finite_v, 95.0))
    dynamic_span = max(0.0, upper - baseline)
    return baseline + max(20.0, 0.5 * dynamic_span)


def _fallback_find_peaks(v: np.ndarray, *, distance: int, prominence: float) -> np.ndarray:
    """Small local-maxima fallback when SciPy is unavailable."""
    if len(v) < 3:
        return np.array([], dtype=int)
    candidates = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])) + 1
    if len(candidates) == 0:
        return candidates
    if prominence > 0:
        left = np.maximum(candidates - 1, 0)
        right = np.minimum(candidates + 1, len(v) - 1)
        local_prominence = v[candidates] - np.maximum(v[left], v[right])
        candidates = candidates[local_prominence >= float(prominence)]
    if len(candidates) <= 1 or distance <= 1:
        return candidates

    kept: list[int] = []
    for index in candidates:
        if not kept or index - kept[-1] >= distance:
            kept.append(int(index))
        elif v[index] > v[kept[-1]]:
            kept[-1] = int(index)
    return np.asarray(kept, dtype=int)


def detect_soma_spikes(
    t: np.ndarray | list[float],
    v: np.ndarray | list[float],
    threshold: float | None = DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
    *,
    min_prominence_mv: float = DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
    refractory_ms: float = DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
) -> np.ndarray:
    """Detect soma spike peak times from one voltage trace."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) < 3:
        return np.array([], dtype=float)

    finite_mask = np.isfinite(t) & np.isfinite(v)
    if not np.all(finite_mask):
        t = t[finite_mask]
        v = v[finite_mask]
    if len(t) < 3:
        return np.array([], dtype=float)

    dt_ms = float(np.median(np.diff(t)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        dt_ms = 0.1
    min_distance = max(1, int(round(float(refractory_ms) / dt_ms)))

    if _scipy_find_peaks is not None:
        peaks, _properties = _scipy_find_peaks(
            v,
            prominence=float(min_prominence_mv),
            distance=min_distance,
        )
    else:
        peaks = _fallback_find_peaks(
            v,
            prominence=float(min_prominence_mv),
            distance=min_distance,
        )
    if len(peaks) == 0:
        return np.array([], dtype=float)

    peak_floor = float(threshold) if threshold is not None else adaptive_soma_spike_peak_floor(v)
    keep = v[peaks] >= peak_floor
    return t[peaks[keep]].astype(np.float64, copy=False)


def _quantize_linear_int16(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize one voltage trace with per-trace linear int16 scaling."""
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.int16), 0.0, 1.0

    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        offset = v_min if np.isfinite(v_min) else 0.0
        return np.zeros(arr.shape, dtype=np.int16), offset, 1.0

    scale = (v_max - v_min) / 65535.0
    offset = (v_max + v_min) * 0.5
    quantized = np.rint((arr - offset) / scale)
    quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
    return quantized, float(offset), float(scale)


def _decode_linear_int16(values: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Decode one linear int16 voltage trace to float32 millivolts."""
    return values.astype(np.float32) * np.float32(scale) + np.float32(offset)


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

    dtype_name = _normalize_trace_dtype_name(trace_dtype)
    dtype = np.dtype(np.float32 if dtype_name == "int16" else dtype_name)
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
    format_version = SOMA_TRACE_QUANTIZED_FORMAT_VERSION if dtype_name == "int16" else SOMA_TRACE_FORMAT_VERSION
    value_encoding = "linear_int16" if dtype_name == "int16" else "plain"

    if shared_t:
        time_payload = time_arrays[0] if time_arrays else np.asarray([], dtype=dtype)
        if dtype_name == "int16":
            quantized_rows = []
            offsets = []
            scales = []
            for value_array in value_arrays:
                quantized, offset, scale = _quantize_linear_int16(value_array)
                quantized_rows.append(quantized)
                offsets.append(offset)
                scales.append(scale)
            value_payload = (
                np.stack(quantized_rows)
                if quantized_rows
                else np.empty((0, 0), dtype=np.int16)
            )
            extra_payload = {
                "v_offset": np.asarray(offsets, dtype=np.float32),
                "v_scale": np.asarray(scales, dtype=np.float32),
            }
        else:
            value_payload = (
                np.stack(value_arrays).astype(dtype, copy=False)
                if value_arrays
                else np.empty((0, 0), dtype=dtype)
            )
            extra_payload = {}
        np.savez_compressed(
            result_path,
            format_version=np.asarray(format_version),
            layout=np.asarray("shared_t"),
            v_encoding=np.asarray(value_encoding),
            labels=labels_array,
            t=time_payload,
            v=value_payload,
            **extra_payload,
        )
        return result_path

    lengths = np.asarray([len(values) for values in value_arrays], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    time_payload = np.zeros((len(time_arrays), max_len), dtype=dtype)
    value_payload = np.zeros(
        (len(value_arrays), max_len),
        dtype=np.int16 if dtype_name == "int16" else dtype,
    )
    offsets = []
    scales = []
    for row, (time_array, value_array) in enumerate(zip(time_arrays, value_arrays)):
        count = min(len(time_array), len(value_array))
        if count <= 0:
            if dtype_name == "int16":
                offsets.append(0.0)
                scales.append(1.0)
            continue
        time_payload[row, :count] = time_array[:count]
        if dtype_name == "int16":
            quantized, offset, scale = _quantize_linear_int16(value_array[:count])
            value_payload[row, :count] = quantized
            offsets.append(offset)
            scales.append(scale)
        else:
            value_payload[row, :count] = value_array[:count]

    extra_payload = {}
    if dtype_name == "int16":
        extra_payload = {
            "v_offset": np.asarray(offsets, dtype=np.float32),
            "v_scale": np.asarray(scales, dtype=np.float32),
        }

    np.savez_compressed(
        result_path,
        format_version=np.asarray(format_version),
        layout=np.asarray("ragged"),
        v_encoding=np.asarray(value_encoding),
        labels=labels_array,
        lengths=lengths,
        t=time_payload,
        v=value_payload,
        **extra_payload,
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
        value_encoding = _npz_scalar_to_text(payload["v_encoding"]) if "v_encoding" in payload else "plain"
        labels = [str(label) for label in payload["labels"].tolist()]
        if layout == "shared_t":
            shared_t = payload["t"]
            values = payload["v"]
            if value_encoding == "linear_int16":
                offsets = payload["v_offset"]
                scales = payload["v_scale"]
                return [
                    (label, shared_t, _decode_linear_int16(values[index], offsets[index], scales[index]))
                    for index, label in enumerate(labels)
                ]
            return [(label, shared_t, values[index]) for index, label in enumerate(labels)]
        if layout == "ragged":
            lengths = payload["lengths"]
            times = payload["t"]
            values = payload["v"]
            if value_encoding == "linear_int16":
                offsets = payload["v_offset"]
                scales = payload["v_scale"]
                return [
                    (
                        label,
                        times[index, : int(lengths[index])],
                        _decode_linear_int16(
                            values[index, : int(lengths[index])],
                            offsets[index],
                            scales[index],
                        ),
                    )
                    for index, label in enumerate(labels)
                ]
            return [
                (label, times[index, : int(lengths[index])], values[index, : int(lengths[index])])
                for index, label in enumerate(labels)
            ]
    raise ValueError(f"Unsupported soma trace layout in {path}")


def save_soma_spike_artifact(
    traces: list[tuple[str, Any, Any]],
    path_or_dir: str | Path,
    *,
    threshold: float | None = DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
    min_prominence_mv: float = DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
    refractory_ms: float = DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
) -> Path:
    """Save compact per-soma spike times detected from recorded voltage traces."""
    result_path = Path(path_or_dir)
    if result_path.is_dir() or result_path.suffix == "":
        result_path = result_path / SOMA_SPIKES_FILENAME_NPZ
    result_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    counts: list[int] = []
    offsets: list[int] = [0]
    flat_spikes: list[np.ndarray] = []
    for label, times, values in traces:
        spikes = detect_soma_spikes(
            times,
            values,
            threshold=threshold,
            min_prominence_mv=min_prominence_mv,
            refractory_ms=refractory_ms,
        ).astype(np.float32, copy=False)
        labels.append(str(label))
        counts.append(int(len(spikes)))
        flat_spikes.append(spikes)
        offsets.append(offsets[-1] + int(len(spikes)))

    spike_times = (
        np.concatenate(flat_spikes).astype(np.float32, copy=False)
        if flat_spikes
        else np.asarray([], dtype=np.float32)
    )
    threshold_value = np.nan if threshold is None else float(threshold)
    np.savez_compressed(
        result_path,
        format_version=np.asarray(SOMA_SPIKES_FORMAT_VERSION),
        labels=np.asarray(labels, dtype=str),
        counts=np.asarray(counts, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        spike_times=spike_times,
        threshold_mv=np.asarray(threshold_value, dtype=np.float32),
        threshold_is_adaptive=np.asarray(threshold is None),
        min_prominence_mv=np.asarray(float(min_prominence_mv), dtype=np.float32),
        refractory_ms=np.asarray(float(refractory_ms), dtype=np.float32),
        detector=np.asarray("scipy_find_peaks" if _scipy_find_peaks is not None else "numpy_local_maxima"),
    )
    return result_path


def load_soma_spike_artifact(path_or_dir: str | Path) -> dict[str, Any]:
    """Load compact soma spike times from ``soma_spikes.npz``."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / SOMA_SPIKES_FILENAME_NPZ
    if not path.exists():
        raise FileNotFoundError(f"No soma spike artifact found at {path}")

    with np.load(path, allow_pickle=False) as payload:
        labels = [str(label) for label in payload["labels"].tolist()]
        counts = payload["counts"].astype(np.int32, copy=False)
        offsets = payload["offsets"].astype(np.int64, copy=False)
        flat = payload["spike_times"].astype(np.float32, copy=False)
        threshold_value = float(payload["threshold_mv"])
        threshold = None if bool(payload["threshold_is_adaptive"]) or np.isnan(threshold_value) else threshold_value
        spike_times = [
            flat[int(offsets[index]): int(offsets[index + 1])]
            for index in range(len(labels))
        ]
        return {
            "format_version": _npz_scalar_to_text(payload["format_version"]),
            "labels": labels,
            "counts": counts,
            "spike_times": spike_times,
            "metadata": {
                "threshold_mv": threshold,
                "threshold_is_adaptive": bool(payload["threshold_is_adaptive"]),
                "min_prominence_mv": float(payload["min_prominence_mv"]),
                "refractory_ms": float(payload["refractory_ms"]),
                "detector": _npz_scalar_to_text(payload["detector"]),
            },
        }


def _cell_type_for_label(label: str) -> str:
    for candidate in ("MC", "TC", "GC"):
        if str(label).startswith(candidate):
            return candidate
    return "other"


def save_voltage_summary_artifact(
    traces: list[tuple[str, Any, Any]],
    path_or_dir: str | Path,
    *,
    dtype: str | np.dtype = np.float32,
) -> Path:
    """Save compact per-cell-class voltage moments from recorded soma traces."""
    result_path = Path(path_or_dir)
    if result_path.is_dir() or result_path.suffix == "":
        result_path = result_path / VOLTAGE_SUMMARY_FILENAME_NPZ
    result_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {key: [] for key in ("MC", "TC", "GC", "other")}
    for label, times, values in traces:
        t = np.asarray(times, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        count = min(len(t), len(v))
        if count <= 0:
            continue
        grouped[_cell_type_for_label(str(label))].append((t[:count], v[:count]))

    dtype = np.dtype(dtype)
    cell_types: list[str] = []
    lengths: list[int] = []
    counts: list[int] = []
    t_rows: list[np.ndarray] = []
    moment_rows: dict[str, list[np.ndarray]] = {
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
        "m2": [],
        "m3": [],
        "m4": [],
        "skew": [],
        "kurtosis": [],
    }

    for cell_type in ("MC", "TC", "GC", "other"):
        rows = grouped[cell_type]
        if not rows:
            continue
        ref_t = rows[0][0]
        aligned_values = []
        for t, v in rows:
            if t.shape == ref_t.shape and np.array_equal(t, ref_t):
                aligned_values.append(v)
            else:
                aligned_values.append(np.interp(ref_t, t, v))
        stack = np.vstack(aligned_values)
        mean = np.mean(stack, axis=0)
        centered = stack - mean
        m2 = np.mean(centered ** 2, axis=0)
        m3 = np.mean(centered ** 3, axis=0)
        m4 = np.mean(centered ** 4, axis=0)
        std = np.sqrt(m2)
        safe_std = np.where(std > 0, std, np.nan)
        skew = np.nan_to_num(m3 / (safe_std ** 3), nan=0.0, posinf=0.0, neginf=0.0)
        kurtosis = np.nan_to_num(m4 / (safe_std ** 4), nan=0.0, posinf=0.0, neginf=0.0)

        cell_types.append(cell_type)
        lengths.append(int(len(ref_t)))
        counts.append(int(len(rows)))
        t_rows.append(ref_t.astype(dtype, copy=False))
        moment_rows["mean"].append(mean.astype(dtype, copy=False))
        moment_rows["std"].append(std.astype(dtype, copy=False))
        moment_rows["min"].append(np.min(stack, axis=0).astype(dtype, copy=False))
        moment_rows["max"].append(np.max(stack, axis=0).astype(dtype, copy=False))
        moment_rows["m2"].append(m2.astype(dtype, copy=False))
        moment_rows["m3"].append(m3.astype(dtype, copy=False))
        moment_rows["m4"].append(m4.astype(dtype, copy=False))
        moment_rows["skew"].append(skew.astype(dtype, copy=False))
        moment_rows["kurtosis"].append(kurtosis.astype(dtype, copy=False))

    max_len = max(lengths, default=0)
    row_count = len(cell_types)
    payload: dict[str, Any] = {
        "format_version": np.asarray(VOLTAGE_SUMMARY_FORMAT_VERSION),
        "cell_types": np.asarray(cell_types, dtype=str),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "n_traces": np.asarray(counts, dtype=np.int32),
        "t": np.zeros((row_count, max_len), dtype=dtype),
    }
    for name in moment_rows:
        payload[name] = np.zeros((row_count, max_len), dtype=dtype)

    for row, t in enumerate(t_rows):
        count = int(lengths[row])
        payload["t"][row, :count] = t[:count]
        for name, rows in moment_rows.items():
            payload[name][row, :count] = rows[row][:count]

    np.savez_compressed(result_path, **payload)
    return result_path


def load_voltage_summary_artifact(path_or_dir: str | Path) -> dict[str, Any]:
    """Load compact per-cell-class voltage moments from ``voltage_summary.npz``."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / VOLTAGE_SUMMARY_FILENAME_NPZ
    if not path.exists():
        raise FileNotFoundError(f"No voltage summary artifact found at {path}")

    moment_names = ("mean", "std", "min", "max", "m2", "m3", "m4", "skew", "kurtosis")
    with np.load(path, allow_pickle=False) as payload:
        cell_types = [str(value) for value in payload["cell_types"].tolist()]
        lengths = payload["lengths"].astype(np.int32, copy=False)
        result: dict[str, Any] = {
            "format_version": _npz_scalar_to_text(payload["format_version"]),
            "cell_types": cell_types,
            "n_traces": {cell_type: int(payload["n_traces"][index]) for index, cell_type in enumerate(cell_types)},
            "t_by_type": {},
        }
        for name in moment_names:
            result[f"{name}_by_type"] = {}
        for index, cell_type in enumerate(cell_types):
            count = int(lengths[index])
            result["t_by_type"][cell_type] = payload["t"][index, :count].astype(np.float32, copy=False)
            for name in moment_names:
                result[f"{name}_by_type"][cell_type] = payload[name][index, :count].astype(np.float32, copy=False)
        return result


def load_saved_result_artifact(path: str | Path) -> Any:
    """Load one saved result artifact, supporting both NPZ and pickle encodings."""
    path = Path(path)
    if path.name in {SOMA_TRACE_FILENAME_NPZ, SOMA_TRACE_FILENAME_PKL}:
        return load_soma_trace_artifact(path)
    if path.name == SOMA_SPIKES_FILENAME_NPZ:
        return load_soma_spike_artifact(path)
    if path.name == VOLTAGE_SUMMARY_FILENAME_NPZ:
        return load_voltage_summary_artifact(path)
    with open(path, "rb") as handle:
        return pickle.load(handle)
