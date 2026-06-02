"""Reusable event-series analysis and raster plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .spectral import normalize_time_modulus


@dataclass(frozen=True)
class EventRateTrace:
    """One named event-rate series prepared for plotting."""

    base_label: str
    times_ms: np.ndarray | list[float]
    rate_hz: np.ndarray | list[float]
    metadata: dict[str, Any]
    color: Any = "black"


@dataclass(frozen=True)
class EventRateSeriesSpec:
    """One requested event-rate subset to compute and render."""

    base_label: str
    selection: Any = None
    color: Any = "black"


@dataclass(frozen=True)
class PreparedEventRows:
    """Prepared labeled event rows plus derived display metadata."""

    rows: tuple[tuple[str, np.ndarray], ...]
    max_label_length: int


@dataclass(frozen=True)
class EventOverviewLayout:
    """Layout hints for a raster-plus-rate overview figure."""

    n_rows: int
    label_fontsize: float
    line_spacing: float
    raster_height: float
    rate_height: float
    total_height: float
    left_margin: float


@dataclass(frozen=True)
class FrequencySampleCollection:
    """Instantaneous frequency samples collected from labeled event rows."""

    times_ms: np.ndarray
    freqs_hz: np.ndarray
    labels: tuple[str, ...]
    rows: tuple[Any, ...]


@dataclass(frozen=True)
class EventRateNormalizationRule:
    """One supported normalization mode for event-rate computation."""

    unit: str
    aliases: tuple[str, ...]
    denominator_fn: Callable[[Sequence[Any]], float]
    metadata_fn: Callable[[Sequence[Any]], dict[str, Any]] = lambda _rows: {}


@dataclass(frozen=True)
class ResultEventFamilySpec:
    """One reusable event-family definition backed by a loaded result mapping."""

    rows_from_result_fn: Callable[[dict[str, Any]], Sequence[Any]]
    filter_label_fn: Callable[[Any], str]
    times_fn: Callable[[Any], np.ndarray | list[float]]
    sample_label_fn: Callable[[Any], str] | None = None
    normalize_label_fn: Callable[[str], str] | None = None
    normalization_rules: Mapping[str, EventRateNormalizationRule] | None = None
    default_normalization: str = "total"


@dataclass(frozen=True)
class ResultEventFamilySuite:
    """Behavioral wrapper around one reusable result-backed event family."""

    spec: ResultEventFamilySpec
    infer_t_stop_fn: Callable[[dict[str, Any], Sequence[Any]], float] | None = None

    def filter_rows(
        self,
        result: dict[str, Any],
        *,
        include_prefixes: Sequence[str] | None = None,
    ) -> list[Any]:
        """Filter rows for this family by normalized label prefixes."""
        return filter_result_event_family_rows(
            result,
            self.spec,
            include_prefixes=include_prefixes,
        )

    def collect_samples(
        self,
        result: dict[str, Any],
        *,
        indices: Sequence[int] | range | None = None,
        include_prefixes: Sequence[str] | None = None,
        modulus: float | int | None = None,
    ) -> FrequencySampleCollection:
        """Collect instantaneous frequency samples for this family."""
        return collect_result_event_family_samples(
            result,
            self.spec,
            indices=indices,
            include_prefixes=include_prefixes,
            modulus=modulus,
        )

    def compute_rate(
        self,
        result: dict[str, Any],
        *,
        bin_ms: float,
        smooth_sigma_ms: float,
        include_prefixes: Sequence[str] | None = None,
        normalization: str | None = None,
        return_metadata: bool = False,
        t_stop: float | None = None,
    ) -> Any:
        """Compute one normalized event-rate trace for this family."""
        resolved_t_stop = float(t_stop) if t_stop is not None else self._infer_t_stop(
            result,
            include_prefixes=include_prefixes,
        )
        return compute_result_event_family_rate(
            result,
            self.spec,
            t_stop=resolved_t_stop,
            bin_ms=bin_ms,
            smooth_sigma_ms=smooth_sigma_ms,
            include_prefixes=include_prefixes,
            normalization=normalization,
            return_metadata=return_metadata,
        )

    def _infer_t_stop(
        self,
        result: dict[str, Any],
        *,
        include_prefixes: Sequence[str] | None = None,
    ) -> float:
        """Infer the event-family time span when no explicit t_stop is provided."""
        if self.infer_t_stop_fn is None:
            raise ValueError("Result event family suite does not define t_stop inference")
        rows = self.filter_rows(result, include_prefixes=include_prefixes)
        return float(self.infer_t_stop_fn(result, rows))


def calculate_event_frequency(times: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert event times into midpoint/frequency samples."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.array([]), np.array([])
    t_freq = (times[:-1] + times[1:]) / 2.0
    event_hz = 1000.0 / np.diff(times)
    return t_freq, event_hz


def collect_frequency_samples_from_rows(
    rows: Sequence[Any],
    *,
    label_fn: Callable[[Any], str],
    times_fn: Callable[[Any], np.ndarray | list[float]],
    indices: Sequence[int] | range | None = None,
    include_prefixes: Sequence[str] | None = None,
    modulus: float | int | None = None,
) -> FrequencySampleCollection:
    """Collect instantaneous frequency samples from labeled event-time rows."""
    row_sequence = list(rows)
    prefixes = tuple(str(name) for name in include_prefixes) if include_prefixes else None
    selected_indices: Sequence[int] | range
    if indices is None:
        selected_indices = range(len(row_sequence))
    else:
        selected_indices = indices

    modulus_value = normalize_time_modulus(modulus)
    all_freq_t: list[np.ndarray] = []
    all_freq: list[np.ndarray] = []
    labels: list[str] = []
    selected_rows: list[Any] = []

    for index in selected_indices:
        if int(index) >= len(row_sequence):
            break
        row = row_sequence[int(index)]
        label = str(label_fn(row))
        if prefixes is not None and not any(label.startswith(prefix) for prefix in prefixes):
            continue
        t_freq, event_hz = calculate_event_frequency(times_fn(row))
        if len(t_freq) == 0:
            continue
        t_freq = np.asarray(t_freq, dtype=float)
        if modulus_value is not None:
            t_freq = np.mod(t_freq, modulus_value)
        all_freq_t.append(t_freq)
        all_freq.append(np.asarray(event_hz, dtype=float))
        labels.append(label)
        selected_rows.append(row)

    if all_freq_t:
        times = np.concatenate(all_freq_t)
        freqs = np.concatenate(all_freq)
    else:
        times = np.array([], dtype=float)
        freqs = np.array([], dtype=float)

    return FrequencySampleCollection(
        times_ms=times,
        freqs_hz=freqs,
        labels=tuple(labels),
        rows=tuple(selected_rows),
    )


def calculate_trace_event_frequency(
    t: np.ndarray | list[float],
    values: np.ndarray | list[float],
    *,
    event_times_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive event times from one continuous trace and convert them to frequency samples."""
    event_times = event_times_fn(np.asarray(t, dtype=float), np.asarray(values, dtype=float))
    return calculate_event_frequency(event_times)


def collect_frequency_samples_from_trace_rows(
    rows: Sequence[Any],
    *,
    label_fn: Callable[[Any], str],
    time_fn: Callable[[Any], np.ndarray | list[float]],
    value_fn: Callable[[Any], np.ndarray | list[float]],
    event_times_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | list[float]],
    indices: Sequence[int] | range | None = None,
    include_prefixes: Sequence[str] | None = None,
    modulus: float | int | None = None,
) -> FrequencySampleCollection:
    """Collect instantaneous frequency samples from labeled continuous-trace rows."""

    def _row_event_times(row: Any) -> np.ndarray:
        times = np.asarray(time_fn(row), dtype=float)
        values = np.asarray(value_fn(row), dtype=float)
        return np.asarray(event_times_fn(times, values), dtype=float)

    return collect_frequency_samples_from_rows(
        rows,
        label_fn=label_fn,
        times_fn=_row_event_times,
        indices=indices,
        include_prefixes=include_prefixes,
        modulus=modulus,
    )


def filter_rows_by_label_prefix(
    rows: Sequence[Any],
    *,
    label_fn: Callable[[Any], str],
    include_prefixes: Sequence[str] | None = None,
    normalize_label_fn: Callable[[str], str] | None = None,
) -> list[Any]:
    """Filter rows whose normalized labels start with one of the requested prefixes."""
    if not include_prefixes:
        return list(rows)

    prefixes = tuple(str(name) for name in include_prefixes)
    normalizer = normalize_label_fn or (lambda label: label)
    filtered = []
    for row in rows:
        label = normalizer(str(label_fn(row)))
        if any(label.startswith(prefix) for prefix in prefixes):
            filtered.append(row)
    return filtered


def _canonical_event_rate_normalization(
    normalization: str | None,
    *,
    default: str,
    rules: Mapping[str, EventRateNormalizationRule],
) -> tuple[str, EventRateNormalizationRule]:
    requested = str(normalization or default)
    for canonical_name, rule in rules.items():
        if requested == canonical_name or requested in rule.aliases:
            return canonical_name, rule
    raise ValueError(f"Unsupported event normalization mode {requested!r}")


def compute_event_rate_from_rows(
    rows: Sequence[Any],
    *,
    times_fn: Callable[[Any], np.ndarray | list[float]],
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    normalization: str | None,
    default_normalization: str,
    normalization_rules: Mapping[str, EventRateNormalizationRule],
    return_metadata: bool = False,
) -> Any:
    """Compute a normalized event-rate trace from arbitrary event rows."""
    canonical_name, normalization_rule = _canonical_event_rate_normalization(
        normalization,
        default=default_normalization,
        rules=normalization_rules,
    )
    event_series = [np.asarray(times_fn(row), dtype=float) for row in rows]
    denominator = normalization_rule.denominator_fn(rows)
    centers, rate_hz = binned_event_rate(
        event_series,
        t_stop=t_stop,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        denominator=denominator,
    )
    if not return_metadata:
        return centers, rate_hz
    metadata = dict(normalization_rule.metadata_fn(rows))
    metadata.update(
        {
            "normalization": canonical_name,
            "unit": normalization_rule.unit,
            "denominator": max(float(denominator), 1.0),
        }
    )
    return centers, rate_hz, metadata


def build_event_rate_trace_series(
    result: dict[str, Any],
    series_specs: Sequence[EventRateSeriesSpec],
    *,
    compute_rate_fn: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    selection_kwarg: str = "target_types",
    compute_rate_kwargs: Mapping[str, Any] | None = None,
) -> list[EventRateTrace]:
    """Build plotted event-rate traces for a family of named subsets."""
    shared_kwargs = dict(compute_rate_kwargs or {})
    shared_kwargs.pop("return_metadata", None)

    traces: list[EventRateTrace] = []
    for spec in series_specs:
        current_kwargs = dict(shared_kwargs)
        if selection_kwarg:
            current_kwargs[selection_kwarg] = spec.selection
        times_ms, rate_hz, metadata = compute_rate_fn(
            result,
            return_metadata=True,
            **current_kwargs,
        )
        traces.append(
            EventRateTrace(
                base_label=str(spec.base_label),
                times_ms=times_ms,
                rate_hz=rate_hz,
                metadata=dict(metadata),
                color=spec.color,
            )
        )
    return traces


def filter_result_event_family_rows(
    result: dict[str, Any],
    family_spec: ResultEventFamilySpec,
    *,
    include_prefixes: Sequence[str] | None = None,
) -> list[Any]:
    """Filter one result-backed event family by normalized label prefixes."""
    return filter_rows_by_label_prefix(
        list(family_spec.rows_from_result_fn(result)),
        label_fn=family_spec.filter_label_fn,
        include_prefixes=include_prefixes,
        normalize_label_fn=family_spec.normalize_label_fn,
    )


def collect_result_event_family_samples(
    result: dict[str, Any],
    family_spec: ResultEventFamilySpec,
    *,
    indices: Sequence[int] | range | None = None,
    include_prefixes: Sequence[str] | None = None,
    modulus: float | int | None = None,
) -> FrequencySampleCollection:
    """Collect frequency samples from one reusable result-backed event family."""
    rows = filter_result_event_family_rows(
        result,
        family_spec,
        include_prefixes=include_prefixes,
    )
    return collect_frequency_samples_from_rows(
        rows,
        label_fn=family_spec.sample_label_fn or family_spec.filter_label_fn,
        times_fn=family_spec.times_fn,
        indices=indices,
        modulus=modulus,
    )


def compute_result_event_family_rate(
    result: dict[str, Any],
    family_spec: ResultEventFamilySpec,
    *,
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    include_prefixes: Sequence[str] | None = None,
    normalization: str | None = None,
    return_metadata: bool = False,
) -> Any:
    """Compute one normalized event-rate trace from a reusable result-backed event family."""
    if family_spec.normalization_rules is None:
        raise ValueError("Result event family spec does not define normalization rules")
    rows = filter_result_event_family_rows(
        result,
        family_spec,
        include_prefixes=include_prefixes,
    )
    return compute_event_rate_from_rows(
        rows,
        times_fn=family_spec.times_fn,
        t_stop=t_stop,
        bin_ms=bin_ms,
        smooth_sigma_ms=smooth_sigma_ms,
        normalization=normalization,
        default_normalization=family_spec.default_normalization,
        normalization_rules=family_spec.normalization_rules,
        return_metadata=return_metadata,
    )


def prepare_event_display_rows(
    rows: Sequence[Any],
    *,
    label_fn: Callable[[Any], str],
    times_fn: Callable[[Any], np.ndarray | list[float]],
    sort_key_fn: Callable[[Any], Any] | None = None,
    limit: int | None = None,
    label_transform_fn: Callable[[str], str] | None = None,
) -> PreparedEventRows:
    """Prepare labeled event rows for raster/overview display."""
    row_sequence = list(rows)
    if sort_key_fn is not None:
        row_sequence = sorted(row_sequence, key=sort_key_fn)
    if limit is not None:
        row_sequence = row_sequence[: max(int(limit), 0)]

    transformer = label_transform_fn or (lambda label: label)
    prepared_rows: list[tuple[str, np.ndarray]] = []
    max_label_length = 0
    for row in row_sequence:
        label = str(transformer(str(label_fn(row))))
        times = np.asarray(times_fn(row), dtype=float)
        prepared_rows.append((label, times))
        max_label_length = max(max_label_length, len(label))

    return PreparedEventRows(
        rows=tuple(prepared_rows),
        max_label_length=int(max_label_length),
    )


def smooth_rate_series(
    rate_hz: np.ndarray,
    *,
    bin_ms: float,
    smooth_sigma_ms: float,
) -> np.ndarray:
    """Gaussian-smooth a binned rate trace."""
    if smooth_sigma_ms and smooth_sigma_ms > 0:
        sigma_bins = float(smooth_sigma_ms) / float(bin_ms)
        radius = max(1, int(round(4.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        smoothed = np.convolve(rate_hz, kernel, mode="same")
        if smoothed.shape != rate_hz.shape:
            extra = smoothed.shape[0] - rate_hz.shape[0]
            start = max(extra // 2, 0)
            stop = start + rate_hz.shape[0]
            smoothed = smoothed[start:stop]
        rate_hz = smoothed
    return rate_hz


def binned_event_rate(
    event_series: Sequence[np.ndarray | list[float]],
    *,
    t_stop: float,
    bin_ms: float,
    smooth_sigma_ms: float,
    denominator: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin one or more event series into a smoothed rate trace."""
    if t_stop <= 0.0:
        return np.array([]), np.array([])

    edges = np.arange(0.0, t_stop + float(bin_ms), float(bin_ms))
    if edges.size < 2:
        edges = np.array([0.0, float(bin_ms)], dtype=float)

    flat_times = []
    for times in event_series:
        times = np.asarray(times, dtype=float)
        if times.size:
            flat_times.append(times)

    if flat_times:
        counts, _edges = np.histogram(np.concatenate(flat_times), bins=edges)
    else:
        counts = np.zeros(len(edges) - 1, dtype=float)

    rate_hz = counts.astype(float) / (float(bin_ms) / 1000.0)
    denom = max(float(denominator), 1.0)
    rate_hz /= denom
    rate_hz = smooth_rate_series(rate_hz, bin_ms=bin_ms, smooth_sigma_ms=smooth_sigma_ms)
    centers = edges[:-1] + float(bin_ms) / 2.0
    return centers, rate_hz


def rate_series_label(base_label: str, metadata: dict[str, Any]) -> str:
    """Append denominator information to a plotted rate-series label."""
    normalization = str(metadata.get("normalization", ""))
    if normalization == "per_target_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization == "per_source_cell":
        return f"{base_label} (n={metadata.get('n_source_cells', 0)} sources)"
    if normalization == "per_connection":
        return f"{base_label} (n={metadata.get('n_connections', 0)} connections)"
    if normalization == "per_cell":
        return f"{base_label} (n={metadata.get('n_target_cells', 0)} cells)"
    if normalization in {"per_segment", "per_input_segment"}:
        return f"{base_label} (n={metadata.get('n_segments', 0)} segments)"
    return base_label


def recommended_raster_fontsize(n_rows: int, *, default: float = 7.0) -> float:
    """Choose a compact but readable y-label font size for dense rasters."""
    if n_rows >= 140:
        return 5.0
    if n_rows >= 80:
        return 6.0
    return float(default)


def recommended_raster_height(n_rows: int, *, min_height: float = 4.0) -> float:
    """Estimate a reasonable figure height for a raster plot."""
    if n_rows <= 0:
        return float(min_height)
    return max(float(min_height), 0.06 * float(n_rows) + 1.5)


def recommended_raster_line_spacing(
    n_rows: int,
    *,
    threshold: int = 80,
    dense_value: float = 1.6,
    default: float = 1.4,
) -> float:
    """Pick a slightly wider line spacing for dense rasters."""
    return float(dense_value if int(n_rows) > int(threshold) else default)


def overview_left_margin(
    max_label_len: int,
    *,
    min_margin: float = 0.22,
    max_margin: float = 0.5,
    base: float = 0.15,
    per_char: float = 0.006,
) -> float:
    """Estimate a figure left margin from the longest raster label."""
    return min(float(max_margin), max(float(min_margin), float(base) + float(per_char) * float(max_label_len)))


def build_event_overview_layout(
    *,
    n_rows: int,
    max_label_len: int,
    raster_min_height: float = 4.5,
    rate_height: float = 4.0,
    default_label_fontsize: float = 7.0,
    dense_line_spacing_threshold: int = 80,
    dense_line_spacing: float = 1.6,
    default_line_spacing: float = 1.4,
    left_margin_min: float = 0.22,
    left_margin_max: float = 0.5,
    left_margin_base: float = 0.15,
    left_margin_per_char: float = 0.006,
) -> EventOverviewLayout:
    """Build shared layout hints for a raster-plus-rate overview figure."""
    label_fontsize = recommended_raster_fontsize(n_rows, default=default_label_fontsize)
    line_spacing = recommended_raster_line_spacing(
        n_rows,
        threshold=dense_line_spacing_threshold,
        dense_value=dense_line_spacing,
        default=default_line_spacing,
    )
    raster_height = recommended_raster_height(n_rows, min_height=raster_min_height)
    left_margin = overview_left_margin(
        max_label_len,
        min_margin=left_margin_min,
        max_margin=left_margin_max,
        base=left_margin_base,
        per_char=left_margin_per_char,
    )
    return EventOverviewLayout(
        n_rows=int(n_rows),
        label_fontsize=float(label_fontsize),
        line_spacing=float(line_spacing),
        raster_height=float(raster_height),
        rate_height=float(rate_height),
        total_height=float(raster_height + rate_height),
        left_margin=float(left_margin),
    )


def build_event_overview_layout_for_rows(
    prepared_rows: PreparedEventRows | Sequence[tuple[str, np.ndarray | list[float]]],
    *,
    raster_min_height: float = 4.5,
    rate_height: float = 4.0,
    default_label_fontsize: float = 7.0,
    dense_line_spacing_threshold: int = 80,
    dense_line_spacing: float = 1.6,
    default_line_spacing: float = 1.4,
    left_margin_min: float = 0.22,
    left_margin_max: float = 0.5,
    left_margin_base: float = 0.15,
    left_margin_per_char: float = 0.006,
) -> EventOverviewLayout:
    """Build overview layout hints from prepared or already-labeled event rows."""
    if isinstance(prepared_rows, PreparedEventRows):
        n_rows = len(prepared_rows.rows)
        max_label_len = int(prepared_rows.max_label_length)
    else:
        row_sequence = list(prepared_rows)
        n_rows = len(row_sequence)
        max_label_len = max((len(str(label)) for label, _times in row_sequence), default=0)

    return build_event_overview_layout(
        n_rows=n_rows,
        max_label_len=max_label_len,
        raster_min_height=raster_min_height,
        rate_height=rate_height,
        default_label_fontsize=default_label_fontsize,
        dense_line_spacing_threshold=dense_line_spacing_threshold,
        dense_line_spacing=dense_line_spacing,
        default_line_spacing=default_line_spacing,
        left_margin_min=left_margin_min,
        left_margin_max=left_margin_max,
        left_margin_base=left_margin_base,
        left_margin_per_char=left_margin_per_char,
    )


def ensure_raster_axis(
    ax: Any,
    n_rows: int,
    *,
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.22,
) -> Any:
    """Create a raster axis sized to the current row count when needed."""
    if ax is None:
        height = max(min_height, per_row_height * max(int(n_rows), 1) + 1.0)
        _fig, ax = plt.subplots(figsize=(width, height))
    return ax


def style_raster_axis(
    ax: Any,
    labels: list[str],
    *,
    ylabel: str,
    title: str,
    fontsize: float = 7.0,
    line_spacing: float = 1.4,
) -> np.ndarray:
    """Apply shared styling and row offsets to a raster axis."""
    n_rows = len(labels)
    offsets = np.arange(n_rows, dtype=float) * float(line_spacing)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontsize=fontsize)
    if n_rows:
        pad = max(0.7, line_spacing)
        ax.set_ylim(offsets[0] - pad, offsets[-1] + pad)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return offsets


def fit_raster_labels(
    ax: Any,
    offsets: np.ndarray,
    *,
    min_fontsize: float = 4.5,
    target_ratio: float = 0.9,
    min_height: float = 4.0,
    max_iter: int = 8,
) -> Any:
    """Shrink labels or grow the figure until label height fits the row spacing."""
    if len(offsets) < 2:
        return ax

    fig = ax.figure
    labels = [label for label in ax.get_yticklabels() if label.get_text()]
    if not labels:
        return ax

    for _ in range(max_iter):
        fig.canvas.draw()
        labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if not labels:
            return ax

        renderer = fig.canvas.get_renderer()
        max_label_height_px = max(label.get_window_extent(renderer=renderer).height for label in labels)
        p0 = ax.transData.transform((0.0, float(offsets[0])))[1]
        p1 = ax.transData.transform((0.0, float(offsets[1])))[1]
        spacing_px = abs(float(p1 - p0))
        if spacing_px <= 0:
            return ax

        ratio = max_label_height_px / spacing_px
        if ratio > target_ratio:
            current_font = labels[0].get_fontsize()
            if current_font > min_fontsize + 0.05:
                scale = max(target_ratio / ratio * 0.98, min_fontsize / current_font)
                new_font = max(min_fontsize, current_font * scale)
                for label in labels:
                    label.set_fontsize(new_font)
                continue

            width, height = fig.get_size_inches()
            new_height = max(float(min_height), height * (ratio / target_ratio) * 1.02)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        if ratio < target_ratio * 0.65:
            width, height = fig.get_size_inches()
            shrink = max(ratio / target_ratio, 0.75)
            new_height = max(float(min_height), height * shrink)
            if abs(new_height - height) < 0.05:
                break
            fig.set_size_inches(width, new_height, forward=True)
            continue

        break

    return ax


def plot_event_raster_rows(
    rows: Sequence[tuple[str, np.ndarray | list[float]]],
    *,
    ax: Any = None,
    ylabel: str = "Row",
    title: str = "Event Raster",
    width: float = 14.0,
    min_height: float = 4.0,
    per_row_height: float = 0.10,
    fontsize: float | None = None,
    line_spacing: float = 1.4,
    modulus: float | int | None = None,
    colors: Sequence[Any] | Any = "black",
    linelengths: float = 1.0,
    no_data_message: str = "No events saved",
) -> Any:
    """Plot a generic event raster from labeled event-time rows."""
    ax = ensure_raster_axis(
        ax,
        len(rows),
        width=width,
        min_height=min_height,
        per_row_height=per_row_height,
    )
    if not rows:
        ax.set_title(no_data_message)
        return ax

    modulus_value = normalize_time_modulus(modulus)
    times = [
        np.mod(np.asarray(times, dtype=float), modulus_value)
        if modulus_value is not None
        else np.asarray(times, dtype=float)
        for _label, times in rows
    ]
    labels = [str(label) for label, _times in rows]
    font_value = recommended_raster_fontsize(len(rows)) if fontsize is None else min(
        float(fontsize),
        recommended_raster_fontsize(len(rows), default=float(fontsize)),
    )
    offsets = style_raster_axis(
        ax,
        labels,
        ylabel=ylabel,
        title=title,
        fontsize=font_value,
        line_spacing=line_spacing,
    )
    ax.eventplot(times, lineoffsets=offsets, linelengths=linelengths, colors=colors)
    if modulus_value is not None:
        ax.set_xlim(0.0, modulus_value)
        ax.set_xlabel(f"Time modulo {modulus_value:g} ms")
    fit_raster_labels(ax, offsets, min_height=min_height)
    return ax


def plot_event_rate_traces(
    traces: Sequence[EventRateTrace],
    *,
    ax: Any = None,
    title: str = "Event Rate",
    ylabel_fallback: str = "events/s",
    xlabel: str = "Time (ms)",
    linewidth: float = 1.2,
    legend_loc: str = "upper right",
    no_data_message: str = "No events saved",
    label_formatter: Callable[[str, dict[str, Any]], str] = rate_series_label,
) -> Any:
    """Plot one or more event-rate traces with consistent label handling."""
    ax = ax or plt.subplots(figsize=(14, 4))[1]
    plotted = False
    ylabel = None
    for trace in traces:
        times = np.asarray(trace.times_ms, dtype=float)
        rate = np.asarray(trace.rate_hz, dtype=float)
        if len(times) == 0 or len(rate) == 0:
            continue
        ylabel = str(trace.metadata.get("unit", ylabel_fallback))
        ax.plot(
            times,
            rate,
            color=trace.color,
            linewidth=linewidth,
            label=label_formatter(trace.base_label, trace.metadata),
        )
        plotted = True

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ylabel_fallback)
    ax.set_title(title)
    if plotted:
        ax.legend(loc=legend_loc, fontsize=8)
    else:
        ax.text(0.5, 0.5, no_data_message, ha="center", va="center", transform=ax.transAxes)
    return ax


def plot_event_overview(
    *,
    layout: EventOverviewLayout,
    raster_plotter: Callable[[Any, EventOverviewLayout], Any],
    rate_plotter: Callable[[Any, EventOverviewLayout], Any],
    figure_width: float = 16.0,
    hspace: float = 0.25,
) -> tuple[Any, Any]:
    """Render a two-row raster-plus-rate overview using a shared layout."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, layout.total_height),
        sharex=False,
        gridspec_kw={"height_ratios": [layout.raster_height, layout.rate_height]},
    )
    raster_plotter(axes[0], layout)
    rate_plotter(axes[1], layout)
    fig.subplots_adjust(left=layout.left_margin, hspace=hspace)
    return fig, axes
