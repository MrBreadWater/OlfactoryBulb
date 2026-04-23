"""Spike input specifications for OlfactoryBulb stimulation.

Each InputSpec produces a sorted numpy array of spike times via
``generate_spike_times(onset_ms, rng, intensity)``.  The result is fed
directly into ``stim_glom_segments`` in the same way the existing Gaussian
spike train is, so all specs are compatible with VecStim / scheduled /
patternstim delivery modes.

Per-segment independent randomization is preserved: the caller supplies a
per-segment ``np.random.RandomState`` seeded from the segment name, time, and
intensity, matching the existing behavior.

JSON-serializable specs: GaussianInput, PoissonInput, SpikeListInput.
Callable-containing specs: RateEnvelopeInput, BinaryFunctionInput.
  These require ``dill`` for subprocess serialization (available in OBGPU env).
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


class InputSpec(ABC):
    """Abstract base for all spike input generators."""

    @abstractmethod
    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Return a sorted array of spike times in ms.

        Parameters
        ----------
        onset_ms:
            The simulation time (ms) at which this input window starts.
        rng:
            Per-segment RandomState for independent randomization.
        intensity:
            0–1 scaling factor.  Specs that ignore intensity (SpikeListInput,
            BinaryFunctionInput) simply disregard this parameter.
        """

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation.  Override for callable specs."""
        raise NotImplementedError(
            f"{type(self).__name__} is not JSON-serializable; use dill serialization."
        )

    @classmethod
    def from_dict(cls, d: dict) -> "InputSpec":
        """Reconstruct an InputSpec from a dict produced by to_dict()."""
        type_map = {
            "GaussianInput": GaussianInput,
            "PoissonInput": PoissonInput,
            "SpikeListInput": SpikeListInput,
        }
        spec_type = d.get("type")
        klass = type_map.get(spec_type)
        if klass is None:
            raise ValueError(
                f"Cannot deserialize InputSpec type {spec_type!r} from dict; "
                "use dill for callable-based specs."
            )
        return klass._from_dict(d)


class GaussianInput(InputSpec):
    """Gaussian-envelope spike train — reproduces the existing model behavior.

    Spikes are drawn from a normal distribution centred at
    ``onset_ms + duration_ms / 2`` whose 99% range spans ``duration_ms``.
    ``intensity`` scales the spike count exactly as the original code does.
    """

    def __init__(
        self,
        max_firing_rate_hz: float = 150.0,
        duration_ms: float = 125.0,
    ) -> None:
        self.max_firing_rate_hz = float(max_firing_rate_hz)
        self.duration_ms = float(duration_ms)

    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        spike_count = int(round(self.max_firing_rate_hz * intensity * (self.duration_ms / 1000.0)))
        if spike_count <= 0:
            return np.array([])
        normal_stdev = self.duration_ms / (2.576 * 2)
        times = rng.normal(onset_ms + self.duration_ms / 2.0, normal_stdev, spike_count)
        times = times[(times > onset_ms) & (times < onset_ms + self.duration_ms)]
        times.sort()
        return times

    def to_dict(self) -> dict:
        return {
            "type": "GaussianInput",
            "max_firing_rate_hz": self.max_firing_rate_hz,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "GaussianInput":
        return cls(
            max_firing_rate_hz=float(d["max_firing_rate_hz"]),
            duration_ms=float(d["duration_ms"]),
        )


class PoissonInput(InputSpec):
    """Constant-rate Poisson process over a fixed window.

    Inter-spike intervals are drawn from an exponential distribution with
    rate ``rate_hz * intensity`` (when ``scale_with_intensity=True``).
    """

    def __init__(
        self,
        rate_hz: float,
        duration_ms: float,
        scale_with_intensity: bool = True,
    ) -> None:
        self.rate_hz = float(rate_hz)
        self.duration_ms = float(duration_ms)
        self.scale_with_intensity = bool(scale_with_intensity)

    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        effective_rate = self.rate_hz * (intensity if self.scale_with_intensity else 1.0)
        if effective_rate <= 0:
            return np.array([])
        # Expected number of spikes; sample enough ISIs to cover the window
        mean_spikes = effective_rate * self.duration_ms / 1000.0
        # Draw extra ISIs to ensure we cover the window
        n_draw = max(1, int(mean_spikes * 3 + 20))
        isis_ms = rng.exponential(1000.0 / effective_rate, n_draw)
        times = onset_ms + np.cumsum(isis_ms)
        times = times[times < onset_ms + self.duration_ms]
        return times

    def to_dict(self) -> dict:
        return {
            "type": "PoissonInput",
            "rate_hz": self.rate_hz,
            "duration_ms": self.duration_ms,
            "scale_with_intensity": self.scale_with_intensity,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "PoissonInput":
        return cls(
            rate_hz=float(d["rate_hz"]),
            duration_ms=float(d["duration_ms"]),
            scale_with_intensity=bool(d.get("scale_with_intensity", True)),
        )


class RateEnvelopeInput(InputSpec):
    """Inhomogeneous Poisson process driven by a user-supplied rate function.

    ``rate_fn(t_ms) -> rate_hz`` is evaluated via the thinning
    (acceptance-rejection) algorithm: spikes are first generated at
    ``max_rate_hz`` using a homogeneous Poisson process, then each spike is
    accepted with probability ``rate_fn(t) / max_rate_hz``.

    If ``max_rate_hz`` is None it is estimated by sampling ``rate_fn`` at 200
    evenly-spaced points across the window; provide it explicitly for speed or
    when the function has narrow peaks.

    Requires ``dill`` for subprocess serialization.
    """

    def __init__(
        self,
        rate_fn: Callable[[float], float],
        duration_ms: float,
        max_rate_hz: float | None = None,
        scale_with_intensity: bool = True,
    ) -> None:
        self.rate_fn = rate_fn
        self.duration_ms = float(duration_ms)
        self.max_rate_hz = float(max_rate_hz) if max_rate_hz is not None else None
        self.scale_with_intensity = bool(scale_with_intensity)

    def _estimate_max_rate(self, onset_ms: float) -> float:
        ts = np.linspace(onset_ms, onset_ms + self.duration_ms, 200)
        return max(float(self.rate_fn(t)) for t in ts)

    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        scale = intensity if self.scale_with_intensity else 1.0
        max_rate = (self.max_rate_hz if self.max_rate_hz is not None
                    else self._estimate_max_rate(onset_ms))
        effective_max = max_rate * scale
        if effective_max <= 0:
            return np.array([])

        # Generate candidate times at max rate (homogeneous Poisson)
        mean_spikes = effective_max * self.duration_ms / 1000.0
        n_draw = max(1, int(mean_spikes * 3 + 20))
        isis_ms = rng.exponential(1000.0 / effective_max, n_draw)
        candidates = onset_ms + np.cumsum(isis_ms)
        candidates = candidates[candidates < onset_ms + self.duration_ms]

        if len(candidates) == 0:
            return np.array([])

        # Thinning: accept each candidate with probability rate(t) / max_rate
        rates = np.array([float(self.rate_fn(t)) * scale for t in candidates])
        accept_probs = np.clip(rates / effective_max, 0.0, 1.0)
        accepted = candidates[rng.uniform(0, 1, len(candidates)) < accept_probs]
        return accepted

    # Not JSON-serializable; dill handles it at the subprocess boundary.


class SpikeListInput(InputSpec):
    """Deterministic explicit spike times.

    All tuft segments receive the same spike times regardless of ``intensity``
    or ``rng``.  Times are stored relative to ``onset_ms=0``; the caller's
    ``onset_ms`` is added as an offset at generation time so the list can be
    reused across different sniff onsets.

    Set ``absolute=True`` if ``times_ms`` are already absolute simulation
    times and should not be shifted by ``onset_ms``.
    """

    def __init__(self, times_ms: list[float], absolute: bool = False) -> None:
        self.times_ms = list(times_ms)
        self.absolute = bool(absolute)

    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        offset = 0.0 if self.absolute else onset_ms
        times = np.array(self.times_ms, dtype=float) + offset
        times.sort()
        return times

    def to_dict(self) -> dict:
        return {
            "type": "SpikeListInput",
            "times_ms": list(self.times_ms),
            "absolute": self.absolute,
        }

    @classmethod
    def _from_dict(cls, d: dict) -> "SpikeListInput":
        return cls(times_ms=list(d["times_ms"]), absolute=bool(d.get("absolute", False)))


class BinaryFunctionInput(InputSpec):
    """Deterministic binary function evaluated at a fixed time resolution.

    ``fn(t_ms) -> {0, 1}`` is evaluated at every ``dt_ms`` step within
    ``[onset_ms, onset_ms + duration_ms)``.  Time points where the function
    returns a nonzero value become spike times.

    All tuft segments receive identical spike times (the function is
    deterministic); ``intensity`` and ``rng`` are ignored.

    Requires ``dill`` for subprocess serialization.
    """

    def __init__(
        self,
        fn: Callable[[float], int],
        duration_ms: float,
        dt_ms: float = 0.1,
    ) -> None:
        self.fn = fn
        self.duration_ms = float(duration_ms)
        self.dt_ms = float(dt_ms)

    def generate_spike_times(
        self,
        onset_ms: float,
        rng: np.random.RandomState,
        intensity: float = 1.0,
    ) -> np.ndarray:
        t_points = np.arange(onset_ms, onset_ms + self.duration_ms, self.dt_ms)
        values = np.array([self.fn(t) for t in t_points])
        return t_points[values != 0]

    # Not JSON-serializable; dill handles it at the subprocess boundary.


def serialize_input_stimuli(input_stimuli: dict) -> tuple[dict, bytes | None]:
    """Split input_stimuli into a JSON-safe dict and an optional dill blob.

    Returns
    -------
    json_safe : dict
        Entries whose InputSpec is JSON-serializable.  Callable-based entries
        are omitted from this dict and included only in the dill blob.
    dill_blob : bytes or None
        dill-serialized bytes of the full input_stimuli dict, or None when all
        specs are JSON-serializable.
    """
    json_safe = {}
    needs_dill = False
    for key, entry in input_stimuli.items():
        spec = entry if isinstance(entry, InputSpec) else entry.get("input")
        try:
            if isinstance(entry, InputSpec):
                json_safe[key] = {"input": spec.to_dict()}
            elif spec is not None:
                json_entry = dict(entry)
                json_entry["input"] = spec.to_dict()
                json_safe[key] = json_entry
            else:
                json_safe[key] = dict(entry)
        except NotImplementedError:
            needs_dill = True

    if not needs_dill:
        return json_safe, None

    import dill
    return json_safe, dill.dumps(input_stimuli)


def deserialize_input_stimuli(blob: bytes) -> dict:
    """Deserialize a dill blob produced by serialize_input_stimuli."""
    import dill
    return dill.loads(blob)


def deserialize_json_input_stimuli(input_stimuli: dict) -> dict:
    """Reconstruct JSON-serialized InputSpec entries from overrides JSON."""
    normalized = {}
    for key, entry in input_stimuli.items():
        if isinstance(entry, InputSpec):
            normalized[key] = entry
            continue
        if not isinstance(entry, dict):
            raise TypeError(f"input_stimuli[{key!r}] must be an InputSpec or dict entry")
        restored = dict(entry)
        spec = restored.get("input")
        if isinstance(spec, dict):
            restored["input"] = InputSpec.from_dict(spec)
        normalized[key] = restored
    return normalized
