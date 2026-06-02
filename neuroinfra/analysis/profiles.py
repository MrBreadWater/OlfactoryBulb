"""Reusable domain-profile helpers for result analysis surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .catalog import CategoryCatalogHooks
from .events import ResultEventFamilySuite, ResultEventPlotSuite
from .frequency_plots import ResultFrequencyPlotSuite
from .signal_views import ResultSignalViewSuite
from .signals import ResultSignalRegistry
from .sweeps import SweepPlotRegistry, list_registry_plot_names, resolve_registry_plot


@dataclass(frozen=True)
class ResultAnalysisProfile:
    """One domain-facing analysis profile built on top of reusable neuroinfra suites."""

    category_hooks: CategoryCatalogHooks | None = None
    signal_registry: ResultSignalRegistry | None = None
    signal_views: ResultSignalViewSuite | None = None
    event_families: Mapping[str, ResultEventFamilySuite] = field(default_factory=dict)
    event_plots: Mapping[str, ResultEventPlotSuite] = field(default_factory=dict)
    frequency_plots: Mapping[str, ResultFrequencyPlotSuite] = field(default_factory=dict)
    sweep_plot_registry: SweepPlotRegistry | None = None

    def list_available_signals(
        self,
        result: dict[str, Any],
        **context: Any,
    ) -> list[str]:
        """List named signals through the profile's configured signal registry."""
        return self.require_signal_registry().list_available(result, **context)

    def resolve_signal(
        self,
        result: dict[str, Any],
        signal: str,
        *,
        dt_ms: float | None = None,
        **context: Any,
    ) -> tuple[Any, Any]:
        """Resolve one named signal through the profile's signal-view surface."""
        return self.require_signal_views().resolve(
            result,
            signal=signal,
            dt_ms=dt_ms,
            **context,
        )

    def require_signal_registry(self) -> ResultSignalRegistry:
        """Return the configured signal registry or raise a descriptive error."""
        if self.signal_registry is None:
            raise RuntimeError("ResultAnalysisProfile does not define a signal registry")
        return self.signal_registry

    def require_signal_views(self) -> ResultSignalViewSuite:
        """Return the configured signal-view suite or synthesize it from the registry."""
        if self.signal_views is not None:
            return self.signal_views
        return ResultSignalViewSuite(self.require_signal_registry())

    def event_family(self, name: str) -> ResultEventFamilySuite:
        """Resolve one named result-backed event family."""
        try:
            return self.event_families[str(name)]
        except KeyError as exc:
            available = ", ".join(sorted(self.event_families))
            raise KeyError(f"Unknown event family {name!r}. Available: {available}") from exc

    def event_plot_suite(self, name: str) -> ResultEventPlotSuite:
        """Resolve one named result-backed event plotting suite."""
        try:
            return self.event_plots[str(name)]
        except KeyError as exc:
            available = ", ".join(sorted(self.event_plots))
            raise KeyError(f"Unknown event plot suite {name!r}. Available: {available}") from exc

    def frequency_plot_suite(self, name: str) -> ResultFrequencyPlotSuite:
        """Resolve one named result-backed frequency plotting suite."""
        try:
            return self.frequency_plots[str(name)]
        except KeyError as exc:
            available = ", ".join(sorted(self.frequency_plots))
            raise KeyError(f"Unknown frequency plot suite {name!r}. Available: {available}") from exc

    def list_builtin_sweep_plot_names(self) -> list[str]:
        """List built-in sweep plots exposed by this profile."""
        return list_registry_plot_names(self.require_sweep_plot_registry())

    def resolve_builtin_sweep_plot(self, name: str) -> Any:
        """Resolve one built-in sweep plot callable exposed by this profile."""
        return resolve_registry_plot(self.require_sweep_plot_registry(), name)

    def require_sweep_plot_registry(self) -> SweepPlotRegistry:
        """Return the configured sweep-plot registry or raise a descriptive error."""
        if self.sweep_plot_registry is None:
            raise RuntimeError("ResultAnalysisProfile does not define a sweep plot registry")
        return self.sweep_plot_registry
