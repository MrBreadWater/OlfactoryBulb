"""Smoke tests for concrete olfactory-bulb HFO/LFP view adapters."""

from __future__ import annotations

import numpy as np

from olfactorybulb.analysis_hfo_views import (
    DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ,
    DEFAULT_PSD_TEMPLATE_FLOOR,
    build_psd_template_overlays,
    plot_hfo_power_summary,
    plot_lfp_overview,
)


class _FakeSignalViews:
    def __init__(self) -> None:
        self.psd_calls = []
        self.band_calls = []

    def plot_signal_psd_overview(self, result, **kwargs):
        self.psd_calls.append((result, kwargs))
        return "fig", ("ax0", "ax1", "ax2")

    def plot_band_power_summary(self, result, **kwargs):
        self.band_calls.append((result, kwargs))
        return "fig", ("ax0", "ax1"), {"signal": kwargs["signal"]}


class _FakeProfile:
    def __init__(self) -> None:
        self.signal_views = _FakeSignalViews()

    def require_signal_views(self):
        return self.signal_views


def main() -> None:
    overlays = build_psd_template_overlays(
        psd_template_kind="ketamine",
        psd_template_fit_band_hz=(100.0, 200.0),
        psd_template_scale_method="area",
        psd_template_floor=1e-5,
        psd_template_color="tab:orange",
        scaled_curve_fn=lambda kind, freqs, power, **kwargs: (
            np.asarray(freqs, dtype=float),
            np.asarray(power, dtype=float) * 0.5,
        ),
    )
    built = overlays(np.array([10.0, 20.0]), np.array([2.0, 4.0]))
    assert len(built) == 1
    assert built[0].label == "Template (ketamine)"
    np.testing.assert_allclose(built[0].freqs_hz, [10.0, 20.0])
    np.testing.assert_allclose(built[0].power, [1.0, 2.0])

    broken = build_psd_template_overlays(
        psd_template_kind="ketamine",
        psd_template_fit_band_hz=(100.0, 200.0),
        psd_template_scale_method="area",
        psd_template_floor=1e-5,
        psd_template_color="tab:orange",
        scaled_curve_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert broken(np.array([10.0]), np.array([2.0])) == []

    profile = _FakeProfile()
    result = {"lfp": [1.0, 2.0]}

    fig, axes = plot_lfp_overview(
        profile,
        result,
        show_psd_target_template=False,
    )
    assert fig == "fig"
    assert axes == ("ax0", "ax1", "ax2")
    assert len(profile.signal_views.psd_calls) == 1
    call_result, call_kwargs = profile.signal_views.psd_calls[0]
    assert call_result is result
    assert call_kwargs["signal"] == "lfp"
    assert call_kwargs["signal_label"] == "LFP"
    assert call_kwargs["psd_overlay_builder"] is None

    profile = _FakeProfile()
    plot_lfp_overview(profile, result)
    _, call_kwargs = profile.signal_views.psd_calls[0]
    assert call_kwargs["psd_overlay_builder"] is not None

    fig, axes, summary = plot_hfo_power_summary(
        profile,
        result,
        signal="lfp",
        bands={"hfo": (80.0, 120.0)},
        dt_ms=0.1,
        relative_band=(60.0, 140.0),
    )
    assert fig == "fig"
    assert axes == ("ax0", "ax1")
    assert summary["signal"] == "lfp"
    assert len(profile.signal_views.band_calls) == 1
    _, band_kwargs = profile.signal_views.band_calls[0]
    assert band_kwargs["signal"] == "lfp"
    assert band_kwargs["bands"] == {"hfo": (80.0, 120.0)}
    assert band_kwargs["relative_band"] == (60.0, 140.0)

    assert DEFAULT_PSD_TEMPLATE_FIT_BAND_HZ == (130.0, 230.0)
    assert DEFAULT_PSD_TEMPLATE_FLOOR == 1e-5

    print("olfactorybulb HFO view adapters: OK")


if __name__ == "__main__":
    main()
