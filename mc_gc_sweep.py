"""Helpers for MC↔GC two-cell parameter sweeps and notebook-friendly animations."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from olfactorybulb.output_paths import label_has_timestamp, label_with_timestamp
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, spectrogram


PRESETS = {
    "tuned": {
        "tstop": 300.0,
        "dt_ms": 0.1,
        "init_v": -65.0,
        "mc_input_delay": 50.0,
        "mc_input_dur": 150.0,
        "mc_input_amp": 1.5,
        "mc_to_gc_sections": [0, 1, 2],
        "mc_to_gc_include_soma": True,
        "mc_to_gc_gmax": 100.0,
        "mc_to_gc_weight": 10.0,
        "mc_to_gc_delay": 1.0,
        "mc_to_gc_threshold": 0.0,
        "gc_nmda_factor": 0.02,
        "gc_to_mc_gmax": 0.2,
        "gc_to_mc_weight": 10.0,
        "gc_to_mc_delay": 1.0,
        "gc_to_mc_threshold": -40.0,
        "gc_to_mc_tau1": 1.0,
        "gc_to_mc_tau2": 50.0,
    },
    "official": {
        "tstop": 300.0,
        "dt_ms": 0.1,
        "init_v": -65.0,
        "mc_input_delay": 50.0,
        "mc_input_dur": 150.0,
        "mc_input_amp": 1.5,
        "mc_to_gc_sections": [0],
        "mc_to_gc_include_soma": False,
        "mc_to_gc_gmax": 0.1,
        "mc_to_gc_weight": 1.0,
        "mc_to_gc_delay": 0.5,
        "mc_to_gc_threshold": 0.0,
        "gc_nmda_factor": 0.0035,
        "gc_to_mc_gmax": 0.005,
        "gc_to_mc_weight": 1.0,
        "gc_to_mc_delay": 0.5,
        "gc_to_mc_threshold": 0.0,
        "gc_to_mc_tau1": 1.0,
        "gc_to_mc_tau2": 100.0,
    },
}


_SIMULATION_SCRIPT = r"""
import json
import sys

import numpy as np
from neuron import h
from prev_ob_models.Birgiolas2020.isolated_cells import GC1, MC1


config = json.loads(sys.argv[1])
output_path = sys.argv[2]

h.load_file("stdrun.hoc")
h.cvode.active(0)
h.dt = float(config["dt_ms"])

mc = MC1()
gc = GC1()

mc_stim = h.IClamp(mc.soma(0.5))
mc_stim.delay = float(config["mc_input_delay"])
mc_stim.dur = float(config["mc_input_dur"])
mc_stim.amp = float(config["mc_input_amp"])

gc_target_sections = [gc.cell.apic[i] for i in config["mc_to_gc_sections"]]
if config["mc_to_gc_include_soma"]:
    gc_target_sections.append(gc.soma)

mc_to_gc_syns = []
mc_to_gc_netcons = []
for sec in gc_target_sections:
    syn = h.AmpaNmdaSyn(sec(0.5))
    syn.gmax = float(config["mc_to_gc_gmax"])
    syn.nmdafactor = float(config["gc_nmda_factor"])

    nc = h.NetCon(mc.soma(0.5)._ref_v, syn, sec=mc.soma)
    nc.threshold = float(config["mc_to_gc_threshold"])
    nc.delay = float(config["mc_to_gc_delay"])
    nc.weight[0] = float(config["mc_to_gc_weight"])

    mc_to_gc_syns.append(syn)
    mc_to_gc_netcons.append(nc)

gc_to_mc_syn = h.GabaSyn(mc.cell.dend[0](0.5))
gc_to_mc_syn.gmax = float(config["gc_to_mc_gmax"])
gc_to_mc_syn.tau1 = float(config["gc_to_mc_tau1"])
gc_to_mc_syn.tau2 = float(config["gc_to_mc_tau2"])

gc_to_mc_netcon = h.NetCon(gc.soma(0.5)._ref_v, gc_to_mc_syn, sec=gc.soma)
gc_to_mc_netcon.threshold = float(config["gc_to_mc_threshold"])
gc_to_mc_netcon.delay = float(config["gc_to_mc_delay"])
gc_to_mc_netcon.weight[0] = float(config["gc_to_mc_weight"])

time = h.Vector().record(h._ref_t)
mc_voltage = h.Vector().record(mc.soma(0.5)._ref_v)
gc_voltage = h.Vector().record(gc.soma(0.5)._ref_v)
mc_input_current = h.Vector().record(mc_stim._ref_i)

gc_input_total_vecs = [h.Vector().record(syn._ref_i) for syn in mc_to_gc_syns]
gc_input_ampa_vecs = [h.Vector().record(syn._ref_iampa) for syn in mc_to_gc_syns]
gc_input_nmda_vecs = [h.Vector().record(syn._ref_inmda) for syn in mc_to_gc_syns]
gc_output_current = h.Vector().record(gc_to_mc_syn._ref_i)
gc_output_conductance = h.Vector().record(gc_to_mc_syn._ref_g)

h.finitialize(float(config["init_v"]))
h.continuerun(float(config["tstop"]))

def as_array(vec):
    return np.array(vec, dtype=float)

def summed_arrays(vecs):
    if not vecs:
        return np.zeros_like(as_array(time))
    return np.sum([as_array(v) for v in vecs], axis=0)

np.savez(
    output_path,
    t=as_array(time),
    mc_v=as_array(mc_voltage),
    gc_v=as_array(gc_voltage),
    mc_i=as_array(mc_input_current),
    gc_input_total=summed_arrays(gc_input_total_vecs),
    gc_input_ampa=summed_arrays(gc_input_ampa_vecs),
    gc_input_nmda=summed_arrays(gc_input_nmda_vecs),
    gc_out_i=as_array(gc_output_current),
    gc_out_g=as_array(gc_output_conductance),
)
"""


def _repo_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).resolve().parent


def build_config(preset: str = "tuned", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a sweep configuration from a named preset and optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {sorted(PRESETS)}")
    config = dict(PRESETS[preset])
    if overrides:
        config.update(overrides)
    return config


def run_single_simulation(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run one two-cell sweep simulation in a subprocess and return its saved arrays."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        subprocess.run(
            [sys.executable, "-c", _SIMULATION_SCRIPT, json.dumps(config), str(output_path)],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        with np.load(output_path) as data:
            return {key: data[key] for key in data.files}
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        msg = stderr or stdout or "NEURON subprocess failed"
        raise RuntimeError(msg) from exc
    finally:
        output_path.unlink(missing_ok=True)


def run_parameter_sweep(
    parameter_name: str,
    values: list[Any],
    preset: str = "tuned",
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run a one-parameter sweep and attach the swept value to each returned frame."""
    base_config = build_config(preset=preset, overrides=overrides)
    results = []
    for value in values:
        run_config = dict(base_config)
        run_config[parameter_name] = value
        traces = run_single_simulation(run_config)
        traces["swept_parameter"] = parameter_name
        traces["swept_value"] = value
        traces["config"] = run_config
        results.append(traces)
    return results


def _format_value(value: Any) -> str:
    """Format a sweep value compactly for plot titles."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _validate_results(results: list[dict[str, Any]]) -> None:
    """Raise when a sweep result set is empty."""
    if not results:
        raise ValueError("No sweep results were provided.")


def animate_trace_sweep(
    results: list[dict[str, Any]],
    parameter_name: str | None = None,
    interval: int = 800,
):
    """Animate voltage and current traces across a sweep."""
    _validate_results(results)
    parameter_name = parameter_name or results[0]["swept_parameter"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].set_ylabel("nA")
    axes[0].set_title("MC External Input")
    axes[1].set_ylabel("nA")
    axes[1].set_title("GC Excitatory Input From MC")
    axes[2].set_ylabel("nA / uS")
    axes[2].set_title("GC Inhibitory Output Onto MC")
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_ylabel("mV")
    axes[3].set_title("Cell Outputs")

    def draw_frame(frame_index):
        result = results[frame_index]
        t = result["t"]

        for ax in axes:
            ax.clear()

        axes[0].plot(t, result["mc_i"], color="black")
        axes[0].set_ylabel("nA")
        axes[0].set_title("MC External Input")

        axes[1].plot(t, result["gc_input_total"], label="Total", color="tab:blue")
        axes[1].plot(t, result["gc_input_ampa"], label="AMPA", color="tab:orange", alpha=0.8)
        axes[1].plot(t, result["gc_input_nmda"], label="NMDA", color="tab:green", alpha=0.8)
        axes[1].set_ylabel("nA")
        axes[1].set_title("GC Excitatory Input From MC")
        axes[1].legend(loc="best")

        axes[2].plot(t, result["gc_out_i"], label="GC -> MC inhibitory current", color="tab:red")
        axes[2].plot(t, result["gc_out_g"], label="GC -> MC conductance", color="tab:purple", alpha=0.8)
        axes[2].set_ylabel("nA / uS")
        axes[2].set_title("GC Inhibitory Output Onto MC")
        axes[2].legend(loc="best")

        axes[3].plot(t, result["mc_v"], label="MC soma", color="tab:blue")
        axes[3].plot(t, result["gc_v"], label="GC soma", color="tab:red")
        axes[3].set_xlabel("Time (ms)")
        axes[3].set_ylabel("mV")
        axes[3].set_title("Cell Outputs")
        axes[3].legend(loc="best")

        fig.suptitle(
            f"Sweep: {parameter_name} = {_format_value(result['swept_value'])}",
            fontsize=14,
        )
        fig.tight_layout()

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(results), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def compute_spectrogram(
    signal,
    time_ms,
    max_freq=100,
    nperseg=512,
    noverlap=480,
    interpolate_dt_ms=0.01,
    apply_lowpass=True,
    lowpass_hz=100,
    butter_order=3,
):
    """Compute a uniformly sampled spectrogram for a single time-series signal."""
    time_ms = np.asarray(time_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time_ms.ndim != 1 or signal.ndim != 1:
        raise ValueError("time_ms and signal must be one-dimensional arrays.")
    if len(time_ms) != len(signal):
        raise ValueError("time_ms and signal must have the same length.")

    dt = np.diff(time_ms)
    if len(dt) == 0:
        raise ValueError("Signal must contain at least two samples.")

    if interpolate_dt_ms is not None:
        uniform_time = np.arange(time_ms[0], time_ms[-1], float(interpolate_dt_ms))
        if len(uniform_time) < 2:
            raise ValueError("Interpolated time grid is too short for spectrogram computation.")
        interpolator = interp1d(time_ms, signal, kind="linear", fill_value="extrapolate")
        signal = interpolator(uniform_time)
        time_ms = uniform_time
    elif not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-9):
        uniform_time = np.arange(time_ms[0], time_ms[-1], np.median(dt))
        if len(uniform_time) < 2:
            raise ValueError("Interpolated time grid is too short for spectrogram computation.")
        interpolator = interp1d(time_ms, signal, kind="linear", fill_value="extrapolate")
        signal = interpolator(uniform_time)
        time_ms = uniform_time

    dt = np.diff(time_ms)

    fs = 1000.0 / dt[0]

    if apply_lowpass and lowpass_hz is not None and 0 < lowpass_hz < (fs / 2):
        b, a = butter(int(butter_order), float(lowpass_hz) / (fs / 2), btype="low")
        try:
            signal = filtfilt(b, a, signal)
        except ValueError:
            # Too few points for filtfilt padding; keep the unfiltered signal.
            pass

    nperseg = min(int(nperseg), len(signal))
    noverlap = min(int(noverlap), max(nperseg - 1, 0))

    freqs, times_s, power = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )

    mask = freqs <= max_freq
    return times_s * 1000.0, freqs[mask], power[mask]


def animate_spectrogram_sweep(
    results,
    signal_key="gc_out_g",
    parameter_name=None,
    max_freq=100,
    nperseg=512,
    noverlap=480,
    interpolate_dt_ms=0.01,
    apply_lowpass=True,
    lowpass_hz=100,
    butter_order=3,
    log_floor=1e-4,
    interval=800,
):
    """Animate a selected signal's spectrogram across sweep conditions."""
    _validate_results(results)
    parameter_name = parameter_name or results[0]["swept_parameter"]

    spec_data = [
        compute_spectrogram(
            result[signal_key],
            result["t"],
            max_freq=max_freq,
            nperseg=nperseg,
            noverlap=noverlap,
            interpolate_dt_ms=interpolate_dt_ms,
            apply_lowpass=apply_lowpass,
            lowpass_hz=lowpass_hz,
            butter_order=butter_order,
        )
        for result in results
    ]

    db_data = [10 * np.log10(power + float(log_floor)) for _, _, power in spec_data]
    vmax = max(np.max(db) for db in db_data)
    vmin = min(np.min(db) for db in db_data)
    if np.isclose(vmax, vmin):
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(10, 4))

    def draw_frame(frame_index):
        ax.clear()
        times_ms, freqs, _ = spec_data[frame_index]
        db_power = db_data[frame_index]
        mesh = ax.pcolormesh(
            times_ms,
            freqs,
            db_power,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, max_freq)
        ax.set_title(
            f"Spectrogram of {signal_key} | {parameter_name} = {_format_value(results[frame_index]['swept_value'])}"
        )
        return [mesh]

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(results), interval=interval, repeat=True)
    plt.close(fig)
    return anim


def _safe_output_name(name: str) -> str:
    """Return a filesystem-safe basename for saved sweep artifacts."""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(name)).strip("._")


def display_animation(anim, name, fps=2, output_dir="notebook_outputs"):
    """Save and display a sweep animation, preferring a GIF for PyCharm/Jupyter compatibility."""
    output_dir = str(output_dir)
    output_path_root = Path(output_dir)
    if not label_has_timestamp(output_path_root.name):
        output_path_root = output_path_root.parent / label_with_timestamp(output_path_root.name)
    output_path = output_path_root
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_output_name(name) or "animation"
    gif_path = (output_path / f"{safe_name}.gif").resolve()

    try:
        from IPython.display import HTML, Image, display
    except Exception:
        # Non-notebook execution path.
        writer = animation.PillowWriter(fps=max(1, int(fps)))
        anim.save(str(gif_path), writer=writer)
        return str(gif_path)

    try:
        writer = animation.PillowWriter(fps=max(1, int(fps)))
        anim.save(str(gif_path), writer=writer)
        display(Image(filename=str(gif_path)))
        return str(gif_path)
    except Exception:
        try:
            display(HTML(anim.to_html5_video()))
            return None
        except Exception:
            display(HTML(anim.to_jshtml()))
            return None
