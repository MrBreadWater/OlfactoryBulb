"""Run a timestamped benchmark simulation and summarize the saved outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olfactorybulb.output_paths import configure_output_env


def first_nonfinite_index(values: Any) -> tuple[int, float] | None:
    """Return the first non-finite index/value in a numeric sequence."""
    try:
        import numpy as np

        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None

    bad = np.flatnonzero(~np.isfinite(arr))
    if len(bad) == 0:
        return None
    index = int(bad[0])
    return index, float(arr.reshape(-1)[index])


def validate_numeric_trace(label: str, times: Any, values: Any) -> list[str]:
    """Return validation errors for one saved time/value trace."""
    errors = []
    bad_time = first_nonfinite_index(times)
    if bad_time is not None:
        index, value = bad_time
        errors.append(f"{label}: non-finite time at index {index}: {value!r}")

    bad_value = first_nonfinite_index(values)
    if bad_value is not None:
        index, value = bad_value
        time_hint = ""
        try:
            time_hint = f" at t={float(times[index]):.6g} ms"
        except (TypeError, ValueError, IndexError):
            pass
        errors.append(f"{label}: non-finite value at index {index}{time_hint}: {value!r}")
    return errors


def validate_saved_outputs(out_dir: str | Path, *, require_lfp: bool) -> None:
    """Fail fast when saved benchmark outputs contain NaN or Inf values."""
    out_path = Path(out_dir)
    errors: list[str] = []

    soma_path = out_path / "soma_vs.pkl"
    if soma_path.exists():
        with open(soma_path, "rb") as f:
            soma_traces = pickle.load(f)
        for trace_index, trace in enumerate(soma_traces):
            try:
                label, times, values = trace
            except (TypeError, ValueError):
                errors.append(f"soma_vs.pkl[{trace_index}]: expected (label, times, values)")
                continue
            errors.extend(validate_numeric_trace(f"soma_vs.pkl[{trace_index}] {label}", times, values))

    lfp_path = out_path / "lfp.pkl"
    if lfp_path.exists():
        with open(lfp_path, "rb") as f:
            lfp_times, lfp_values = pickle.load(f)
        errors.extend(validate_numeric_trace("lfp.pkl", lfp_times, lfp_values))
    elif require_lfp:
        errors.append("lfp.pkl: missing even though LFP recording is enabled")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:10])
        remaining = len(errors) - 10
        if remaining > 0:
            preview += f"\n- ... {remaining} additional trace errors"
        raise RuntimeError(
            "Simulation produced non-finite saved outputs. "
            "Treating this run as failed instead of publishing a misleading summary.\n"
            f"{preview}"
        )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for an in-memory byte payload."""
    return hashlib.sha256(data).hexdigest()


def canonicalize(obj: Any) -> Any:
    """Recursively normalize containers so order-insensitive comparisons are reproducible."""
    if isinstance(obj, tuple):
        return tuple(canonicalize(x) for x in obj)

    if isinstance(obj, list):
        canon_items = [canonicalize(x) for x in obj]
        try:
            canon_items = sorted(canon_items)
        except TypeError:
            pass
        return canon_items

    return obj


def summarize_pickle(path: str | Path) -> dict[str, Any]:
    """Summarize a saved pickle with stable hashes for regression comparisons."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple) and len(obj) == 2:
        t, values = obj
        return {
            "type": "tuple2",
            "len_0": len(t),
            "len_1": len(values),
            "sha256": sha256_file(path),
            "canonical_sha256": sha256_bytes(pickle.dumps(canonicalize(obj), protocol=4)),
        }

    if isinstance(obj, list):
        return {
            "type": "list",
            "items": len(obj),
            "sha256": sha256_file(path),
            "canonical_sha256": sha256_bytes(pickle.dumps(canonicalize(obj), protocol=4)),
        }

    return {
        "type": type(obj).__name__,
        "sha256": sha256_file(path),
    }


def rank0_path(base_dir: str | Path, label: str) -> Path:
    """Return the rank-0 results directory for a benchmark label."""
    return Path(base_dir) / label


def parse_positive_int_env(*names: str) -> int | None:
    """Return the first positive integer environment value among names."""
    for name in names:
        raw_value = os.environ.get(name)
        if raw_value in (None, ""):
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


class NeuronParallelComm:
    """Small MPI-like wrapper around NEURON's ParallelContext."""

    def __init__(self, pc: Any):
        self.pc = pc

    def Get_rank(self) -> int:
        """Return the current NEURON MPI rank."""
        return int(self.pc.id())

    def Get_size(self) -> int:
        """Return the current NEURON MPI world size."""
        return int(self.pc.nhost())

    def bcast(self, value: Any, root: int = 0) -> Any:
        """Broadcast a Python object from one NEURON MPI rank."""
        return self.pc.py_broadcast(value, int(root))

    def Barrier(self) -> None:
        """Synchronize all NEURON MPI ranks."""
        self.pc.barrier()

    def allreduce_sum(self, value: int | float) -> int | float:
        """Return the sum of a scalar over all NEURON MPI ranks."""
        return self.pc.allreduce(value, 1)

    def allreduce_max(self, value: int | float) -> int | float:
        """Return the maximum of a scalar over all NEURON MPI ranks."""
        return self.pc.allreduce(value, 2)


def deep_update_mapping(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge nested override dictionaries into a target mapping in place."""
    for key, value in overrides.items():
        if isinstance(value, dict):
            current = target.get(key)
            if isinstance(current, dict):
                deep_update_mapping(current, value)
            else:
                target[key] = dict(value)
        else:
            target[key] = value


def normalize_input_odors(value: Any) -> Any:
    """Convert JSON-decoded odor onset keys back to numeric timestamps when possible."""
    if not isinstance(value, dict):
        return value

    normalized = {}
    for key, entry in value.items():
        try:
            time_key = float(key)
        except (TypeError, ValueError):
            time_key = key
        else:
            if isinstance(time_key, float) and time_key.is_integer():
                time_key = int(time_key)

        normalized[time_key] = entry

    return normalized


def apply_param_overrides(params: Any, overrides: dict[str, Any]) -> None:
    """Apply nested CLI overrides to a paramset instance."""
    for key, value in overrides.items():
        if key == "input_odors":
            value = normalize_input_odors(value)
        elif key == "input_stimuli":
            from olfactorybulb.inputs import deserialize_json_input_stimuli

            value = deserialize_json_input_stimuli(value)
        current = getattr(params, key, None)
        if isinstance(value, dict):
            if isinstance(current, dict):
                deep_update_mapping(current, value)
            else:
                setattr(params, key, dict(value))
        else:
            setattr(params, key, value)


def main() -> None:
    """Entry point for the benchmark runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--paramset", default="GammaSignature")
    parser.add_argument("--label", required=True)
    parser.add_argument("--results-base", default="results/benchmarks")
    parser.add_argument("--tstop-override", type=float, default=None)
    parser.add_argument("--coreneuron", action="store_true")
    parser.add_argument("--coreneuron-gpu", action="store_true")
    parser.add_argument("--coreneuron-file-mode", action="store_true")
    parser.add_argument("--coreneuron-verbose", type=int, default=0)
    parser.add_argument("--coreneuron-warp-balance", type=int, default=None)
    parser.add_argument("--runtime-mode", choices=["scientific", "exploratory"], default=None)
    parser.add_argument("--input-event-strategy", choices=["vecstim", "scheduled", "patternstim"], default=None)
    parser.add_argument("--force-gid-synapses", choices=["true", "false"], default=None)
    parser.add_argument("--disable-status-report", action="store_true")
    parser.add_argument("--disable-lfp-electrode", action="store_true")
    parser.add_argument("--parallel-timeout", type=float, default=None)
    parser.add_argument("--overrides-json", default=None)
    parser.add_argument("--overrides-file", default=None)
    parser.add_argument("--input-spec-file", default=None)
    parser.add_argument("--add-connections-json", default=None)
    parser.add_argument("--modify-connections-json", default=None)
    parser.add_argument("--swap-cell-types-json", default=None)
    args, _unknown = parser.parse_known_args()
    overrides_json = json.loads(args.overrides_json) if args.overrides_json is not None else None
    add_connections = (
        json.loads(args.add_connections_json) if args.add_connections_json is not None else None
    )
    modify_connections = (
        json.loads(args.modify_connections_json) if args.modify_connections_json is not None else None
    )
    swap_cell_types = (
        json.loads(args.swap_cell_types_json) if args.swap_cell_types_json is not None else None
    )
    if args.coreneuron_warp_balance is None:
        if args.coreneuron_gpu:
            args.coreneuron_warp_balance = int(os.environ.get("OB_CORENRN_WARP_BALANCE", "128"))
        else:
            args.coreneuron_warp_balance = 0

    if args.repo_root is not None:
        repo_root = Path(args.repo_root).resolve()
        os.chdir(repo_root)
        sys.path.insert(0, str(repo_root))
    else:
        repo_root = Path.cwd()

    from neuron import h
    try:
        from neuron import coreneuron
    except ImportError:
        coreneuron = None
    import olfactorybulb.model as obmodel
    from olfactorybulb.model import OlfactoryBulb

    pc = h.ParallelContext()
    comm = NeuronParallelComm(pc)
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    requested_slurm_tasks = parse_positive_int_env(
        "SLURM_STEP_NUM_TASKS",
        "SLURM_NTASKS",
        "PMI_SIZE",
        "PMIX_SIZE",
    )
    if requested_slurm_tasks and requested_slurm_tasks > 1 and nranks == 1:
        raise RuntimeError(
            "MPI launch did not form a multi-rank NEURON ParallelContext: "
            f"ParallelContext.nhost() is {nranks}, but Slurm/PMI requested "
            f"{requested_slurm_tasks} tasks. Check the remote MPI launcher, "
            "Slurm --mpi mode, and the MPI implementation used by nrniv."
        )
    final_label, run_timestamp = configure_output_env(args.label, comm=comm, results_base=args.results_base)

    out_dir = rank0_path(args.results_base, final_label)

    if rank == 0 and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    comm.Barrier()

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    params = getattr(obmodel, args.paramset)()
    params.runtime_mode = args.runtime_mode or getattr(params, "runtime_mode", "scientific")
    params.coreneuron = SimpleNamespace(
        enable=args.coreneuron,
        gpu=args.coreneuron_gpu,
        file_mode=args.coreneuron_file_mode,
        verbose=args.coreneuron_verbose,
        cell_permute=None,
        warp_balance=args.coreneuron_warp_balance,
    )
    if args.input_event_strategy is not None:
        params.input_event_strategy = args.input_event_strategy
    if args.force_gid_synapses is not None:
        params.force_gid_synapses = args.force_gid_synapses == "true"
    if args.overrides_json is not None and args.overrides_file is not None:
        raise ValueError("Use only one of --overrides-json or --overrides-file")
    if overrides_json is not None:
        apply_param_overrides(params, overrides_json)
    if args.input_spec_file is not None:
        import dill

        with open(args.input_spec_file, "rb") as f:
            input_stimuli = dill.load(f)
        params.input_stimuli = input_stimuli
    if args.overrides_file is not None:
        with open(args.overrides_file) as f:
            apply_param_overrides(params, json.load(f))
    if args.runtime_mode is not None and not (overrides_json and "legacy_parallel_dt" in overrides_json):
        params.legacy_parallel_dt = args.runtime_mode == "scientific"

    build_start = time.perf_counter()
    ob = OlfactoryBulb(params, autorun=False)
    if add_connections is not None or modify_connections is not None or swap_cell_types is not None:
        from obgpu_experiment_helpers import (
            add_new_connections,
            modify_existing_connections,
            perform_cell_type_swaps,
        )
    if add_connections is not None:
        add_new_connections(ob, add_connections)
    if modify_connections is not None:
        modify_existing_connections(ob, modify_connections)
    if swap_cell_types is not None:
        perform_cell_type_swaps(ob, swap_cell_types)
    if args.tstop_override is not None:
        ob.params.tstop = args.tstop_override
    if args.disable_status_report:
        ob.params.enable_status_report = False
    if args.disable_lfp_electrode:
        ob.params.enable_lfp = False
    if args.parallel_timeout is not None:
        ob.params.parallel_timeout = args.parallel_timeout
    build_local = time.perf_counter() - build_start
    build_max = comm.allreduce_max(build_local)

    ob.results_dir = str(out_dir)
    comm.Barrier()

    if coreneuron is None:
        if any([args.coreneuron, args.coreneuron_gpu, args.coreneuron_file_mode, args.coreneuron_verbose]):
            raise RuntimeError("This NEURON build does not expose neuron.coreneuron")
    else:
        coreneuron.enable = args.coreneuron
        coreneuron.gpu = args.coreneuron_gpu
        coreneuron.file_mode = args.coreneuron_file_mode
        coreneuron.verbose = args.coreneuron_verbose
        if args.coreneuron_warp_balance is not None:
            coreneuron.warp_balance = args.coreneuron_warp_balance
        cell_permute_override = os.environ.get("OB_CORENRN_CELL_PERMUTE")
        if cell_permute_override is not None:
            cell_permute_value = int(cell_permute_override)
        else:
            cell_permute_value = 2 if args.coreneuron_gpu else 0
        if any([args.coreneuron, args.coreneuron_gpu, args.coreneuron_file_mode]):
            coreneuron.cell_permute = cell_permute_value
        params.coreneuron.cell_permute = cell_permute_value
        params.coreneuron.warp_balance = getattr(coreneuron, "warp_balance", None)

    run_start = time.perf_counter()
    ob.run(ob.params.tstop)
    run_local = time.perf_counter() - run_start
    run_max = comm.allreduce_max(run_local)

    save_start = time.perf_counter()
    ob.save_recorded_vectors()
    if getattr(ob.params, "enable_lfp", True):
        ob.get_lfp()
    save_local = time.perf_counter() - save_start
    save_max = comm.allreduce_max(save_local)

    total_local = build_local + run_local + save_local
    total_max = comm.allreduce_max(total_local)

    local_cell_counts = {cell_type: len(cells) for cell_type, cells in ob.cells.items()}
    total_cell_counts = {
        cell_type: int(comm.allreduce_sum(count))
        for cell_type, count in local_cell_counts.items()
    }

    if rank == 0:
        validate_saved_outputs(out_dir, require_lfp=getattr(ob.params, "enable_lfp", True))

        file_summaries = {}
        for filename in ["soma_vs.pkl", "input_times.pkl", "lfp.pkl"]:
            path = out_dir / filename
            if path.exists():
                file_summaries[filename] = summarize_pickle(path)

        summary = {
            "label": final_label,
            "requested_label": args.label,
            "timestamp": run_timestamp,
            "paramset": args.paramset,
            "nranks": nranks,
            "timing_seconds": {
                "build_max_rank": build_max,
                "run_max_rank": run_max,
                "save_max_rank": save_max,
                "total_max_rank": total_max,
            },
            "cells_total": total_cell_counts,
            "params": {
                "tstop": ob.params.tstop,
                "sim_dt": ob.params.sim_dt,
                "actual_dt": ob.h.dt,
                "recording_period": ob.params.recording_period,
                "record_from_somas": list(ob.params.record_from_somas),
                "enable_reciprocal_synapses": getattr(ob.params, "enable_reciprocal_synapses", True),
                "force_gid_synapses": bool(getattr(ob.bn_server, "force_gid_synapses", False)),
                "legacy_parallel_dt": getattr(ob.params, "legacy_parallel_dt", True),
                "runtime_mode": getattr(ob.params, "runtime_mode", "scientific"),
                "coreneuron": {
                    "enable": args.coreneuron,
                    "gpu": args.coreneuron_gpu,
                    "file_mode": args.coreneuron_file_mode,
                    "verbose": args.coreneuron_verbose,
                    "cell_permute": int(getattr(coreneuron, "cell_permute", 0)),
                    "warp_balance": getattr(coreneuron, "warp_balance", None) if coreneuron is not None else None,
                },
                "parallel_timeout": getattr(ob.params, "parallel_timeout", None),
            },
            "files": file_summaries,
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print(json.dumps(summary, indent=2, sort_keys=True))

    if nranks > 1:
        try:
            from olfactorybulb.database import database

            database.close()
        except Exception:
            pass

    if os.environ.get("OBGPU_SKIP_H_QUIT", "0") != "1":
        try:
            from neuron import h

            h.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
