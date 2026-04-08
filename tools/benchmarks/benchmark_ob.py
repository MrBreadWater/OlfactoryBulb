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

from olfactorybulb.output_paths import configure_output_env


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonicalize(obj):
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


def summarize_pickle(path):
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


def rank0_path(base_dir, label):
    return Path(base_dir) / label


def deep_update_mapping(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            current = target.get(key)
            if isinstance(current, dict):
                deep_update_mapping(current, value)
            else:
                target[key] = dict(value)
        else:
            target[key] = value


def normalize_input_odors(value):
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


def apply_param_overrides(params, overrides):
    for key, value in overrides.items():
        if key == "input_odors":
            value = normalize_input_odors(value)
        current = getattr(params, key, None)
        if isinstance(value, dict):
            if isinstance(current, dict):
                deep_update_mapping(current, value)
            else:
                setattr(params, key, dict(value))
        else:
            setattr(params, key, value)


def main():
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
    args, _unknown = parser.parse_known_args()
    overrides_json = json.loads(args.overrides_json) if args.overrides_json is not None else None
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

    from mpi4py import MPI
    try:
        from neuron import coreneuron
    except ImportError:
        coreneuron = None
    import olfactorybulb.model as obmodel
    from olfactorybulb.model import OlfactoryBulb

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    final_label, run_timestamp = configure_output_env(args.label, comm=comm, results_base=args.results_base)

    out_dir = rank0_path(args.results_base, final_label)

    if rank == 0 and out_dir.exists():
        shutil.rmtree(out_dir)
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
    if args.overrides_file is not None:
        with open(args.overrides_file) as f:
            apply_param_overrides(params, json.load(f))
    if args.runtime_mode is not None and not (overrides_json and "legacy_parallel_dt" in overrides_json):
        params.legacy_parallel_dt = args.runtime_mode == "scientific"

    build_start = time.perf_counter()
    ob = OlfactoryBulb(params, autorun=False)
    if args.tstop_override is not None:
        ob.params.tstop = args.tstop_override
    if args.disable_status_report:
        ob.params.enable_status_report = False
    if args.disable_lfp_electrode:
        ob.params.enable_lfp = False
    if args.parallel_timeout is not None:
        ob.params.parallel_timeout = args.parallel_timeout
    build_local = time.perf_counter() - build_start
    build_max = comm.allreduce(build_local, op=MPI.MAX)

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
        coreneuron.cell_permute = cell_permute_value
        params.coreneuron.cell_permute = cell_permute_value
        params.coreneuron.warp_balance = getattr(coreneuron, "warp_balance", None)

    run_start = time.perf_counter()
    ob.run(ob.params.tstop)
    run_local = time.perf_counter() - run_start
    run_max = comm.allreduce(run_local, op=MPI.MAX)

    save_start = time.perf_counter()
    ob.save_recorded_vectors()
    if getattr(ob.params, "enable_lfp", True):
        ob.get_lfp()
    save_local = time.perf_counter() - save_start
    save_max = comm.allreduce(save_local, op=MPI.MAX)

    total_local = build_local + run_local + save_local
    total_max = comm.allreduce(total_local, op=MPI.MAX)

    local_cell_counts = {cell_type: len(cells) for cell_type, cells in ob.cells.items()}
    total_cell_counts = {
        cell_type: int(comm.allreduce(count, op=MPI.SUM))
        for cell_type, count in local_cell_counts.items()
    }

    if rank == 0:
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

    try:
        from neuron import h

        h.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
