"""
This is a helper file for running multi-core/MPI simulations. For example:

mpiexec -n 2 nrniv -mpi -python initslice.py -paramset GammaSignature -mpi
"""

import os
import sys
from types import SimpleNamespace

if '-mpi' in sys.argv:
    from mpi4py import MPI

import olfactorybulb.model as obmodel
from olfactorybulb.model import OlfactoryBulb as OB
from olfactorybulb.output_paths import configure_output_env


def env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def env_int(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_choice(name, default=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower()


def build_params(paramset_name):
    params = getattr(obmodel, paramset_name)()
    runtime_mode = env_choice("OB_RUNTIME_MODE", getattr(params, "runtime_mode", "scientific"))
    if runtime_mode not in {"scientific", "exploratory"}:
        raise ValueError(f"Unsupported OB_RUNTIME_MODE={runtime_mode!r}")
    params.runtime_mode = runtime_mode

    default_gpu = os.environ.get("CONDA_DEFAULT_ENV", "") == "OBGPU"
    coreneuron_gpu = env_flag("OB_USE_CORENRN_GPU", default_gpu)
    coreneuron_enable = env_flag("OB_USE_CORENRN", coreneuron_gpu)

    params.coreneuron = SimpleNamespace(
        enable=coreneuron_enable,
        gpu=coreneuron_gpu,
        file_mode=env_flag("OB_CORENRN_FILE_MODE", False),
        verbose=env_int("OB_CORENRN_VERBOSE", 0),
        cell_permute=env_int("OB_CORENRN_CELL_PERMUTE", 2 if coreneuron_gpu else 0),
        warp_balance=env_int("OB_CORENRN_WARP_BALANCE", 128 if coreneuron_gpu else 0),
    )

    parallel_timeout = env_float("OB_PARALLEL_TIMEOUT", None)
    if parallel_timeout is not None:
        params.parallel_timeout = parallel_timeout

    if "OB_ENABLE_STATUS_REPORT" in os.environ:
        params.enable_status_report = env_flag("OB_ENABLE_STATUS_REPORT", params.enable_status_report)

    if "OB_ENABLE_LFP" in os.environ:
        params.enable_lfp = env_flag("OB_ENABLE_LFP", params.enable_lfp)

    if "OB_LEGACY_PARALLEL_DT" in os.environ:
        params.legacy_parallel_dt = env_flag("OB_LEGACY_PARALLEL_DT", params.legacy_parallel_dt)
    elif runtime_mode == "scientific":
        params.legacy_parallel_dt = True
    else:
        params.legacy_parallel_dt = False

    return params


def configure_corenrn_defaults(params):
    coreneuron_cfg = getattr(params, "coreneuron", None)
    if coreneuron_cfg is None or not getattr(coreneuron_cfg, "enable", False):
        return

    from neuron import coreneuron

    warp_balance = getattr(coreneuron_cfg, "warp_balance", None)
    if warp_balance is not None:
        coreneuron.warp_balance = int(warp_balance)


if '-paramset' in sys.argv:
    paramset = sys.argv[sys.argv.index("-paramset") + 1]
else:
    paramset = "ParameterSetBase"

if '-mpi' in sys.argv:
    configure_output_env(paramset, comm=MPI.COMM_WORLD)
else:
    configure_output_env(paramset)

params = build_params(paramset)
configure_corenrn_defaults(params)
ob = OB(params)
