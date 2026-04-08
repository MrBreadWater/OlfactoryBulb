# Modern NEURON Port Notes

## Environment

- Env name: `OBGPU`
- Python: `3.11`
- NEURON: source-built `9.0.1`
- CoreNEURON: enabled in the build
- GPU: enabled via a user-local NVIDIA HPC SDK install on this Jetson AGX Orin

Create or refresh the env with:

```bash
ENABLE_GPU=1 ./tools/setup/setup_ob_modern.sh
```

For Jetson AGX Orin on JetPack 5 / CUDA 11.4, install a matching user-local NVHPC toolchain first:

```bash
./tools/setup/setup_nvhpc_jetson.sh
```

Notes:

- The setup script will auto-detect a user-local NVHPC install under `~/.local/nvidia/hpc_sdk` if `nvc`/`nvc++` are not already on `PATH`.
- You can override the search root explicitly with `NVHPC_SDK_ROOT=/path/to/hpc_sdk`.
- [setup_nvhpc_jetson.sh](/home/alek/OlfactoryBulb/tools/setup/setup_nvhpc_jetson.sh) installs NVIDIA HPC SDK `21.7` for `Linux_aarch64` with CUDA `11.4`, which matches this host's JetPack/L4T CUDA stack.
- The setup script now auto-detects the GPU compute capability from `libcudart` and passes it to CMake as `CMAKE_CUDA_ARCHITECTURES`.
- On this machine, the detected GPU is `Orin` with compute capability `8.7`, so the raw detected architecture is `87`.
- NVHPC `21.7` does not support `cc87` directly; the setup script now falls back automatically to the highest supported target not exceeding the detected architecture, which is `86` on this host.
- The setup script also repairs the NVHPC `libnrnmech.so` fatbin temp-object bug after `nrnivmodl -coreneuron` by stripping the bogus `/tmp/pgcudafat...` dependency and resetting the soname.
- If you need to override that manually, set `CUDA_ARCHITECTURES`, for example:

```bash
ENABLE_GPU=1 ENV_NAME=OBGPU CUDA_ARCHITECTURES=87 ./tools/setup/setup_ob_modern.sh
```

## Current Status

The modern GPU port is now working with the original model behavior preserved.

Default fast launch on this Jetson:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate OBGPU
python runbatch.py
```

That batch path now defaults to the faster CoreNEURON GPU layout:

- `1` MPI rank
- GPU `cell_permute=2`
- legacy parallel `dt` behavior preserved
- native CoreNEURON LFP enabled

For the slower parity-preserving mode, override the rank count explicitly:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate OBGPU
OB_MPI_RANKS=2 python runbatch.py
```

## Verified Working

- `python -c "import neuron; from neuron import h, coreneuron; print(neuron.__version__, h.nrnversion())"`
- `mpiexec -n 2 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label modernsrc_onems_r2 --paramset OneMsTest`
- `mpiexec -n 2 nrniv -mpi -python initslice.py -paramset OneMsTest -mpi`
- `mpiexec -n 2 nrniv -mpi -python tools/benchmarks/benchmark_ob.py --label perf_gpu_lfp_r2_perm2_1800ms_v1 --paramset GammaSignature --coreneuron --coreneuron-gpu`

The current parity-preserving GPU configuration is:

- old baseline: [baseline_gamma_r2](/home/alek/OlfactoryBulb/results/benchmarks/baseline_gamma_r2)
- tuned modern GPU: [perf_gpu_lfp_r2_perm2_1800ms_v1](/home/alek/OlfactoryBulb/results/benchmarks/perf_gpu_lfp_r2_perm2_1800ms_v1)

That full `1800 ms` run matches the historical old-source results on the common prefix to numerical noise:

- `input_times.pkl`: exact match
- `lfp.pkl`: max abs diff `1.674643370988549e-07`
- common soma traces: max abs diff `5.322926099893266e-09`

It is also faster than the original old NEURON 7 baseline:

- total time `1106.96 s -> 886.12 s`
- run time `1101.92 s -> 873.95 s`
- total speedup `1.249x`
- run speedup `1.261x`

The faster default `1`-rank GPU mode is:

- [perf_gpu_lfp_r1_perm2_1800ms_v1](/home/alek/OlfactoryBulb/results/benchmarks/perf_gpu_lfp_r1_perm2_1800ms_v1)
- total time `429.86 s`
- run time `394.29 s`
- but it drifts from the historical `2`-rank trajectory later in the run, so it is the speed-first default rather than the parity default

## Repo Changes For The Port

- [model.py](/home/alek/OlfactoryBulb/olfactorybulb/model.py)
  - deferred LFP electrode creation until after `setup_transfer()`
  - removed the hard-coded `pc.timeout(1)` override
  - moved MPI LFP reduction out of per-step callbacks into post-run gathering
  - updated gap-junction target registration to use the point-process form of `pc.target_var(...)`
  - normalized both half-gap constructors to `GapJunction(x, sec=...)`
- [isolated_cells.py](/home/alek/OlfactoryBulb/prev_ob_models/Birgiolas2020/isolated_cells.py)
  - robust mechanism loading for modern installs
- [utils.py](/home/alek/OlfactoryBulb/prev_ob_models/utils.py)
  - mechanism-load helper for source/wheel layouts
- [gapjunction.mod](/home/alek/OlfactoryBulb/prev_ob_models/Birgiolas2020/Mechanisms/gapjunction.mod)
  - restored the legacy `NONSPECIFIC_CURRENT` behavior needed for parity
- [report_event.cpp](/home/alek/OlfactoryBulb/external/nrn-9.0.1/src/coreneuron/io/reports/report_event.cpp)
  - fixed native GPU LFP reporting by syncing only the `fast_imem` RHS array at report time
- [core2nrn_data_return.cpp](/home/alek/OlfactoryBulb/external/nrn-9.0.1/src/coreneuron/io/core2nrn_data_return.cpp)
  - ensured CoreNEURON data-return arrays are synchronized back to host after GPU runs
- [benchmark_ob.py](/home/alek/OlfactoryBulb/tools/benchmarks/benchmark_ob.py)
  - now defaults GPU runs to `cell_permute=2` and records the chosen permutation in `summary.json`

## Practical Recommendation

Use `OBGPU` for:

- speed-first GPU runs on this Jetson via `runbatch.py`
- parity-preserving GPU runs when launched with `OB_MPI_RANKS=2`

Keep `OB` for:

- legacy comparison runs and historical reproduction work
