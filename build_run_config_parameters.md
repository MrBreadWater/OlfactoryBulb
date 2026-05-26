# `build_run_config` Parameters

Reference for the controllable keys returned by `build_run_config()` in
[obgpu_experiment_helpers.py](/home/alek/OlfactoryBulb/obgpu_experiment_helpers.py).

Status as of 2026-05-26: this is a human-maintained notebook-facing reference,
not generated source of truth. The authoritative defaults remain in
`build_run_config(...)`, `build_slurm_remote_config(...)`, and
`control_help()`.

Defaults below are the normalized defaults from the helper itself, not necessarily the underlying paramset defaults. A value of `None` generally means "leave the paramset/mechanism default unchanged unless explicitly overridden."

## Notes

- `mode` changes a few defaults:
  - `fast`: `nranks=1`, `legacy_parallel_dt=False`
  - `parity`: `nranks=2`, `legacy_parallel_dt=True`
- `mpi_exec` is environment-dependent:
  - inside Slurm: `srun --mpi=${OB_SLURM_MPI_TYPE:-pmix}`
  - otherwise: `mpiexec`
- Most remote fields are only used when `runner_backend` is `sol_slurm` or `slurm_remote`.

## Global run and runtime controls

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `mode` | `"fast"` | global | Helper preset controlling some other defaults. |
| `paramset` | `"GammaSignature"` | global | Base paramset class to instantiate. |
| `label_prefix` | `"obgpu_experiment"` | global | Prefix for timestamped result directories. |
| `results_base` | `results/notebook_runs` | global | Local result root. |
| `nranks` | `1` or `2` by mode | global | MPI rank count. |
| `tstop_ms` | `None` | global | Simulation duration in ms. |
| `sim_dt_ms` | `0.1` | global | Requested simulation dt in ms. |
| `recording_period_ms` | `0.1` | global | Sample period for saved voltages and LFP. |
| `legacy_parallel_dt` | `False` or `True` by mode | global | Preserve old parallel dt behavior or use direct `sim_dt_ms` control. |
| `parallel_timeout` | `None` | global | Optional NEURON parallel timeout override. |
| `rnd_seed` | `None` | global | Random seed for odor/input generation. |
| `enable_reciprocal_synapses` | `True` | MC/TC <-> GC loop | Toggle reciprocal dendrodendritic circuitry. |
| `extra_overrides` | `{}` | global | Raw paramset override dict for anything not surfaced explicitly. |

## Recording and analysis outputs

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `enable_lfp` | `True` | LFP / global | Enable LFP generation. |
| `lfp_electrode_location` | `[116, 1078, -61]` | LFP / global | Probe location in microns. |
| `disable_status_report` | `True` | global | Suppress periodic simulation status output. |
| `record_from_somas` | `["MC", "TC", "GC"]` | MC, TC, GC | Which soma voltages to save. |
| `record_gc_output_events` | `True` | GC output | Save GC -> MC/TC inhibitory event times. |
| `keep_native_lfp_debug_files` | `False` | LFP / CoreNEURON | Keep raw native-LFP artifacts after conversion. |
| `gc_output_bin_ms` | `5.0` | GC output analysis | Bin width for GC output rates. |
| `gc_output_smooth_sigma_ms` | `10.0` | GC output analysis | Gaussian smoothing sigma for GC output rates. |
| `gc_output_max_connections` | `120` | GC output analysis | Max reciprocal connections shown in GC-output raster. |
| `gc_output_rate_normalization` | `"per_target_cell"` | GC output analysis | Rate normalization mode for GC output plots. |
| `input_bin_ms` | `5.0` | input analysis | Bin width for odor/input event rates. |
| `input_smooth_sigma_ms` | `10.0` | input analysis | Gaussian smoothing sigma for input rates. |
| `input_max_segments` | `120` | input analysis | Max stimulated target segments in input raster. |
| `input_rate_normalization` | `"per_target_cell"` | input analysis | Rate normalization mode for input plots. |
| `analysis_dt_ms` | `0.1` | analysis | Time step assumed by downstream analysis helpers. |
| `spectrogram_signal` | `"lfp"` | analysis | Signal used for spectrogram plots. |
| `wavelet_signal` | `"lfp"` | analysis | Signal used for wavelet plots. |
| `max_voltage_traces_per_type` | `4` | analysis | Max voltage traces shown per cell type. |
| `max_spike_raster_cells_per_type` | `24` | analysis | Max raster cells shown per cell type. |

## OSN / afferent input controls

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `input_odors` | `None` | OSN / glomerular input | Explicit odor schedule keyed by onset ms. |
| `input_stimuli` | `None` | OSN / glomerular input | Custom `InputSpec`-driven stimuli schedule. |
| `max_firing_rate_hz` | `None` | OSN | Override ORN firing-rate ceiling. |
| `inhale_duration_ms` | `None` | OSN | Override inhalation duration. |
| `input_syn_tau1_ms` | `None` | OSN -> MC/TC | Fast `Exp2Syn` rise time. |
| `input_syn_tau2_ms` | `None` | OSN -> MC/TC | Fast `Exp2Syn` decay time. |

## MC and TC input weighting

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `mc_input_weight` | `None` | MC | MC odor-input synaptic weight override. |
| `tc_input_weight` | `None` | TC | TC odor-input synaptic weight override. |
| `mc_input_delay_ms` | `None` | MC | MC odor-input delay override. |
| `tc_input_delay_ms` | `None` | TC | TC odor-input delay override. |

## MC and TC coupling

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `gap_mc` | `None` | MC | MC gap-junction conductance override. |
| `gap_tc` | `None` | TC | TC gap-junction conductance override. |

## Fast excitatory / inhibitory loop

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `ampa_nmda_gmax` | `None` | recurrent glutamatergic synapses | Global `AmpaNmdaSyn.gmax` override. |
| `ampa_nmda_nmdafactor` | `None` | recurrent glutamatergic synapses | Global `AmpaNmdaSyn.nmdafactor` override. |
| `ketamine_block` | `None` | NMDA component | Multiplier on `AmpaNmdaSyn` NMDA current. |
| `ampa_block` | `None` | AMPA component | Multiplier on `AmpaNmdaSyn` AMPA current. |
| `gaba_gmax` | `None` | GC -> MC/TC inhibition | Global `GabaSyn.gmax` override. |
| `gaba_tau2_ms` | `None` | GC -> MC/TC inhibition | Global `GabaSyn.tau2` override. |

## Kainate receptor controls

These are the main knobs for the KAR hypothesis work.

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `kar_mt_gmax` | `None` | OSN -> MC/TC KAR | Slow KAR conductance on MC/TC tuft inputs. |
| `enable_gc_kar` | `None` | MC/TC -> GC KAR | Enable optional KAR synapses on GCs. |
| `kar_gc_gmax` | `None` | MC/TC -> GC KAR | Slow KAR conductance on GCs. |
| `kar_tau1_ms` | `None` | KAR kernel | KAR rise time override. |
| `kar_tau2_ms` | `None` | KAR kernel | KAR decay time override. |
| `kar_tau3_ms` | `None` | KAR kernel | Slow KAR tail time constant override. |
| `kar_amp1` | `None` | KAR kernel | First conductance-kernel amplitude. |
| `kar_amp2` | `None` | KAR kernel | Second conductance-kernel amplitude. |
| `kar_amp3` | `None` | KAR kernel | Third conductance-kernel amplitude. |
| `kar_kd` | `None` | KAR nonlinearity | Half-saturation for event-driven glutamate proxy. |
| `kar_block` | `None` | KAR current | Multiplier on KAR current. |
| `kar_osn_weight_scale` | `None` | OSN -> MC/TC KAR | Scale factor on OSN event weights into KAR synapses. |
| `kar_gc_weight_scale` | `None` | MC/TC -> GC KAR | Scale factor on reciprocal excitation event weights into GC KAR synapses. |

## Granule-cell-specific excitability

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `gc_ka_gbar_scale` | `None` | GC `I_A` / KA | Scale GC A-type potassium conductance. `0` removes GC `I_A`. |

## Direct structural edits

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `add_connections` | `[]` | structure | Add new synaptic connections between existing cells. |
| `modify_connections` | `[]` | structure | Modify existing synaptic connections. |
| `swap_cell_types` | `[]` | structure | Swap selected instantiated cells to another type. |

## Local execution backend

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `runner_backend` | `"local"` | execution | Execution backend: local or remote Slurm. |
| `use_corenrn` | `None` | execution | Local CoreNEURON toggle. |
| `use_gpu` | `None` | execution | Local GPU toggle. |
| `cell_permute` | `2` | CoreNEURON | CoreNEURON cell permutation mode. |
| `mpi_exec` | environment-dependent | local execution | MPI launcher used for local notebook runs. |

## Remote Slurm execution

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `remote_mpi_exec` | `"srun --mpi=pmix_v4 --cpu-bind=none"` | remote execution | MPI launcher on the remote host. |
| `remote_host` | `None` | remote execution | SSH target. |
| `remote_repo_root` | `None` | remote execution | Absolute remote repo path. |
| `remote_results_root` | `None` | remote execution | Absolute remote result root. |
| `remote_conda_activate_cmd` | `"source tools/setup/activate_obgpu.sh"` | remote execution | Primary remote environment activation command. |
| `remote_runtime_profiles` | `[]` | remote execution | Ordered runtime-profile selectors. |
| `remote_fallback_conda_activate_cmd` | `None` | remote execution | Fallback remote activation command. |
| `remote_fast_node_feature` | `None` | remote execution | Required feature for the primary remote runtime. |
| `remote_mechanism_profile` | `"default"` | remote execution | Primary mechanism build/cache profile. |
| `remote_fallback_mechanism_profile` | `"portable"` | remote execution | Fallback mechanism profile. |
| `remote_repo_mode` | `"shared"` | remote execution | Remote repo checkout strategy. |
| `remote_git_ref` | `None` | remote execution | Git ref/commit to run remotely. |
| `remote_git_fetch` | `False` | remote execution | Fetch remote Git before using `remote_git_ref`. |
| `remote_git_remote` | `"origin"` | remote execution | Remote Git remote name. |
| `slurm_allocation_job_id` | `None` | Slurm | Existing allocation/job to reuse. |
| `slurm_reuse_allocation` | `False` | Slurm | Cache and reuse a Slurm allocation. |
| `slurm_allocation_time` | `None` | Slurm | Walltime for reusable allocation. |
| `slurm_allocation_name` | `None` | Slurm | Job-name prefix for reusable allocation. |
| `remote_poll_interval_s` | `1.0` | remote execution | Poll interval for remote status. |
| `remote_live_status` | `True` | remote execution | Emit live Slurm state updates. |
| `remote_live_logs` | `True` | remote execution | Stream remote bootstrap/stdout/stderr/slurm logs. |
| `remote_heartbeat_timeout_s` | `120` | remote execution | Watchdog timeout for notebook-managed remote jobs. |
| `remote_cleanup_stale_allocations` | `True` | remote execution | Cancel stale reusable allocations before a new run. |
| `remote_sync_compress` | `True` | remote execution | Stream compressed remote results back over Paramiko. |
| `remote_preserve_paramiko_session` | `True` | remote execution | Fail closed instead of prompting for a fresh SSH login after the notebook already authenticated once. |
| `slurm_partition` | `None` | Slurm | Partition to submit to. |
| `slurm_account` | `None` | Slurm | Slurm account. |
| `slurm_time` | `None` | Slurm | Slurm walltime string. |
| `slurm_gpus` | `None` | Slurm | GPU count request. |
| `slurm_cpus_per_task` | `None` | Slurm | CPU-per-task request. |
| `slurm_mem` | `None` | Slurm | Memory request string. |
| `slurm_extra_args` | `[]` | Slurm | Extra raw `sbatch` arguments. |

## SSH transport

| Key | Default | Applies to | Meaning |
| --- | --- | --- | --- |
| `ssh_options` | `[]` | remote transport | Extra SSH options. |
| `ssh_transport` | `"paramiko"` | remote transport | Deprecated compatibility guard. `auto` and `paramiko` are accepted; `openssh` is rejected. |
| `ssh_keepalive_s` | `30` | remote transport | Paramiko keepalive interval. |

The old OpenSSH control-master and rsync path was removed. Remote shell
commands, file uploads, streamed result sync, selected-file sync, and deferred
artifact fetches all use the cached Paramiko session.

## Quick interpretation by cell population

### OSN / afferent drive

- `input_odors`
- `input_stimuli`
- `max_firing_rate_hz`
- `inhale_duration_ms`
- `input_syn_tau1_ms`
- `input_syn_tau2_ms`

### MC-specific

- `mc_input_weight`
- `mc_input_delay_ms`
- `gap_mc`

### TC-specific

- `tc_input_weight`
- `tc_input_delay_ms`
- `gap_tc`

### Shared MC/TC excitatory state

- `ampa_nmda_gmax`
- `ampa_nmda_nmdafactor`
- `ketamine_block`
- `ampa_block`
- `kar_mt_gmax`
- `kar_osn_weight_scale`

### GC-specific / reciprocal inhibition

- `gaba_gmax`
- `gaba_tau2_ms`
- `record_gc_output_events`
- `enable_gc_kar`
- `kar_gc_gmax`
- `kar_gc_weight_scale`
- `gc_ka_gbar_scale`

### Global KAR kernel shape

- `kar_tau1_ms`
- `kar_tau2_ms`
- `kar_tau3_ms`
- `kar_amp1`
- `kar_amp2`
- `kar_amp3`
- `kar_kd`
- `kar_block`
