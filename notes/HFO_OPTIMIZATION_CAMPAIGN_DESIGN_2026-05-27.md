HFO optimization campaign design (2026-05-27)

Goal
- Find parameter regimes that produce a clean LFP band near 180 +/- 20 Hz under ketamine-like NMDA block.
- Reward regimes where that target HFO stands out more strongly with ketamine block than without it.
- Keep receptor and membrane time constants fixed; vary conductance-like knobs and related weights only.

Why this is batch-first, not Nelder-Mead
- The objective is expensive, noisy, and likely non-smooth because oscillations can appear or disappear abruptly.
- Nelder-Mead is fundamentally serial and wastes a large Phoenix allocation.
- Phoenix is better used by evaluating many independent candidates concurrently inside one long-lived allocation.
- The implemented strategy is:
  1. Latin-hypercube seeding for global coverage.
  2. Paired control / ketamine evaluation for every candidate.
  3. Elite-centered refinement with truncated Gaussian proposals plus explicit exploration.

Remote execution model
- Reuse the user's existing manual Slurm allocation by setting `slurm_allocation_job_id`.
- Do not request a new allocation from the optimizer notebook.
- Let the sweep driver launch many independent `srun` MPI steps inside that allocation.
- Throughput is controlled by:
  - `nranks` per item
  - `sweep_parallelism = floor(total_tasks / nranks)`
- For a 120-task allocation:
  - `nranks=15` -> `8` concurrent items
  - `nranks=12` -> `10` concurrent items
  - `nranks=10` -> `12` concurrent items

Default search dimensions
- `kar_mt_gmax`
- `kar_gc_gmax`
- `gaba_gmax`
- `ampa_nmda_gmax`
- `gap_tc`
- `gap_mc`
- `tc_input_weight`
- `mc_input_weight`

Condition pairing
- Every candidate is run twice:
  - `control`: `ketamine_block = 1.0`
  - `ketamine`: `ketamine_block = 0.0`
- The optimizer scores the pair, not just the ketamine condition in isolation.

Condition-level score components
- peak frequency match to 180 Hz with a 20 Hz scale
- target-band relative power in 160-200 Hz
- peak prominence over broad and local shoulders
- dominance of 160-200 Hz over side HFO bands
- modest support from beta/gamma power
- phase locking of TC/MC/EPLI spikes to the target band
- penalty for implausibly high mean firing rates

Pair-level score components
- ketamine condition score
- positive contrast in target-band relative power
- positive contrast in peak prominence
- penalty when the control condition is also too HFO-strong

Campaign files
- `campaign_config.json`
- `state.json`
- `candidate_archive.jsonl`
- `item_archive.jsonl`
- `batches/batch_XXXX_plan.json`
- `batches/batch_XXXX_run.json`
- `batches/batch_XXXX_scored.json`

Expected workflow
1. Open the notebook.
2. Set the existing manual allocation job id.
3. Run the Paramiko auth probe.
4. Initialize the campaign directory.
5. Launch one LHS seed batch.
6. Score it.
7. Launch elite-refinement batches iteratively.
8. Re-run top candidates at longer duration if needed.

Interpretation discipline
- A high score means the model reproduces the target spectral signature under the current scoring assumptions.
- A high score does not by itself prove biological correctness.
- If the optimizer keeps pushing parameters to the search boundary, expand or revise the search space deliberately instead of silently trusting the result.

## Phoenix-authenticated notebook-run workflow (interactive continuation)

Because Phoenix SSH on `localhost:2223` requires Duo, this environment cannot initiate new remote runs non-interactively. Use the existing authenticated notebook kernel (the one where you already passed `paramiko_auth_probe`) to run autonomous batches.

From that notebook kernel:

```python
from olfactorybulb.hfo_optimizer import *
import os

# Reuse your authenticated kernel context (do not pass verify_auth in first run)
from tools.run_hfo_campaign import run_campaign

campaign_dir = run_campaign(
    allocation="14537854",   # replace with your active 7-day job id
    campaign_name="epli_live",  # optional: any string under results/notebook_runs/optimization
    max_batches=10,
    total_tasks=120,
    nranks=15,
    tstop_ms=9000.0,
    cell_permute=0,
    early_stop_score=2.2,
    verify_auth=False,
)
print(campaign_dir)
```

This keeps the 120-task allocation saturated and records:
- `batches/batch_*.json`
- `candidate_archive.jsonl`
- `item_archive.jsonl`

If needed later, stop and inspect top candidates:

```python
from olfactorybulb.hfo_optimizer import top_candidate_rows
rows = top_candidate_rows(campaign_dir, limit=20)
rows[:5]
```
