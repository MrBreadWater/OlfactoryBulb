#!/usr/bin/env python3
"""Run batched HFO parameter searches for an authenticated Phoenix notebook session."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import obgpu_experiment_helpers as hlp
from olfactorybulb.hfo_optimizer import (
    build_manual_allocation_remote_config,
    default_campaign_run_config,
    default_hfo_search_space,
    ensure_campaign_dir,
    initialize_campaign,
    infer_remote_template_from_recent_runs,
    load_campaign_state,
    paramiko_auth_probe,
    propose_elite_batch,
    propose_lhs_batch,
    run_hfo_batch,
    score_hfo_batch,
    top_candidate_rows,
)


def _build_configs(allocation: str, total_tasks: int, nranks: int, tstop_ms: float, cell_permute: int):
    template = infer_remote_template_from_recent_runs()
    if template is None:
        raise RuntimeError("Could not infer remote template from recent runs")

    remote_config = build_manual_allocation_remote_config(
        slurm_allocation_job_id=allocation,
        base_template=template,
        total_tasks=total_tasks,
    )
    base_config = default_campaign_run_config(
        remote_config,
        paramset="GammaSignature_EPLI_Provisional_TCOnly",
        nranks=nranks,
        total_tasks=total_tasks,
        tstop_ms=tstop_ms,
        cell_permute=cell_permute,
    )
    return remote_config, base_config


def _load_or_init_campaign(campaign_dir: Path, base_config: dict, search_space):
    campaign_dir = Path(campaign_dir)
    if not campaign_dir.exists():
        campaign_dir = ensure_campaign_dir(campaign_dir.name)
    if not (campaign_dir / "campaign_config.json").exists():
        initialize_campaign(
            campaign_dir,
            base_config=base_config,
            search_space=search_space,
            notes="Autonomous hfo campaign from authenticated session",
        )
    return campaign_dir


def _pick_batch(campaign_dir: Path, search_space, batch_index: int):
    if batch_index == 0:
        return propose_lhs_batch(
            campaign_dir,
            search_space=search_space,
            n_candidates=6,
            seed=20260527,
            stage="seed",
        )
    if batch_index == 1:
        return propose_lhs_batch(
            campaign_dir,
            search_space=search_space,
            n_candidates=8,
            seed=20260528,
            stage="seed-refine",
        )
    return propose_elite_batch(
        campaign_dir,
        search_space=search_space,
        n_candidates=8,
        seed=20260527 + batch_index,
        elite_frac=0.30,
        explore_frac=0.30,
        stage="refine",
    )


def _summary_line(record: dict) -> str:
    pair = record.get("pair_score")
    control = record.get("control_metrics", {}).get("condition_score")
    ketamine = record.get("ketamine_metrics", {}).get("condition_score")
    ratio = record.get("target_contrast_log10")
    return (
        f"candidate={record.get('candidate_id')} pair_score={pair:.3f} "
        f"control={control:.3f} ketamine={ketamine:.3f} target_contrast_log10={ratio:.3f}"
    )


def run_campaign(
    *,
    allocation: str,
    campaign_name: str | None,
    max_batches: int,
    total_tasks: int,
    nranks: int,
    tstop_ms: float,
    cell_permute: int,
    early_stop_score: float,
    verify_auth: bool,
) -> Path:
    search_space = default_hfo_search_space()
    remote_config, base_config = _build_configs(allocation, total_tasks, nranks, tstop_ms, cell_permute)

    if verify_auth:
        probe = paramiko_auth_probe(remote_config)
        if int(probe.get("returncode", 1)) != 0:
            raise RuntimeError(f"Authentication probe failed: {probe}")

    campaign_slug = campaign_name or f"hfo_epli_live_{hlp.make_timestamp()}"
    campaign_dir = _load_or_init_campaign(Path("/home/alek/OlfactoryBulb/results/notebook_runs/optimization") / campaign_slug, base_config, search_space)

    state = load_campaign_state(campaign_dir)
    print(json.dumps({"campaign_dir": str(campaign_dir), "state": state}, indent=2))

    completed = len(state.get("completed_batches", []))
    for _ in range(completed, max_batches):
        batch_plan = _pick_batch(campaign_dir, search_space, _)
        print(json.dumps({"status": "launch", "batch": batch_plan["batch_name"], "strategy": batch_plan["strategy"]}))

        sweep = run_hfo_batch(
            campaign_dir,
            base_config=base_config,
            batch_plan=batch_plan,
            ketamine_block_values={"control": 1.0, "ketamine": 0.0},
        )

        scored = score_hfo_batch(
            campaign_dir,
            batch_plan=batch_plan,
            sweep=sweep,
        )
        candidates = sorted(
            scored["candidate_rows"],
            key=lambda row: row.get("pair_score", float("-inf")),
            reverse=True,
        )

        if candidates:
            print(json.dumps({"status": "batch_completed", "batch": batch_plan["batch_name"], "best": candidates[0]["candidate_id"], "pair_score": candidates[0]["pair_score"]}))
            for candidate in candidates[:3]:
                print(_summary_line(candidate))
        else:
            print(json.dumps({"status": "batch_completed", "batch": batch_plan["batch_name"], "note": "no scored candidates"}))

        global_top = top_candidate_rows(campaign_dir, limit=1)
        if global_top:
            best_pair = float(global_top[0].get("pair_score", float("-inf")))
            if best_pair >= early_stop_score:
                print(json.dumps({"status": "early_stop", "reason": "pair score reached target", "pair_score": best_pair}))
                break

    return campaign_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation", required=True, help="Existing Phoenix allocation/job id")
    parser.add_argument("--campaign", default=None, help="Campaign folder under results/notebook_runs/optimization")
    parser.add_argument("--max-batches", default=10, type=int)
    parser.add_argument("--total-tasks", default=120, type=int)
    parser.add_argument("--nranks", default=15, type=int)
    parser.add_argument("--tstop-ms", default=9000.0, type=float)
    parser.add_argument("--cell-permute", default=0, type=int)
    parser.add_argument("--early-stop-score", default=2.2, type=float)
    parser.add_argument("--verify-auth", action="store_true", help="Run auth probe in this process before starting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = run_campaign(
        allocation=args.allocation,
        campaign_name=args.campaign,
        max_batches=args.max_batches,
        total_tasks=args.total_tasks,
        nranks=args.nranks,
        tstop_ms=args.tstop_ms,
        cell_permute=args.cell_permute,
        early_stop_score=args.early_stop_score,
        verify_auth=args.verify_auth,
    )
    print(json.dumps({"status": "done", "campaign_dir": str(campaign_dir)}, indent=2))


if __name__ == "__main__":
    main()
