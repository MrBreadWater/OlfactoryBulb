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


def _meets_criteria(
    row: dict,
    *,
    min_ketamine_target: float,
    max_control_target: float,
    min_ketamine_peak_ratio: float,
    min_target_contrast_log10: float,
    max_control_score: float,
) -> bool:
    control_metrics = row.get("control_metrics") or {}
    ketamine_metrics = row.get("ketamine_metrics") or {}
    control_relative = (control_metrics.get("relative_band_power") or {}).get("target_hfo", 0.0)
    ketamine_relative = (ketamine_metrics.get("relative_band_power") or {}).get("target_hfo", 0.0)
    control_score = float(control_metrics.get("condition_score", -float("inf")))
    ketamine_peak_ratio = float(ketamine_metrics.get("peak_ratio", 0.0))
    contrast = float(row.get("target_contrast_log10", -float("inf")))

    return (
        control_relative <= float(max_control_target)
        and ketamine_relative >= float(min_ketamine_target)
        and ketamine_peak_ratio >= float(min_ketamine_peak_ratio)
        and contrast >= float(min_target_contrast_log10)
        and control_score <= float(max_control_score)
    )


def _candidate_sort_key(row: dict, *, require_criteria: bool) -> tuple[int, float]:
    meets = bool(row.get("meets_criteria", False))
    score = float(row.get("pair_score", float("-inf")))
    if not require_criteria:
        return (1, score)
    return (1, score) if meets else (0, score)


def _summary_line(record: dict) -> str:
    status = "PASS" if record.get("meets_criteria") else "FAIL"
    pair = float(record.get("pair_score", float("-inf")))
    control = record.get("control_metrics", {}).get("condition_score", float("-inf"))
    ketamine = record.get("ketamine_metrics", {}).get("condition_score", float("-inf"))
    ratio = record.get("target_contrast_log10", float("-inf"))
    return (
        f"[{status}] candidate={record.get('candidate_id')} pair_score={pair:.3f} "
        f"control={control:.3f} ketamine={ketamine:.3f} target_contrast_log10={ratio:.3f}"
    )


def _annotate_criteria(
    candidates: list[dict],
    *,
    min_ketamine_target: float,
    max_control_target: float,
    min_ketamine_peak_ratio: float,
    min_target_contrast_log10: float,
    max_control_score: float,
) -> list[dict]:
    for row in candidates:
        row["meets_criteria"] = _meets_criteria(
            row,
            min_ketamine_target=min_ketamine_target,
            max_control_target=max_control_target,
            min_ketamine_peak_ratio=min_ketamine_peak_ratio,
            min_target_contrast_log10=min_target_contrast_log10,
            max_control_score=max_control_score,
        )
    return candidates


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
    min_ketamine_target: float,
    max_control_target: float,
    min_ketamine_peak_ratio: float,
    min_target_contrast_log10: float,
    max_control_score: float,
    require_criteria_for_early_stop: bool,
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
    for batch_idx in range(completed, max_batches):
        batch_plan = _pick_batch(campaign_dir, search_space, batch_idx)
        print(
            json.dumps(
                {
                    "status": "launch",
                    "batch": batch_plan["batch_name"],
                    "strategy": batch_plan["strategy"],
                }
            )
        )

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
        candidates = _annotate_criteria(
            scored["candidate_rows"],
            min_ketamine_target=min_ketamine_target,
            max_control_target=max_control_target,
            min_ketamine_peak_ratio=min_ketamine_peak_ratio,
            min_target_contrast_log10=min_target_contrast_log10,
            max_control_score=max_control_score,
        )

        candidates = sorted(
            candidates,
            key=lambda row: _candidate_sort_key(
                row,
                require_criteria=require_criteria_for_early_stop,
            ),
            reverse=True,
        )

        if candidates:
            print(
                json.dumps(
                    {
                        "status": "batch_completed",
                        "batch": batch_plan["batch_name"],
                        "best": candidates[0].get("candidate_id"),
                        "pair_score": candidates[0].get("pair_score"),
                        "meets_criteria": candidates[0].get("meets_criteria"),
                    }
                )
            )
            for candidate in candidates[:3]:
                print(_summary_line(candidate))
        else:
            print(json.dumps({"status": "batch_completed", "batch": batch_plan["batch_name"], "note": "no scored candidates"}))

        global_top = top_candidate_rows(campaign_dir, limit=1)
        if global_top:
            best = global_top[0]
            best = _annotate_criteria(
                [best],
                min_ketamine_target=min_ketamine_target,
                max_control_target=max_control_target,
                min_ketamine_peak_ratio=min_ketamine_peak_ratio,
                min_target_contrast_log10=min_target_contrast_log10,
                max_control_score=max_control_score,
            )[0]
            if not require_criteria_for_early_stop or best.get("meets_criteria", False):
                best_pair = float(best.get("pair_score", float("-inf")))
                if best_pair >= early_stop_score:
                    print(
                        json.dumps(
                            {
                                "status": "early_stop",
                                "reason": "pair score reached target",
                                "pair_score": best_pair,
                                "candidate_id": best.get("candidate_id"),
                                "meets_criteria": best.get("meets_criteria"),
                            }
                        )
                    )
                    break
            else:
                print(
                    json.dumps(
                        {
                            "status": "best_candidate_rejected",
                            "reason": "global best fails criteria",
                            "candidate_id": best.get("candidate_id"),
                            "pair_score": best.get("pair_score"),
                        }
                    )
                )

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
    parser.add_argument("--min-ketamine-target", default=0.05, type=float, help="Min ketamine target-band relative power")
    parser.add_argument("--max-control-target", default=0.01, type=float, help="Max control target-band relative power")
    parser.add_argument("--min-ketamine-peak-ratio", default=2.0, type=float, help="Min ketamine peak prominence ratio")
    parser.add_argument("--min-target-contrast-log10", default=0.20, type=float, help="Min ketamine-control target contrast in log10")
    parser.add_argument("--max-control-score", default=1.0, type=float, help="Max allowed control condition score")
    parser.add_argument(
        "--require-criteria-for-early-stop",
        action="store_true",
        default=True,
        help="Require candidate criteria before stopping early",
    )
    parser.add_argument(
        "--no-criteria-early-stop",
        dest="require_criteria_for_early_stop",
        action="store_false",
        help="Do not require criteria check for early stop",
    )
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
        min_ketamine_target=args.min_ketamine_target,
        max_control_target=args.max_control_target,
        min_ketamine_peak_ratio=args.min_ketamine_peak_ratio,
        min_target_contrast_log10=args.min_target_contrast_log10,
        max_control_score=args.max_control_score,
        require_criteria_for_early_stop=args.require_criteria_for_early_stop,
        verify_auth=args.verify_auth,
    )
    print(json.dumps({"status": "done", "campaign_dir": str(campaign_dir)}, indent=2))


if __name__ == "__main__":
    main()
