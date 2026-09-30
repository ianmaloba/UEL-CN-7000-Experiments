"""Freeze the approved DocGround API campaign from existing model targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


MODEL_SOURCE = Path("configs/model_variant_coverage_baseline_v2.json")
TASKSET_PATH = Path("configs/docground_api_pilot_v1_candidate.json")
REVIEW_PATH = Path("results/protocol_review/docground_api_pilot_v1_candidate_v3.json")
CONDITIONS = ["baseline", "shifted_baseline", "shifted_docground"]
EXCLUDED_PROVIDERS = {"openai", "minimax"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_config(
    evaluator_image: str,
    max_tokens: int = 256,
    budget_holdback_usd: float = 0.5,
    protocol_version: str = "docground-api-pilot-v1",
    recovery_reason: str | None = None,
) -> dict[str, object]:
    source_bytes = MODEL_SOURCE.read_bytes()
    source = json.loads(source_bytes)
    taskset_bytes = TASKSET_PATH.read_bytes()
    taskset = json.loads(taskset_bytes)
    review_bytes = REVIEW_PATH.read_bytes()
    review = json.loads(review_bytes)
    if review.get("status") != "approved":
        raise RuntimeError("Six-task review is not approved")
    if any(item.get("approval") != "approve_suggestion" for item in review.get("tasks", [])):
        raise RuntimeError("Every task must have explicit approval")
    if review.get("taskset_sha256") != sha256(taskset_bytes):
        raise RuntimeError("Approved review does not match the task-set file")
    if review.get("taskset_id") != taskset.get("taskset_id"):
        raise RuntimeError("Approved review task-set ID does not match")
    task_ids = [str(item["task_id"]) for item in taskset["tasks"]]
    if {item.get("task_id") for item in review["tasks"]} != set(task_ids):
        raise RuntimeError("Approved review task IDs do not match the task set")

    models = []
    for item in source["models"]:
        if item["provider"] in EXCLUDED_PROVIDERS:
            continue
        model = dict(item)
        model["max_tokens"] = max_tokens
        models.append(model)
    if len(models) != 68:
        raise RuntimeError(f"Expected the frozen 68 non-OpenAI, non-MiniMax variants, found {len(models)}")
    if any(not isinstance(item.get("rates_usd_per_million"), dict) for item in models):
        raise RuntimeError("A selected model is missing its frozen rate mapping")

    config: dict[str, object] = {
        "protocol_version": protocol_version,
        "campaign_id": "docground-api-pilot-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_source_path": str(MODEL_SOURCE),
        "model_source_sha256": sha256(source_bytes),
        "taskset_path": str(TASKSET_PATH),
        "taskset_sha256": sha256(taskset_bytes),
        "task_ids": task_ids,
        "conditions": CONDITIONS,
        "docground_review_path": str(REVIEW_PATH),
        "docground_review_sha256": sha256(review_bytes),
        "docground_version": review["docground_version"],
        "documentation_snapshot_sha256": review["documentation_snapshot_sha256"],
        "replicates": 1,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "medium",
        "max_tokens": max_tokens,
        "timeout_seconds": 300,
        "max_attempts": 2,
        "rate_limit_recovery_delay_seconds": 120,
        "evaluation_mode": "docker_isolated",
        "evaluator_image": evaluator_image,
        "result_root": "results/raw/docground-api-pilot-v1",
        "budget_accounting_root": "results/raw",
        "budget_limit_per_provider_usd": 5.0,
        "budget_holdback_usd": budget_holdback_usd,
        "failure_policy": "Preserve every provider response and attempt. Keep functional failures as outcomes. Fix evaluator infrastructure only by a versioned evaluator image, then reevaluate the original response before considering any new model request.",
        "models": models,
    }
    if recovery_reason:
        config["recovery_reason"] = recovery_reason
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/docground_api_pilot_v1.json"))
    parser.add_argument(
        "--evaluator-image", default="uel-cn7000/docground-api-eval:py3.14.7-v1",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--budget-holdback-usd", type=float, default=0.5)
    parser.add_argument("--protocol-version", default="docground-api-pilot-v1")
    parser.add_argument("--recovery-reason")
    args = parser.parse_args()
    config = build_config(
        args.evaluator_image,
        max_tokens=args.max_tokens,
        budget_holdback_usd=args.budget_holdback_usd,
        protocol_version=args.protocol_version,
        recovery_reason=args.recovery_reason,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(config['models'])} model variants x {len(config['task_ids'])} tasks "
        f"x {len(config['conditions'])} conditions = "
        f"{len(config['models']) * len(config['task_ids']) * len(config['conditions'])} planned outputs"
    )


if __name__ == "__main__":
    main()
