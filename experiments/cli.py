"""Local protocol commands. Live provider generation is intentionally absent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .providers import preflight
from .run_calibration import run
from .schema import Condition, TaskSet


def main() -> None:
    parser = argparse.ArgumentParser(prog="uel-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-tasks", help="validate a task-set JSON file")
    validate.add_argument("path", type=Path)
    provider_preflight = commands.add_parser("preflight-providers", help="list accessible models without generation")
    provider_preflight.add_argument("--output", type=Path, default=Path("artefacts/generated/provider-preflight.json"))
    calibration = commands.add_parser("run-calibration", help="run the frozen 24-output generation calibration")
    calibration.add_argument("--config", type=Path, default=Path("configs/calibration_v1.json"))
    calibration.add_argument("--provider", choices=["openai", "glm", "deepseek", "xai", "mistral"])
    coverage = commands.add_parser("run-coverage", help="run the frozen paired model-variant sweep")
    coverage.add_argument("--config", type=Path, default=Path("configs/model_variant_coverage_v1.json"))
    coverage.add_argument("--provider", choices=["openai", "glm", "deepseek", "xai", "mistral"])
    coverage.add_argument("--model", help="exact model ID; use with --provider for a paired smoke or resumed subset")
    coverage.add_argument("--task-id", help="restrict an exact repair run to one configured task")
    coverage.add_argument(
        "--condition", choices=[item.value for item in Condition],
        help="restrict an exact repair run to one configured condition",
    )
    coverage.add_argument(
        "--retry-recorded-errors", action="store_true",
        help="create a later attempt after adapter hardening; requires exact --provider and --model",
    )
    args = parser.parse_args()
    if args.command == "validate-tasks":
        taskset = TaskSet.from_path(args.path)
        print(f"valid: {taskset.taskset_id} ({len(taskset.tasks)} tasks, schema {taskset.schema_version})")
    elif args.command == "preflight-providers":
        report = preflight(args.output)
        summary = ", ".join(f"{item['provider']}={item['status']}" for item in report["providers"])
        print(f"preflight saved to {args.output}: {summary}")
    elif args.command == "run-calibration":
        paths = run(args.config, provider_filter=args.provider)
        print(f"calibration wrote {len(paths)} local run directories")
    elif args.command == "run-coverage":
        if args.retry_recorded_errors and not (args.provider and args.model):
            parser.error("--retry-recorded-errors requires both --provider and --model")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        paths = run(
            args.config, result_root=Path(config.get("result_root", "results/raw")),
            provider_filter=args.provider, model_filter=args.model,
            retry_recorded_errors=args.retry_recorded_errors,
            task_id_filter=args.task_id, condition_filter=args.condition,
        )
        print(f"coverage sweep wrote {len(paths)} append-only run attempts")


if __name__ == "__main__":
    main()
