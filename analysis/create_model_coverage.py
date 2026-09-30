"""Freeze the first no-tools paired sweep from the current provider snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from experiments.costs import rates_for

REGISTRY = Path("configs/model_registry_v1.json")
OUTPUT = Path("configs/model_variant_coverage_v1_1.json")
OPENAI_OUTPUT_CAP_OVERRIDES = {
    "babbage-002": 4096,
    "davinci-002": 8192,
    "gpt-3.5-turbo-instruct": 512,
    "gpt-3.5-turbo-instruct-0914": 512,
    "gpt-5-mini-2025-08-07": 2048,
    "gpt-5-mini": 2048,
    "gpt-5-nano-2025-08-07": 4096,
    "gpt-5-nano": 4096,
    "gpt-5-2025-08-07": 2048,
    "gpt-5": 1024,
    "gpt-5.1": 512,
    "gpt-5.1-2025-11-13": 512,
    "gpt-5.2": 512,
    "gpt-5.2-2025-12-11": 512,
    "gpt-5.3-codex": 512,
    "gpt-5.4-2026-03-05": 512,
    "gpt-5.4-nano-2026-03-17": 512,
    "gpt-5.5": 512,
    "gpt-5.4-nano": 512,
    "gpt-5.6-sol": 512,
    "gpt-6-astra": 512,
    "gpt-6-luna": 512,
    "o1": 768,
    "o1-2024-12-17": 768,
    "o1-pro": 512,
    "o1-pro-2025-03-19": 512,
    "o3": 1024,
    "o3-2025-04-16": 512,
    "o3-mini": 512,
    "o3-mini-2025-01-31": 1024,
    "o4-mini": 1024,
    "o4-mini-2025-04-16": 1024,
    "gpt-5-pro-2025-10-06": 2048,
    "gpt-5-pro": 1536,
}
OTHER_OUTPUT_CAP_OVERRIDES = {
    "glm": {
        "glm-4.5": 4096, "glm-4.5-air": 8192, "glm-4.6": 1024,
        "glm-4.7": 1024, "glm-5": 2048, "glm-5-turbo": 2048,
        "glm-5.1": 2048, "glm-5.2": 1024, "glm-5.3": 4096,
        "glm-5.3-flash": 2048, "glm-5.3-flashx": 2048,
    },
    "deepseek": {"deepseek-flash": 1024, "deepseek-v4-pro": 2048},
    # xAI's Responses API is the only supported route for these variants; reserve
    # more output budget for the multi-agent synthesis response.
    "xai": {
        "grok-4.20-multi-agent": 1024,
        "grok-4.20-multi-agent-latest": 1024,
        "grok-4.20-multi-agent-beta-latest": 1024,
        "grok-4.20-multi-agent-experimental-beta-0304": 1024,
        "grok-4.20-multi-agent-experimental-beta-latest": 1024,
        "grok-4.20-multi-agent-0309": 1024,
        "grok-4.20-multi-agent-beta-0309": 1024,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--task-id", default="humaneval-0")
    parser.add_argument("--task-ids", nargs="+", help="freeze a multi-task continuation cohort")
    parser.add_argument(
        "--providers", nargs="+", choices=["openai", "glm", "deepseek", "xai", "mistral"],
        help="include only these providers; defaults to every included provider",
    )
    parser.add_argument(
        "--exclude-model", action="append", default=[],
        help="exclude an exact model ID from this frozen campaign; may be repeated",
    )
    parser.add_argument("--result-root", type=Path, help="reuse an existing campaign root when continuing it")
    parser.add_argument("--cohort-note", help="record a concise cohort selection note in the config")
    parser.add_argument("--default-max-tokens", type=int, default=256)
    parser.add_argument("--pro-max-tokens", type=int, default=1024)
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selected_providers = set(args.providers) if args.providers else None
    excluded_models = set(args.exclude_model)
    models = []
    missing_rates = []
    for entry in registry["models"]:
        if not entry["disposition"].startswith("include"):
            continue
        if selected_providers is not None and entry["provider"] not in selected_providers:
            continue
        if entry["model_id"] in excluded_models:
            continue
        rates = rates_for(entry["provider"], entry["model_id"], entry.get("catalog_metadata"))
        if rates is None:
            missing_rates.append(f"{entry['provider']}:{entry['model_id']}")
            continue
        model_spec = {
            "provider": entry["provider"], "model_id": entry["model_id"],
            "canonical_id": entry["canonical_id"], "variant_kind": entry["variant_kind"],
            "catalog_metadata": entry.get("catalog_metadata", {}),
            "rates_usd_per_million": {"input": rates[0], "output": rates[1]},
        }
        if entry["provider"] == "openai" and entry["model_id"].startswith("gpt-") and "-pro" in entry["model_id"]:
            # GPT Pro API variants only accept high reasoning effort.
            model_spec["reasoning_effort"] = "high"
            # Short caps can be consumed entirely by reasoning; size the visible
            # code budget for the Pro family's higher reasoning-token share.
            model_spec["max_tokens"] = args.pro_max_tokens
        if entry["provider"] == "openai" and entry["model_id"] in OPENAI_OUTPUT_CAP_OVERRIDES:
            model_spec["max_tokens"] = OPENAI_OUTPUT_CAP_OVERRIDES[entry["model_id"]]
        elif entry["model_id"] in OTHER_OUTPUT_CAP_OVERRIDES.get(entry["provider"], {}):
            model_spec["max_tokens"] = OTHER_OUTPUT_CAP_OVERRIDES[entry["provider"]][entry["model_id"]]
        models.append(model_spec)
    if missing_rates:
        raise RuntimeError("Missing conservative rate mapping: " + ", ".join(missing_rates))
    config = {
        "protocol_version": "2.1",
        "campaign_id": args.output.stem.replace("_", "-"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "registry_path": str(REGISTRY),
        "registry_snapshot_date": registry["generated_at"],
        "taskset_path": "configs/humaneval_syntax_pilot_v1.json",
        "task_ids": args.task_ids or [args.task_id],
        "conditions": ["baseline", "shifted_baseline"],
        "replicates": 1,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "medium",
        "reasoning_effort_policy": "OpenAI GPT Pro IDs use high as required by the API; per-model output caps are recorded in each manifest",
        "max_tokens": args.default_max_tokens,
        "timeout_seconds": 300,
        "max_attempts": 2,
        "evaluation_mode": "docker_isolated",
        "result_root": str(args.result_root or Path(f"results/raw/{args.output.stem.replace('_', '-')}")),
        "budget_accounting_root": "results/raw",
        "budget_limit_per_provider_usd": 5.0,
        "budget_holdback_usd": 0.25,
        "retry_policy": "one automatic retry only for explicit HTTP 429; preserve every attempt; do not resend transport timeouts or ambiguous 5xx responses",
        "rate_limit_recovery_delay_seconds": 120,
        "failure_policy": "provider/evaluator errors are infrastructure records; valid generated code failures remain outcomes and are never resampled until correct",
        "search_specialized_models": "kept in registry as separate exploratory cohort; not mixed into no-tools primary sweep",
        "models": models,
    }
    if args.cohort_note:
        config["cohort_note"] = args.cohort_note
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    by_provider: dict[str, int] = {}
    for model in models:
        by_provider[model["provider"]] = by_provider.get(model["provider"], 0) + 1
    print(f"wrote {len(models)} targets to {args.output}; {len(models) * len(config['task_ids']) * 2} planned outputs")
    for provider, count in sorted(by_provider.items()):
        print(f"{provider}: {count} targets")


if __name__ == "__main__":
    main()
