"""Aggregate the frozen paired model-coverage pilot without hiding failed attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from experiments.run_calibration import (
    _historical_spend, _uncertain_billing_reserve, _usage_cost, request_parts,
)
from experiments.schema import Condition, TaskSet


DEFAULT_CONDITIONS = [Condition.BASELINE.value, Condition.SHIFTED_BASELINE.value]
CONDITION_PREFIX = {
    Condition.BASELINE.value: "baseline",
    Condition.SHIFTED_BASELINE.value: "shifted_baseline",
    Condition.SHIFTED_DOCGROUND.value: "shifted_docground",
}


ERROR_STATUSES = {
    "provider_error", "transport_interrupted", "transport_or_parse_error",
    "configuration_error", "evaluation_error",
}
GENERATION_STATUSES = [
    "complete", "incomplete_response", "empty_response", "provider_error",
    "transport_or_parse_error", "transport_interrupted", "configuration_error",
    "evaluation_error", "missing",
]
PAIR_STATUSES = [
    "complete_pair", "mixed_generation_pair", "incomplete_pair",
    "infrastructure_error_pair", "missing_pair",
]
FUNCTION_STATUSES = ["pass", "fail", "timeout", "not_run", "missing"]
COLORS = {
    "complete": "#2b8c6e",
    "incomplete_response": "#e6ab02",
    "empty_response": "#7570b3",
    "provider_error": "#d95f02",
    "transport_or_parse_error": "#e7298a",
    "transport_interrupted": "#a6761d",
    "configuration_error": "#666666",
    "evaluation_error": "#1b9e77",
    "missing": "#d9d9d9",
    "complete_pair": "#2b8c6e",
    "mixed_generation_pair": "#e6ab02",
    "incomplete_pair": "#7570b3",
    "infrastructure_error_pair": "#d95f02",
    "missing_pair": "#d9d9d9",
    "pass": "#2b8c6e",
    "fail": "#d95f02",
    "timeout": "#a6761d",
    "not_run": "#bdbdbd",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _expected_request(
    spec: dict[str, Any], config: dict[str, Any], taskset: TaskSet,
    task_id: str, condition: str, prompt_text: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    task = next((item for item in taskset.tasks if item.task_id == task_id), None)
    if task is None:
        return None
    temperature = float(config.get("temperature", 0))
    top_p = float(config.get("top_p", 1.0))
    max_tokens = int(spec.get("max_tokens", config.get("max_tokens", 256)))
    reasoning = str(spec.get("reasoning_effort", config.get("reasoning_effort", "medium")))
    prompt = prompt_text if prompt_text is not None else task.prompt_for(Condition(condition))
    _, _, route, params = request_parts(
        str(spec["provider"]), str(spec["model_id"]), prompt,
        temperature, top_p, max_tokens, reasoning,
    )
    return route, params


def _approved_docground_prompt(
    taskset: TaskSet, task_id: str, config: dict[str, Any], manifest: dict[str, Any],
) -> str | None:
    review_path_value = config.get("docground_review_path")
    if not isinstance(review_path_value, str) or not review_path_value:
        return None
    review_path = Path(review_path_value)
    try:
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes)
    except (OSError, json.JSONDecodeError):
        return None
    if review.get("status") != "approved":
        return None
    if hashlib.sha256(review_bytes).hexdigest() != manifest.get("docground_review_sha256"):
        return None
    taskset_path = Path(str(config["taskset_path"]))
    try:
        taskset_sha256 = hashlib.sha256(taskset_path.read_bytes()).hexdigest()
    except OSError:
        return None
    if review.get("taskset_id") != taskset.taskset_id or review.get("taskset_sha256") != taskset_sha256:
        return None
    record = next(
        (item for item in review.get("tasks", []) if isinstance(item, dict) and item.get("task_id") == task_id),
        None,
    )
    if not isinstance(record, dict) or record.get("approval") != "approve_suggestion":
        return None
    prompt = record.get("suggested_prompt")
    hashes = record.get("prompt_hashes", {})
    if not isinstance(prompt, str) or not isinstance(hashes, dict):
        return None
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != hashes.get("suggested"):
        return None
    if record.get("documentation_snapshot_sha256") != manifest.get("documentation_snapshot_sha256"):
        return None
    if review.get("docground_version") != manifest.get("docground_version"):
        return None
    return prompt


def load_campaign(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    taskset = TaskSet.from_path(Path(config["taskset_path"]))
    raw_root = Path(config.get("result_root", "results/raw"))
    selected_specs = {
        (str(spec["provider"]), str(spec["model_id"])): spec
        for spec in config["models"]
    }
    task_ids = [str(item) for item in config.get("task_ids", [])]
    conditions = [str(item) for item in config.get("conditions", DEFAULT_CONDITIONS)]

    all_attempts: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for evaluation_path in raw_root.glob("*/evaluation.json"):
        manifest = _read_json(evaluation_path.parent / "run_manifest.json")
        evaluation = _read_json(evaluation_path)
        if manifest is None or evaluation is None:
            continue
        provider = str(manifest.get("model_provider", ""))
        model = str(manifest.get("model_id", ""))
        task_id = str(manifest.get("task_id", ""))
        condition = str(manifest.get("condition", ""))
        if (provider, model) not in selected_specs or (task_ids and task_id not in task_ids):
            continue
        details = evaluation.get("details", {})
        if not isinstance(details, dict):
            details = {}
        task = next((item for item in taskset.tasks if item.task_id == task_id), None)
        expected_prompt = None
        if task is not None:
            if condition == Condition.SHIFTED_DOCGROUND.value:
                expected_prompt = _approved_docground_prompt(taskset, task_id, config, manifest)
            else:
                try:
                    expected_prompt = task.prompt_for(Condition(condition))
                except ValueError:
                    expected_prompt = None
        selected_prompt_path = evaluation_path.parent / "prompt_selected.txt"
        if selected_prompt_path.exists():
            selected_prompt = selected_prompt_path.read_text(encoding="utf-8")
            if selected_prompt.endswith("\n"):
                selected_prompt = selected_prompt[:-1]
        else:
            selected_prompt = expected_prompt
        prompt_matches = bool(
            expected_prompt is not None
            and selected_prompt == expected_prompt
            and manifest.get("prompt_sha256") == hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        )
        expected = _expected_request(
            selected_specs[(provider, model)], config, taskset, task_id, condition,
            prompt_text=expected_prompt,
        ) if expected_prompt is not None else None
        request_matches = bool(
            expected
            and prompt_matches
            and details.get("route") == expected[0]
            and details.get("request_params") == expected[1]
        )
        generation_status = str(details.get("generation_status", "unknown"))
        cost = _usage_cost(provider, model, details)
        row = {
            "run_id": manifest.get("run_id", evaluation_path.parent.name),
            "provider": provider,
            "model_id": model,
            "task_id": task_id,
            "condition": condition,
            "attempt": int(manifest.get("attempt", 1)),
            "max_tokens": manifest.get("max_tokens"),
            "reasoning_effort": manifest.get("reasoning_effort"),
            "request_matches_current_config": request_matches,
            "generation_status": generation_status,
            "finish_reason": details.get("finish_reason"),
            "http_status": details.get("http_status"),
            "provider_error": details.get("provider_error"),
            "input_tokens": details.get("input_tokens"),
            "output_tokens": details.get("output_tokens"),
            "estimated_cost_usd": cost,
            "elapsed_seconds": details.get("elapsed_seconds"),
            "syntax_status": evaluation.get("syntax_status", "not_run"),
            "functional_status": evaluation.get("functional_status", "not_run"),
            "security_status": evaluation.get("security_status", "not_run"),
            "hallucination_status": evaluation.get("hallucination_status", "not_run"),
            "api_conformance_status": evaluation.get("api_conformance_status", "not_applicable"),
            "artifact_dir": str(evaluation_path.parent),
            "_mtime_ns": evaluation_path.stat().st_mtime_ns,
        }
        all_attempts.append(row)
        grouped[(provider, model, task_id, condition)].append(row)

    latest_rows: list[dict[str, Any]] = []
    for provider, model in selected_specs:
        for task_id in task_ids:
            for condition in conditions:
                candidates = grouped.get((provider, model, task_id, condition), [])
                if not candidates:
                    latest_rows.append({
                        "provider": provider, "model_id": model, "task_id": task_id,
                        "condition": condition, "generation_status": "missing",
                        "functional_status": "missing", "syntax_status": "missing",
                        "security_status": "missing", "hallucination_status": "missing",
                        "request_matches_current_config": False, "max_tokens": selected_specs[(provider, model)].get("max_tokens", config.get("max_tokens")),
                    })
                    continue
                matching = [row for row in candidates if row["request_matches_current_config"]]
                pool = matching or candidates
                latest = max(pool, key=lambda row: (row["attempt"], row["_mtime_ns"]))
                latest["selected_for_current_summary"] = True
                latest_rows.append(latest)

    return config, all_attempts, latest_rows


def _pair_status(left: str, right: str) -> str:
    if left == "missing" or right == "missing":
        return "missing_pair"
    if left == right == "complete":
        return "complete_pair"
    if left in ERROR_STATUSES or right in ERROR_STATUSES:
        return "infrastructure_error_pair"
    if left == "complete" or right == "complete":
        return "mixed_generation_pair"
    return "incomplete_pair"


def make_tables(
    config: dict[str, Any], attempts: list[dict[str, Any]], latest: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    latest_by_key = {
        (row["provider"], row["model_id"], row["task_id"], row["condition"]): row
        for row in latest
    }
    pair_rows: list[dict[str, Any]] = []
    conditions = [str(x) for x in config.get("conditions", ["baseline", "shifted_baseline"])]
    specs = config["models"]
    task_ids = [str(x) for x in config.get("task_ids", [])]
    for spec in specs:
        provider, model = str(spec["provider"]), str(spec["model_id"])
        for task_id in task_ids:
            pair: dict[str, Any] = {
                "provider": provider, "model_id": model, "canonical_id": spec.get("canonical_id"),
                "variant_kind": spec.get("variant_kind"), "task_id": task_id,
            }
            condition_statuses: dict[str, str] = {}
            for condition in conditions:
                row = latest_by_key[(provider, model, task_id, condition)]
                prefix = CONDITION_PREFIX.get(condition, condition)
                for field in (
                    "generation_status", "functional_status", "syntax_status",
                    "security_status", "hallucination_status", "finish_reason",
                    "http_status", "max_tokens", "estimated_cost_usd",
                    "request_matches_current_config",
                ):
                    pair[f"{prefix}_{field}"] = row.get(field)
                condition_statuses[condition] = str(row.get("generation_status", "missing"))
            if Condition.BASELINE.value in condition_statuses and Condition.SHIFTED_BASELINE.value in condition_statuses:
                pair["pair_generation_status"] = _pair_status(
                    condition_statuses[Condition.BASELINE.value],
                    condition_statuses[Condition.SHIFTED_BASELINE.value],
                )
            else:
                pair["pair_generation_status"] = "not_paired"
            if Condition.SHIFTED_BASELINE.value in condition_statuses and Condition.SHIFTED_DOCGROUND.value in condition_statuses:
                pair["grounding_core_pair_generation_status"] = _pair_status(
                    condition_statuses[Condition.SHIFTED_BASELINE.value],
                    condition_statuses[Condition.SHIFTED_DOCGROUND.value],
                )
            pair_rows.append(pair)

    pairs = pd.DataFrame(pair_rows)
    attempt_frame = pd.DataFrame(attempts).drop(columns=["_mtime_ns"], errors="ignore")
    provider_rows: list[dict[str, Any]] = []
    accounting_root = Path(config.get("budget_accounting_root", config.get("result_root", "results/raw")))
    budget_known = _historical_spend(accounting_root)
    budget_unknown, _ = _uncertain_billing_reserve(accounting_root)
    budget_limit = float(config.get("budget_limit_per_provider_usd", 5.0))
    budget_holdback = float(config.get("budget_holdback_usd", 0.25))
    for provider in sorted({str(spec["provider"]) for spec in specs}):
        provider_pairs = pairs[pairs.provider == provider]
        provider_outputs = [row for row in latest if row["provider"] == provider]
        status_counts = Counter(str(row.get("generation_status", "missing")) for row in provider_outputs)
        function_counts = Counter(str(row.get("functional_status", "missing")) for row in provider_outputs)
        cost = sum(float(row["estimated_cost_usd"]) for row in attempts if row["provider"] == provider and isinstance(row.get("estimated_cost_usd"), (int, float)))
        provider_rows.append({
            "provider": provider,
            "model_targets": sum(1 for spec in specs if spec["provider"] == provider),
            "planned_outputs": sum(1 for spec in specs if spec["provider"] == provider) * len(task_ids) * len(conditions),
            "latest_complete_outputs": status_counts["complete"],
            "latest_incomplete_outputs": status_counts["incomplete_response"],
            "latest_empty_outputs": status_counts["empty_response"],
            "latest_provider_errors": status_counts["provider_error"],
            "latest_transport_errors": status_counts["transport_or_parse_error"] + status_counts["transport_interrupted"],
            "latest_missing_outputs": status_counts["missing"],
            "complete_pairs": int(provider_pairs.pair_generation_status.eq("complete_pair").sum()),
            "mixed_pairs": int(provider_pairs.pair_generation_status.eq("mixed_generation_pair").sum()),
            "incomplete_pairs": int(provider_pairs.pair_generation_status.eq("incomplete_pair").sum()),
            "infrastructure_error_pairs": int(provider_pairs.pair_generation_status.eq("infrastructure_error_pair").sum()),
            "complete_grounding_core_pairs": int(provider_pairs.get("grounding_core_pair_generation_status", pd.Series(dtype=str)).eq("complete_pair").sum()),
            "mixed_grounding_core_pairs": int(provider_pairs.get("grounding_core_pair_generation_status", pd.Series(dtype=str)).eq("mixed_generation_pair").sum()),
            "incomplete_grounding_core_pairs": int(provider_pairs.get("grounding_core_pair_generation_status", pd.Series(dtype=str)).eq("incomplete_pair").sum()),
            "infrastructure_error_grounding_core_pairs": int(provider_pairs.get("grounding_core_pair_generation_status", pd.Series(dtype=str)).eq("infrastructure_error_pair").sum()),
            "functional_pass_outputs": function_counts["pass"],
            "functional_fail_outputs": function_counts["fail"],
            "functional_timeout_outputs": function_counts["timeout"],
            "functional_not_run_outputs": function_counts["not_run"] + function_counts["missing"],
            "api_conformance_pass_outputs": sum(1 for row in provider_outputs if row.get("api_conformance_status") == "pass"),
            "api_conformance_fail_outputs": sum(1 for row in provider_outputs if row.get("api_conformance_status") == "fail"),
            "api_conformance_not_run_outputs": sum(1 for row in provider_outputs if row.get("api_conformance_status") not in {"pass", "fail"}),
            "known_campaign_cost_usd_all_attempts": round(cost, 6),
            "known_spend_usd_budget_scope": round(budget_known.get(provider, 0.0), 6),
            "unknown_billing_reserve_usd_budget_scope": round(budget_unknown.get(provider, 0.0), 6),
            "remaining_budget_after_reserve_usd": round(
                budget_limit - budget_holdback - budget_known.get(provider, 0.0) - budget_unknown.get(provider, 0.0),
                6,
            ),
        })
    provider_summary = pd.DataFrame(provider_rows)
    return provider_summary, pairs, attempt_frame


def _write_markdown(frame: pd.DataFrame, path: Path, title: str, note: str) -> None:
    headers = list(frame.columns)
    lines = [f"# {title}", "", note, "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stacked_bar(
    table: pd.DataFrame, index: str, categories: list[str], output: Path,
    title: str, xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    left = [0] * len(table)
    for category in categories:
        values = table[category].astype(int).tolist()
        if not any(values):
            continue
        ax.barh(table[index], values, left=left, label=category.replace("_", " "),
                color=COLORS.get(category, "#888888"), edgecolor="white", linewidth=0.5)
        left = [a + b for a, b in zip(left, values)]
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    config: dict[str, Any], provider_summary: pd.DataFrame,
    pairs: pd.DataFrame, attempts: pd.DataFrame,
    table_root: Path, figure_root: Path,
) -> None:
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    provider_summary.to_csv(table_root / "table_model_coverage_provider.csv", index=False)
    pairs.to_csv(table_root / "table_model_coverage_pairs.csv", index=False)
    attempts.to_csv(table_root / "table_model_coverage_attempts.csv", index=False)
    task_count = len(config.get("task_ids", []))
    note = (
        f"Campaign `{config.get('campaign_id', 'unknown')}`; selected from the latest attempt "
        f"matching the frozen request configuration where available. Descriptive execution audit "
        f"for {task_count} configured task(s); not dissertation evidence or a statistical comparison."
    )
    cohort_note = config.get("cohort_note")
    if isinstance(cohort_note, str) and cohort_note.strip():
        note += f" Cohort: {cohort_note.strip()}"
    if (
        config.get("campaign_id") == "docground-api-pilot-v1"
        and Condition.SHIFTED_DOCGROUND.value in config.get("conditions", [])
    ):
        note += (
            " The `shifted_docground` condition used DocGround 0.2.0's `ground()` core with the approved, "
            "frozen, partial documentation snapshot. The experiment runner sent the selected prompt to "
            "models and evaluated outputs separately; this did not exercise DocGround's complete "
            "interactive workflow, provider adapters, or integrated verification/export path. Treat this "
            "as a grounding-core pilot, not an end-to-end DocGround result."
        )
    _write_markdown(provider_summary, table_root / "table_model_coverage_provider.md", "Model coverage by provider", note)
    _write_markdown(pairs, table_root / "table_model_coverage_pairs.md", "Paired model outputs", note)

    output_counts = provider_summary.set_index("provider")[[
        "latest_complete_outputs", "latest_incomplete_outputs", "latest_empty_outputs",
        "latest_provider_errors", "latest_transport_errors", "latest_missing_outputs",
    ]].copy()
    output_counts.columns = [
        "complete", "incomplete_response", "empty_response", "provider_error",
        "transport_or_parse_error", "missing",
    ]
    _stacked_bar(
        output_counts.reset_index(), "provider", list(output_counts.columns),
        figure_root / "figure_model_coverage_generation_status",
        f"Generation status by provider. {task_count} task execution audit", "Latest outputs",
    )

    pair_counts = provider_summary.set_index("provider")[[
        "complete_pairs", "mixed_pairs", "incomplete_pairs", "infrastructure_error_pairs",
    ]].copy()
    pair_counts.columns = ["complete_pair", "mixed_generation_pair", "incomplete_pair", "infrastructure_error_pair"]
    _stacked_bar(
        pair_counts.reset_index(), "provider", list(pair_counts.columns),
        figure_root / "figure_model_coverage_paired_status",
        f"Paired generation status by provider. {task_count} task execution audit", "Model-task pairs",
    )

    if "complete_grounding_core_pairs" in provider_summary and provider_summary["planned_outputs"].max() > 0:
        grounding_core_pair_counts = provider_summary.set_index("provider")[[
            "complete_grounding_core_pairs", "mixed_grounding_core_pairs",
            "incomplete_grounding_core_pairs", "infrastructure_error_grounding_core_pairs",
        ]].copy()
        grounding_core_pair_counts.columns = [
            "complete_pair", "mixed_generation_pair", "incomplete_pair", "infrastructure_error_pair",
        ]
        if grounding_core_pair_counts.to_numpy().sum() > 0:
            _stacked_bar(
                grounding_core_pair_counts.reset_index(), "provider", list(grounding_core_pair_counts.columns),
                figure_root / "figure_model_coverage_docground_paired_status",
                f"Shifted baseline versus DocGround grounding-core prompt arm. {task_count} task audit",
                "Model-task pairs",
            )

    functional = provider_summary.set_index("provider")[[
        "functional_pass_outputs", "functional_fail_outputs", "functional_timeout_outputs",
        "functional_not_run_outputs",
    ]].copy()
    functional.columns = ["pass", "fail", "timeout", "not_run"]
    _stacked_bar(
        functional.reset_index(), "provider", list(functional.columns),
        figure_root / "figure_model_coverage_functional_status",
        f"Functional evaluator outcomes. {task_count} configured task(s)", "Latest outputs",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_variant_coverage_v1_1.json"))
    parser.add_argument("--table-root", type=Path, default=Path("results/tables"))
    parser.add_argument("--figure-root", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    config, attempts, latest = load_campaign(args.config)
    provider_summary, pairs, attempt_frame = make_tables(config, attempts, latest)
    write_outputs(config, provider_summary, pairs, attempt_frame, args.table_root, args.figure_root)
    print(
        f"wrote coverage tables and figures for {len(config['models'])} models, "
        f"{len(config.get('task_ids', []))} task(s), and {len(attempts)} preserved attempt records"
    )


if __name__ == "__main__":
    main()
