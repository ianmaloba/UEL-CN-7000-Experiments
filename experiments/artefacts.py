"""Safe, append-only-style local artefact writing for a single model response."""

from __future__ import annotations

import os
import csv
from pathlib import Path

from .evaluate import Evaluation
from .schema import RunManifest, Task


def redact_secrets(text: str) -> str:
    """Replace configured secret values without persisting them in evidence files."""
    redacted = text
    for key, value in os.environ.items():
        if key.endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(value) >= 8:
            redacted = redacted.replace(value, f"[REDACTED:{key}]")
    return redacted


def write_metrics(destination: Path, manifest: RunManifest, evaluation: Evaluation) -> None:
    """Write the current evaluation state; used after isolated reevaluation too."""
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "task_id", "condition", "attempt", "generation_status",
            "provider_http_status", "request_id", "input_tokens", "output_tokens",
            "estimated_cost_usd", "syntax_status", "functional_status",
            "security_status", "hallucination_status", "api_conformance_status",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": manifest.run_id, "task_id": manifest.task_id,
            "condition": manifest.condition.value, "attempt": manifest.attempt,
            "generation_status": evaluation.details.get("generation_status", "unknown"),
            "provider_http_status": evaluation.details.get("http_status", evaluation.details.get("provider_http_status", "")),
            "request_id": evaluation.details.get("request_id", ""),
            "input_tokens": evaluation.details.get("input_tokens", ""),
            "output_tokens": evaluation.details.get("output_tokens", ""),
            "estimated_cost_usd": evaluation.details.get("estimated_cost_usd", ""),
            "syntax_status": evaluation.syntax_status,
            "functional_status": evaluation.functional_status,
            "security_status": evaluation.security_status,
            "hallucination_status": evaluation.hallucination_status,
            "api_conformance_status": evaluation.api_conformance_status,
        })


def write_run(
    root: Path,
    manifest: RunManifest,
    task: Task,
    response: str,
    evaluation: Evaluation,
    selected_prompt: str | None = None,
) -> Path:
    """Write once; refuse overwrite so failed/earlier outputs remain inspectable."""
    destination = root / manifest.run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "run_manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (destination / "prompt_original.txt").write_text(task.original_prompt + "\n", encoding="utf-8")
    (destination / "prompt_shifted.txt").write_text(task.shifted_prompt + "\n", encoding="utf-8")
    selected = selected_prompt if selected_prompt is not None else task.prompt_for(manifest.condition)
    (destination / "prompt_selected.txt").write_text(selected + "\n", encoding="utf-8")
    (destination / "response.txt").write_text(redact_secrets(response) + "\n", encoding="utf-8")
    (destination / "evaluation.json").write_text(evaluation.to_json(), encoding="utf-8")
    write_metrics(destination, manifest, evaluation)
    (destination / "interpretation.txt").write_text(
        "STATUS: pilot\n"
        "CLAIM: No dissertation finding; this artefact records one model output in a controlled pilot.\n"
        "LIMITATIONS: One task and replicate do not support generalisation; interpret only with its paired run, provider metadata, and the frozen campaign protocol.\n",
        encoding="utf-8",
    )
    return destination
