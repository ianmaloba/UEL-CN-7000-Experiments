"""Audit saved grounding observations and replay their workflow without API calls.

Run from the experiment repository root. Raw run records are read only. The
workflow receives the experiment's original extraction of each saved response;
current DocGround adapter extraction is compared separately, never substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.aggregate_model_coverage import load_campaign
from docground.adapters.base import extract_code
from docground.grounding.doc_store import DocStore
from docground.grounding.wrapper import ground
from docground.verification.gate import evaluate
from docground.verification.correctness import DockerCorrectnessRunner
from docground.workflow import PromptDecision, ReviewSession
from experiments.evaluate import extract_python
from experiments.schema import Condition, Task, TaskSet


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_text(path: Path) -> str:
    """Remove only the one newline appended by write_run, not model whitespace."""
    return path.read_text(encoding="utf-8").removesuffix("\n")


def replay_environment_issue(status: str, stderr: str) -> str | None:
    """Treat the pinned runner's cleanup warning as inconclusive infrastructure."""
    if status == "unavailable":
        return "docker_unavailable"
    if "Could not confirm cleanup of Docker container " in stderr:
        return "container_cleanup_unconfirmed"
    return None


@dataclass
class RecordedResponseAdapter:
    """A prompt-checked replay, with no transport or access to credentials."""

    provider: str
    model: str
    expected_prompt: str
    saved_response: str
    calls: int = 0

    def generate(self, prompt: str) -> str:
        if prompt != self.expected_prompt:
            raise ValueError("Replay prompt differs from the saved selected prompt")
        self.calls += 1
        return extract_python(self.saved_response)


def matching_latest(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Never use the coverage report's fallback to a nonmatching old attempt."""
    return (
        [row for row in rows if row.get("request_matches_current_config") is True],
        [row for row in rows if row.get("request_matches_current_config") is not True],
    )


def duplicate_responses(attempts: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Count repeated valid provider outcomes separately from transport retries."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        status = row.get("generation_status")
        http = row.get("http_status")
        if (
            row.get("request_matches_current_config") is True
            and status in {"complete", "incomplete_response", "empty_response"}
            and isinstance(http, int) and 200 <= http < 300
        ):
            groups[tuple(str(row[key]) for key in ("provider", "model_id", "task_id", "condition"))].append(row)
    return [{
        **dict(zip(("provider", "model_id", "task_id", "condition"), key)),
        "valid_response_count": len(rows),
        "responses": [{field: row.get(field) for field in ("run_id", "attempt", "generation_status", "selected_for_current_summary")} for row in sorted(rows, key=lambda row: (row["attempt"], row.get("_mtime_ns", 0)))],
    } for key, rows in sorted(groups.items()) if len(rows) > 1]


def prepare_audit_prompts(
    *, review: dict[str, Any], review_sha256: str, taskset: TaskSet,
    taskset_sha256: str, tasks: list[Task], conditions: list[Condition], store: DocStore,
) -> dict[tuple[str, Condition], dict[str, object]]:
    """Reconstruct historical prompts with today's code without changing versions."""
    if review.get("status") != "approved":
        raise ValueError("Prompt review is not approved")
    if review.get("taskset_id") != taskset.taskset_id or review.get("taskset_sha256") != taskset_sha256:
        raise ValueError("Frozen taskset differs from approved review")
    if review.get("documentation_snapshot_sha256") != store.snapshot_sha256:
        raise ValueError("Current snapshot differs from the historical snapshot")
    reviewed = {item["task_id"]: item for item in review["tasks"]}
    prompts = {}
    for task in tasks:
        record = reviewed[task.task_id]
        proposal = ground(task.shifted_prompt, store=store)
        expected_hashes = {
            "original": sha256(task.original_prompt),
            "shifted": sha256(task.shifted_prompt),
            "suggested": sha256(proposal.text),
        }
        if (
            record.get("approval") != "approve_suggestion"
            or record.get("prompt_hashes") != expected_hashes
            or record.get("suggested_prompt") != proposal.text
            or record.get("documentation_snapshot_sha256") != store.snapshot_sha256
        ):
            raise ValueError(f"Current grounding differs from review for {task.task_id}")
        for condition in conditions:
            grounded = condition is Condition.SHIFTED_DOCGROUND
            prompts[(task.task_id, condition)] = {
                "prompt": proposal.text if grounded else task.prompt_for(condition),
                "documentation_snapshot_sha256": store.snapshot_sha256 if grounded else None,
                "docground_version": review["docground_version"] if grounded else None,
                "prompt_decision": "approve_suggestion" if grounded else None,
                "docground_review_sha256": review_sha256 if grounded else None,
            }
    return prompts


def check_manifest(
    manifest: dict[str, Any], task: Task, taskset: TaskSet, selected: str,
    expected_prompt: dict[str, object],
) -> None:
    expected = {
        "taskset_id": taskset.taskset_id,
        "task_id": task.task_id,
        "original_prompt_sha256": sha256(task.original_prompt),
        "shifted_prompt_sha256": sha256(task.shifted_prompt),
        "prompt_sha256": sha256(selected),
        "documentation_snapshot_sha256": expected_prompt["documentation_snapshot_sha256"],
        "docground_version": expected_prompt["docground_version"],
        "prompt_decision": expected_prompt["prompt_decision"],
        "docground_review_sha256": expected_prompt["docground_review_sha256"],
    }
    if selected != expected_prompt["prompt"]:
        raise ValueError(f"Selected prompt changed for {manifest.get('run_id')}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key} mismatch for {manifest.get('run_id')}")


def replay_record(
    *, original_task: str, selected_prompt: str, saved_response: str,
    provider: str, model: str, store: DocStore,
    test_code: str | None = None, evaluator_image: str | None = None,
) -> dict[str, object]:
    """Replay saved code; optional functional execution is Docker-only."""
    session = ReviewSession.prepare(original_task, provider, model, store=store)
    session.decide(PromptDecision.APPROVE_SUGGESTION)
    if session.selected_prompt != selected_prompt:
        raise ValueError("Reconstructed workflow proposal differs from saved prompt")
    if not saved_response.strip():
        return {"status": "no_saved_response", "network_calls": 0}
    adapter = RecordedResponseAdapter(provider, model, selected_prompt, saved_response)
    code = session.generate(adapter)
    if test_code is not None and not evaluator_image:
        raise ValueError("Functional replay requires the original evaluator image")
    runner = DockerCorrectnessRunner(image=evaluator_image) if test_code is not None else None
    # None means unknown. Do not infer that an unknown import exists or is fake.
    report = evaluate(
        code, test=test_code, store=store, package_resolver=lambda package: None,
        correctness_runner=runner, timeout=15.0,
    )
    session.record_verification(report.to_dict())
    result = {
        "status": "replayed_and_verified" if test_code is not None else "replayed_and_statically_verified",
        "workflow_status": session.status.value,
        "recorded_adapter_calls": adapter.calls,
        "network_calls": 0,
        "code_sha256": sha256(code),
        "matches_current_adapter_extraction": code == extract_code(saved_response),
        "functional_tests_executed": bool(report.correctness and report.correctness.status in {"pass", "fail", "timeout"}),
        "export_attempted": False,
    }
    if test_code is None:
        result["static_verification"] = report.to_dict()
    else:
        result.update({
            "verification": report.to_dict(),
            "test_code_sha256": sha256(test_code),
            "evaluator_image": evaluator_image,
            "timeout_seconds": 15.0,
            "functional_runner_invoked": report.correctness is not None,
            "functional_status": report.correctness.status if report.correctness else "not_run",
            "functional_environment_issue": replay_environment_issue(
                report.correctness.status, report.correctness.stderr,
            ) if report.correctness else None,
        })
    return result


def validate_functional_fixtures(tasks: list[Task], store: DocStore, image: str) -> list[dict[str, object]]:
    """Validate the unchanged task tests against references before saved outputs."""
    fixtures = []
    for task in tasks:
        if not task.reference_solution or not task.test_code.strip():
            raise ValueError(f"No reference solution/test for {task.task_id}")
        report = evaluate(
            task.reference_solution, test=task.test_code, store=store,
            package_resolver=lambda package: None,
            correctness_runner=DockerCorrectnessRunner(image=image), timeout=15.0,
        )
        fixtures.append({
            "task_id": task.task_id, "test_code_sha256": sha256(task.test_code),
            "reference_solution_sha256": sha256(task.reference_solution),
            "evaluator_image": image,
            "functional_status": report.correctness.status if report.correctness else "not_run",
            "verification": report.to_dict(),
        })
    return fixtures


def inspect_local_image(image: str) -> dict[str, object]:
    """Check daemon/image access without pulling or executing any container."""
    try:
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Id}}", image],
            text=True, capture_output=True, timeout=15.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "tag": image, "error": type(error).__name__}
    if inspected.returncode != 0:
        return {"available": False, "tag": image, "returncode": inspected.returncode, "error": inspected.stderr.strip()}
    return {"available": True, "tag": image, "image_id": json.loads(inspected.stdout)}


def add_functional_replay(
    result: dict[str, Any], *, tasks: list[Task], store: DocStore,
    evaluator_image: str, workers: int,
) -> None:
    if not 1 <= workers <= 4:
        raise ValueError("Functional concurrency must be between 1 and 4")
    result["functional_mode_requested"] = True
    result["functional_comparison_status"] = "incomplete"
    result["functional_workers"] = workers
    image = inspect_local_image(evaluator_image)
    result["functional_evaluator_image"] = image
    if not image["available"]:
        result["audit_status"] = "incomplete"
        result["errors"].append({"stage": "docker_preflight", "error": "Docker daemon or original local image is unavailable; no fixtures or saved responses were executed."})
        return
    fixtures = validate_functional_fixtures(tasks, store, evaluator_image)
    result["functional_fixture_checks"] = fixtures
    result["functional_tests_executed"] = any(item["functional_status"] in {"pass", "fail", "timeout"} for item in fixtures)
    if any(item["functional_status"] != "pass" for item in fixtures):
        result["audit_status"] = "incomplete"
        result["errors"].append({"stage": "functional_fixtures", "error": "A reference solution did not pass; saved-response execution was skipped."})
        return
    task_map = {task.task_id: task for task in tasks}
    nonempty = [record for record in result["grounded_records"] if record["offline_replay"]["status"] == "replayed_and_statically_verified"]

    def run_record(record: dict[str, Any]) -> dict[str, object]:
        path = Path(record["artifact_dir"])
        if any(file_hash(path / name) != expected for name, expected in record["source_file_sha256"].items()):
            raise ValueError("Source record changed after the integrity audit")
        image = record["original_evaluation"]["evaluator_image"]
        if image != evaluator_image:
            raise ValueError("Original evaluator image differs from frozen configuration")
        task = task_map[record["task_id"]]
        replay = replay_record(
            original_task=task.shifted_prompt,
            selected_prompt=artifact_text(path / "prompt_selected.txt"),
            saved_response=artifact_text(path / "response.txt"),
            provider=record["provider"], model=record["model_id"], store=store,
            test_code=task.test_code, evaluator_image=image,
        )
        replay["matches_original_functional_status"] = replay["functional_status"] == record["original_evaluation"]["functional_status"]
        return replay

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(run_record, record): record for record in nonempty}
        halted = False
        completed = 0
        for future in as_completed(pending):
            if future.cancelled():
                continue
            completed += 1
            record = pending[future]
            try:
                record["functional_replay"] = future.result()
                if record["functional_replay"]["functional_environment_issue"] is not None:
                    halted = True
            except Exception as error:
                result["errors"].append({"run_id": record["source_run_id"], "stage": "functional_replay", "error": f"{type(error).__name__}: {error}"})
                halted = True
            if halted:
                for waiting in pending:
                    waiting.cancel()
            if completed % 30 == 0 or completed == len(nonempty):
                print(f"[offline functional replay] {completed}/{len(nonempty)}", flush=True)
    checked = [record for record in nonempty if "functional_replay" in record]
    infrastructure_errors = [record for record in checked if record["functional_replay"]["functional_environment_issue"] is not None]
    disagreements = [record for record in checked if not record["functional_replay"]["matches_original_functional_status"]]
    result["counts"].update({
        "functional_replay_records": len(checked),
        "functional_runner_invocations": sum(record["functional_replay"]["functional_runner_invoked"] for record in checked),
        "new_functional_status": dict(Counter(record["functional_replay"]["functional_status"] for record in checked)),
        "new_full_verdicts": dict(Counter(record["functional_replay"]["verification"]["verdict"] for record in checked)),
        "functional_disagreements": len(disagreements),
        "functional_infrastructure_errors": len(infrastructure_errors),
        "functional_cleanup_uncertainties": sum(
            record["functional_replay"]["functional_environment_issue"] == "container_cleanup_unconfirmed"
            for record in checked
        ),
        "functional_records_not_processed": len(nonempty) - len(checked),
    })
    result["functional_disagreement_run_ids"] = [record["source_run_id"] for record in disagreements]
    if result["errors"] or infrastructure_errors or len(checked) != len(nonempty):
        result["audit_status"] = "incomplete"
    result["functional_comparison_status"] = (
        "incomplete" if result["audit_status"] != "pass"
        else "requires_review" if disagreements or any(
            record["functional_replay"]["functional_status"] == "timeout" for record in checked
        ) else "consistent_on_observed_statuses"
    )
    result["limitations"][3] = "Original functional outcomes are preserved. New DocGround Docker evaluations rerun the frozen tests on recorded code, with the original image tag and a 15-second timeout. They are separate reanalyses, not new model generations; discrepancies are retained. Export is not attempted."
    result["limitations"].append("The released correctness runner executes candidate.py with runpy and the frozen tests then import candidate; this can execute module-level code twice. The original experiment evaluator imports candidate once. Reference checks establish fixture compatibility, not evaluator identity.")


def audit(config_path: Path, *, functional: bool = False, workers: int = 4) -> dict[str, object]:
    if functional and importlib.metadata.version("docground") != "0.2.1":
        raise ValueError("Functional replay requires the tested DocGround 0.2.1 release")
    config, attempts, candidates = load_campaign(config_path)
    if int(config.get("max_tokens", 0)) != 512:
        raise ValueError("This audit targets the frozen 512-token campaign")
    review_path = Path(config["docground_review_path"])
    if file_hash(review_path) != config.get("docground_review_sha256"):
        raise ValueError("Review file differs from the frozen configuration")
    taskset_path = Path(config["taskset_path"])
    taskset = TaskSet.from_path(taskset_path)
    task_by_id = {task.task_id: task for task in taskset.tasks}
    tasks = [task_by_id[task_id] for task_id in config["task_ids"]]
    conditions = [Condition(condition) for condition in config["conditions"]]
    store = DocStore.load()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("docground_version") != config.get("docground_version"):
        raise ValueError("Historical DocGround version differs between config and review")
    prompts = prepare_audit_prompts(
        review=review, review_sha256=file_hash(review_path), taskset=taskset,
        taskset_sha256=file_hash(taskset_path), tasks=tasks, conditions=conditions, store=store,
    )
    duplicate_groups = duplicate_responses(attempts)
    selected, excluded = matching_latest(candidates)
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    response_extraction_mismatches: list[str] = []
    all_selected_run_ids: list[str] = []
    for row in selected:
        path = Path(row["artifact_dir"])
        manifest_path = path / "run_manifest.json"
        response_path = path / "response.txt"
        evaluation_path = path / "evaluation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        selected_prompt = artifact_text(path / "prompt_selected.txt")
        saved_response = artifact_text(response_path)
        task = task_by_id[row["task_id"]]
        condition = Condition(row["condition"])
        try:
            check_manifest(manifest, task, taskset, selected_prompt, prompts[(task.task_id, condition)])
            if artifact_text(path / "prompt_original.txt") != task.original_prompt:
                raise ValueError("Saved original prompt differs from frozen task")
            if artifact_text(path / "prompt_shifted.txt") != task.shifted_prompt:
                raise ValueError("Saved shifted prompt differs from frozen task")
        except ValueError as error:
            errors.append({"run_id": str(row["run_id"]), "error": str(error)})
            continue
        all_selected_run_ids.append(str(row["run_id"]))
        if saved_response.strip() and extract_python(saved_response) != extract_code(saved_response):
            response_extraction_mismatches.append(str(row["run_id"]))
        if condition is not Condition.SHIFTED_DOCGROUND:
            continue
        try:
            replay = replay_record(
                original_task=task.shifted_prompt, selected_prompt=selected_prompt,
                saved_response=saved_response, provider=row["provider"],
                model=row["model_id"], store=store,
            )
        except (ValueError, RuntimeError) as error:
            errors.append({"run_id": str(row["run_id"]), "error": str(error)})
            continue
        records.append({
            "source_run_id": row["run_id"],
            "artifact_dir": str(path),
            "provider": row["provider"], "model_id": row["model_id"],
            "task_id": task.task_id, "attempt": row["attempt"],
            "generation_status": row["generation_status"],
            "source_file_sha256": {
                "run_manifest.json": file_hash(manifest_path),
                "response.txt": file_hash(response_path),
                "evaluation.json": file_hash(evaluation_path),
            },
            "selected_prompt_sha256": sha256(selected_prompt),
            "original_evaluation": {
                "source": str(evaluation_path),
                "functional_status": original_evaluation["functional_status"],
                "syntax_status": original_evaluation["syntax_status"],
                "api_conformance_status": original_evaluation.get("api_conformance_status"),
                "evaluator_image": manifest["evaluator_image"],
                "newly_executed": False,
            },
            "offline_replay": replay,
        })
    replayed = [record for record in records if record["offline_replay"]["status"] == "replayed_and_statically_verified"]
    result = {
        "schema_version": 1,
        "audit_script_sha256": file_hash(Path(__file__)),
        "created_at": datetime.now(UTC).isoformat(),
        "audit_status": "pass" if not errors and not excluded else "incomplete",
        "audit_status_scope": "Artifact integrity and replay completion only; not dissertation acceptance or live workflow equivalence.",
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path), "config_sha256": file_hash(config_path),
        "taskset_sha256": file_hash(taskset_path),
        "review_sha256": file_hash(review_path),
        "historical_docground_version": review["docground_version"],
        "audit_runtime_docground_version": importlib.metadata.version("docground"),
        "documentation_snapshot_sha256": store.snapshot_sha256,
        "network_calls": 0, "functional_tests_executed": False,
        "functional_mode_requested": False,
        "raw_records_modified": False,
        "counts": {
            "preserved_attempts": len(attempts),
            "planned_slots": len(candidates),
            "matching_latest_slots": len(selected),
            "verified_prompt_manifest_slots": len(all_selected_run_ids),
            "grounded_slots": len(records),
            "grounded_nonempty_responses_replayed": len(replayed),
            "grounded_slots_without_response": len(records) - len(replayed),
            "all_conditions_extraction_differences": len(response_extraction_mismatches),
            "matching_slots_with_multiple_valid_responses": len(duplicate_groups),
            "additional_matching_valid_responses": sum(group["valid_response_count"] - 1 for group in duplicate_groups),
            "grounded_extraction_differences": sum(not record["offline_replay"]["matches_current_adapter_extraction"] for record in replayed),
            "grounded_generation_status": dict(Counter(record["generation_status"] for record in records)),
            "new_static_verdicts": dict(Counter(record["offline_replay"]["static_verification"]["verdict"] for record in replayed)),
        },
        "interpretation": "Saved observations remain an audit-compatible grounding-core pilot. Replay verifies workflow compatibility using recorded responses; it is not a new model observation or proof of live adapter equivalence.",
        "limitations": [
            "The original run used the experiment provider transport, not DocGround's live provider adapters or interactive CLI.",
            "Experiment requests appended a fenced-code instruction and used 512-token caps, top_p, and model-dependent routes. Current DocGround adapter defaults and payloads differ.",
            "The replay returns the experiment's original code extraction. Incomplete open-fence responses can extract differently in the current DocGround adapter; these differences are listed explicitly.",
            "New verification is static only. Original isolated functional outcomes are references, not newly executed tests. Export is not attempted by this audit.",
            "The offline package resolver returns unknown for packages outside the snapshot; no live package-index checks or model calls occur.",
            "Saved manifests pin package version, prompt, review and snapshot hashes, but do not record a DocGround source commit or immutable evaluator image digest.",
            "Historical package versions are retained. A different audit runtime only establishes that the frozen prompts and snapshot can still be reconstructed; it does not retroactively change which release generated the observations.",
            "The coverage report selects the latest configuration-matching attempt. Repeated valid model responses from earlier recovery are listed; this audit does not resolve the resulting selection/resampling limitation or promote the pilot to an accepted dissertation dataset.",
            "Six curated API tasks, one replicate, token truncation, aliases, and unavailable provider outputs limit generalisation. Replay adds no independent observations.",
        ],
        "excluded_nonmatching_slots": [{key: row.get(key) for key in ("run_id", "provider", "model_id", "task_id", "condition")} for row in excluded],
        "errors": errors,
        "verified_source_run_ids": all_selected_run_ids,
        "extraction_difference_run_ids": response_extraction_mismatches,
        "repeated_valid_response_groups": duplicate_groups,
        "grounded_records": records,
    }
    if functional:
        add_functional_replay(result, tasks=tasks, store=store, evaluator_image=config["evaluator_image"], workers=workers)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/docground_api_pilot_v1_recovery_512.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--functional", action="store_true", help="Run frozen tests on saved outputs in the original Docker image, without API calls")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    output = args.output or Path("results/tables/docground_api_pilot_v1") / ("docground_reuse_audit_0_2_1.json" if args.functional else "docground_reuse_audit.json")
    if args.functional and output.resolve() == Path("results/tables/docground_api_pilot_v1/docground_reuse_audit.json").resolve():
        parser.error("Keep the original static audit; choose a separate functional output path")
    if output.exists():
        parser.error("Audit output already exists; choose a new path to preserve the previous result")
    result = audit(args.config, functional=args.functional, workers=args.workers)
    result["completed_at"] = datetime.now(UTC).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"audit_status": result["audit_status"], "counts": result["counts"], "output": str(output)}, sort_keys=True))
    if result["audit_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
