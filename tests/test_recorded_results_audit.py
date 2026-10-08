"""Offline reconciliation tests: mutate evidence, never execute candidates."""
import json
from pathlib import Path

import pytest

from analysis.audit_recorded_dissertation_results import EXPECTED_RUN_FILES, audit, digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def add_run(root, *, name="model-baseline-a1", model="model", task="task", condition="baseline",
            attempt=1, generation="complete", functional="pass", prompt=b" task\n", findings=None,
            code="def answer():\n    return 1\n"):
    directory = root / "raw" / name
    directory.mkdir(parents=True)
    for filename in EXPECTED_RUN_FILES:
        (directory / filename).write_text("", encoding="utf-8")
    write_json(directory / "run_manifest.json", {
        "run_id": name, "model_provider": "provider", "model_id": model, "task_id": task,
        "condition": condition, "replicate": 1, "attempt": attempt, "prompt_sha256": digest(prompt),
    })
    write_json(directory / "evaluation.json", {
        "functional_status": functional, "security_status": "findings" if findings else "pass",
        "details": {"generation_status": generation, "bandit_findings": findings or []},
    })
    (directory / "prompt_selected.txt").write_bytes(prompt + b"\n")
    (directory / "response.txt").write_text(code + "\n", encoding="utf-8")
    return directory


def package(tmp_path):
    (tmp_path / "raw").mkdir()
    write_json(tmp_path / "source_inventory.json", [])
    return tmp_path


def test_prompt_hash_removes_one_lf_and_preserves_spaces_and_existing_newlines(tmp_path):
    root = package(tmp_path)
    directory = add_run(root, prompt=b" task \n")
    result = audit(root)
    assert result["prompt_hashes"]["one_appended_lf_removed_match"] == 1
    assert result["prompt_hashes"]["file_bytes_match"] == 0
    # A second appended newline is corruption; broad strip() would conceal it.
    path = directory / "prompt_selected.txt"
    path.write_bytes(path.read_bytes() + b"\n")
    result = audit(root)
    assert result["prompt_hashes"]["mismatch"] == 1
    assert result["prompt_hashes"]["one_appended_lf_removed_match"] == 0


def test_missing_run_files_are_separate_from_unavailable_model_outputs(tmp_path):
    root = package(tmp_path)
    add_run(root, name="a1", attempt=1, generation="transport_or_parse_error", functional="not_run")
    retry = add_run(root, name="a2", attempt=2, generation="incomplete_response", functional="not_run")
    result = audit(root)
    assert (result["attempts"], result["selected_slots"]) == (2, 1)
    assert result["retry_accounting"] == {"additional_attempts": 1, "slots_with_multiple_attempts": 1, "policy_issues": []}
    assert result["selected_unavailable_outputs"]["count"] == 1
    assert result["run_file_completeness"]["incomplete_directories"] == []
    (retry / "response.txt").unlink()
    result = audit(root)
    assert result["selected_unavailable_outputs"]["count"] == 1
    assert result["run_file_completeness"]["incomplete_directories"] == [{"run_directory": "a2", "missing_files": ["response.txt"]}]


def test_missing_manifest_is_detected_instead_of_dropping_directory_silently(tmp_path):
    root = package(tmp_path)
    directory = add_run(root)
    (directory / "run_manifest.json").unlink()
    result = audit(root)
    assert result["attempts"] == 0
    assert result["run_file_completeness"]["directories"] == 1
    assert result["run_file_completeness"]["complete_directories"] == 0


def test_retry_after_success_is_reported(tmp_path):
    root = package(tmp_path)
    add_run(root, name="a1")
    add_run(root, name="a2", attempt=2)
    assert audit(root)["retry_accounting"]["policy_issues"][0]["reason"] == "retry_not_after_contiguous_transport_error"


def test_b101_is_attributed_to_exact_candidate_line_without_executing_code(tmp_path):
    root = package(tmp_path)
    findings = [{"test_id": "B101", "severity": "LOW", "line_number": 4}]
    directory = add_run(root, findings=findings, code="raise RuntimeError('must never execute')\n")
    original_evaluation = (directory / "evaluation.json").read_bytes()
    item = audit(root)["security_review"][0]
    assert item["review_status"] == "unsupported_source_attribution"
    assert item["assert_nodes_in_saved_code"] == 0
    assert item["finding_matches_saved_code"] is False
    assert item["evaluation_file_sha256"] == digest(original_evaluation)
    assert (directory / "evaluation.json").read_bytes() == original_evaluation
    # An assert on the wrong line still does not corroborate the saved finding.
    (directory / "response.txt").write_text("assert True\n", encoding="utf-8")
    assert audit(root)["security_review"][0]["finding_matches_saved_code"] is False
    (directory / "response.txt").write_text("x=1\ny=2\nz=3\nassert True\n", encoding="utf-8")
    assert audit(root)["security_review"][0]["finding_matches_saved_code"] is True


def test_inventory_records_original_expected_hash_and_detects_mutation(tmp_path):
    root = package(tmp_path)
    add_run(root)
    path = root / "validation.json"
    write_json(path, {"dataset_type": "recorded"})
    expected = digest(path.read_bytes())
    write_json(root / "source_inventory.json", [{"path": "validation.json", "sha256": expected, "bytes": path.stat().st_size}])
    assert audit(root)["inventory_changed"] == []
    write_json(path, {"dataset_type": "changed"})
    result = audit(root)
    assert result["inventory_changed"] == ["validation.json"]
    item = result["original_inventory_review"]["differences"][0]
    assert item["expected_sha256"] == expected
    assert item["current_sha256"] == digest(path.read_bytes())
    path.unlink()
    assert audit(root)["inventory_missing"] == ["validation.json"]


def test_replay_reports_contradictions_and_current_config_without_historical_claim(tmp_path):
    root = package(tmp_path)
    directory = add_run(root)
    write_json(root / "config/campaign_snapshot.json", {"max_tokens": 512})
    replay = {
        "config_path": "config/campaign_snapshot.json", "config_sha256": None,
        "network_calls": 0, "counts": {"grounded_records": 1, "replayed_statically": 1, "replayed_functionally": 0},
        "limitations": ["Functional replay has been run."],
        "grounded_records": [{"source_run_id": directory.name, "artifact_dir": str(directory.relative_to(root)),
                              "source_file_sha256": {"response.txt": digest((directory / "response.txt").read_bytes())},
                              "offline_replay": {"status": "replayed_and_statically_verified", "network_calls": 1,
                                                 "functional_tests_executed": False,
                                                 "code_sha256": digest((directory / "response.txt").read_bytes())}}],
    }
    replay_path = root / "docground_replay/docground_reuse_audit.json"
    write_json(replay_path, replay)
    result = audit(root)["metadata_review"]["replay"]
    assert set(result["issues"]) == {"network_call_totals_disagree", "functional_replay_claim_without_executed_records", "config_hash_missing", "code_hash_describes_response_file_bytes"}
    assert result["network_calls"]["sum_of_recorded_per_record_counts"] == 1
    assert result["config"]["historical_config_verified"] is False
    assert result["config"]["current_snapshot_sha256"] == digest((root / replay["config_path"]).read_bytes())
    assert result["observed_counts"]["functional_tests_executed_records"] == 0
    # A subsequently supplied current hash is useful but does not prove history.
    replay["config_sha256"] = result["config"]["current_snapshot_sha256"]
    replay["network_calls"] = None
    replay["limitations"] = ["Functional replay has not been run."]
    write_json(replay_path, replay)
    result = audit(root)["metadata_review"]["replay"]
    assert "network_call_totals_disagree" not in result["issues"]
    assert "historical_network_call_count_unverified" in result["uncertainties"]
    assert result["config"]["matches_current_snapshot"] is True
    assert result["config"]["historical_config_verified"] is False
    (directory / "response.txt").write_text("changed\n", encoding="utf-8")
    assert "source_file_hash_mismatch" in audit(root)["metadata_review"]["replay"]["issues"]


def test_historical_metadata_is_reviewed_separately(tmp_path):
    root = package(tmp_path)
    add_run(root)
    write_json(root / "validation.json", {"dataset_type": "recorded_observations"})
    write_json(root / "chart_registry.json", {"registry": [{"scope": "Supplied observations"}]})
    saved = root / "review/original_metadata"
    write_json(saved / "validation.json", {"dataset_type": "end_to_end_live_training"})
    write_json(saved / "chart_registry.json", {"registry": [{"scope": "Fixture run records"}]})
    result = audit(root)
    assert result["metadata_review"]["fixture_live_classification_conflict"] is False
    assert result["supplied_metadata_review"]["fixture_live_classification_conflict"] is True


def test_pair_and_task_weighted_estimators_keep_different_denominators(tmp_path):
    root = package(tmp_path)
    for task, models in (("task1", ["m1", "m2"]), ("task2", ["m1"])):
        for model in models:
            for condition in ("shifted_baseline", "shifted_docground"):
                functional = "pass" if condition == "shifted_docground" and task == "task1" else "fail"
                add_run(root, name=f"{task}-{model}-{condition}", task=task, model=model, condition=condition, functional=functional)
    result = audit(root)["comparisons"][1]
    assert result["pairs"] == 3
    assert result["pooled_delta_pp"] == pytest.approx(200 / 3)
    assert result["task_mean_delta_pp"] == 50


def test_validation_summary_changes_are_detected(tmp_path):
    root = package(tmp_path)
    add_run(root)
    write_json(root / "validation.json", {"planned_slots": 1, "complete_latest": 2})
    result = audit(root)["metadata_review"]
    assert result["validation_count_disagreements"] == [{"field": "complete_latest", "reported": 2, "observed": 1}]
