"""Reconcile archived observations without requests or candidate execution.

Hashes describe bytes available now. Saved evaluation labels remain historical
claims; arithmetic agreement and static inspection do not authenticate a run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "results/end_to_end_training_live"
EXPECTED_RUN_FILES = (
    "evaluation.json", "interpretation.txt", "metrics.csv", "prompt_original.txt",
    "prompt_selected.txt", "prompt_shifted.txt", "response.txt", "run_manifest.json",
)
CONDITIONS = ("baseline", "shifted_baseline", "shifted_docground")
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_python(response: str) -> str:
    """Mirror the recorded evaluator's extraction; never execute its result."""
    blocks = _FENCE.findall(response)
    return blocks[0].strip() if blocks else response.strip()


def package_path(root: Path, relative: str) -> Path | None:
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def inventory_review(root: Path) -> dict[str, Any]:
    path = root / "source_inventory.json"
    if not path.exists():
        return {"status": "unavailable", "missing": [], "changed": [], "differences": []}
    differences = []
    entries = read_json(path)
    for entry in entries:
        file = package_path(root, entry["path"])
        actual = digest(file.read_bytes()) if file and file.is_file() else None
        if actual != entry["sha256"]:
            differences.append({
                "path": entry["path"], "status": "missing" if actual is None else "changed",
                "expected_sha256": entry["sha256"], "current_sha256": actual,
                "expected_bytes": entry.get("bytes"),
                "current_bytes": file.stat().st_size if actual is not None else None,
            })
    return {
        "status": "matches" if not differences else "differences_present",
        "inventory_sha256": digest(path.read_bytes()), "entries": len(entries),
        "missing": [item["path"] for item in differences if item["status"] == "missing"],
        "changed": [item["path"] for item in differences if item["status"] == "changed"],
        "differences": differences,
    }


def security_review(run: dict[str, Any], evaluation: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    findings = evaluation.get("details", {}).get("bandit_findings", [])
    if not findings:
        return []
    response_path = path / "response.txt"
    response = response_path.read_bytes() if response_path.exists() else None
    code = extract_python(response.decode("utf-8")) if response is not None else None
    try:
        tree = ast.parse(code) if code is not None else None
        assert_lines = sorted(node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)) if tree else None
    except SyntaxError:
        assert_lines = None
    results = []
    for finding in findings:
        rule = finding.get("test_id")
        matches = finding.get("line_number") in assert_lines if rule == "B101" and assert_lines is not None else None
        results.append({
            "run_id": run["run_id"], "rule": rule, "severity": finding.get("severity"),
            "recorded_security_status": evaluation.get("security_status"),
            "reported_line_number": finding.get("line_number"),
            "assert_nodes_in_saved_code": len(assert_lines) if assert_lines is not None else None,
            "assert_line_numbers_in_saved_code": assert_lines,
            "finding_matches_saved_code": matches,
            "review_status": "unsupported_source_attribution" if matches is False else "source_rule_matches" if matches else "not_verified",
            "response_file_sha256": digest(response) if response is not None else None,
            "extracted_code_sha256": digest(code.encode("utf-8")) if code is not None else None,
            "evaluation_file_sha256": digest((path / "evaluation.json").read_bytes()),
            "interpretation": "Static source attribution only; no new security pass or functional result is assigned.",
        })
    return results


def replay_review(root: Path, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "unavailable", "issues": ["audit_file_missing"]}
    recorded = read_json(path)
    records = recorded.get("grounded_records", [])
    static = sum(r.get("offline_replay", {}).get("status") == "replayed_and_statically_verified" for r in records)
    functional = sum(
        r.get("offline_replay", {}).get("functional_tests_executed") is True
        or r.get("functional_replay", {}).get("functional_tests_executed") is True
        for r in records
    )
    network_values = [r.get("offline_replay", {}).get("network_calls") for r in records]
    valid_network_counts = all(type(value) is int and value >= 0 for value in network_values)
    network_sum = sum(network_values) if valid_network_counts else None
    network_top = recorded.get("network_calls")
    issues = []
    uncertainties = []
    if network_sum is not None and type(network_top) is int and network_top != network_sum:
        issues.append("network_call_totals_disagree")
    if network_top is None or not valid_network_counts:
        uncertainties.append("historical_network_call_count_unverified")
    limitations = recorded.get("limitations", [])
    claims_functional = any("has been run" in text.lower() for text in limitations)
    if claims_functional and functional == 0:
        issues.append("functional_replay_claim_without_executed_records")
    counts = recorded.get("counts", {})
    for key, actual in (("grounded_records", len(records)), ("replayed_statically", static), ("replayed_functionally", functional)):
        if key in counts and counts[key] != actual:
            issues.append(f"{key}_count_disagrees")
    config = package_path(root, recorded.get("config_path", ""))
    config_hash = digest(config.read_bytes()) if config and config.is_file() else None
    supplied_config_hash = recorded.get("config_sha256")
    if supplied_config_hash is None:
        issues.append("config_hash_missing")
    elif supplied_config_hash != config_hash:
        issues.append("config_hash_differs_from_current_file")
    source_mismatches = []
    code_hash_basis = Counter()
    undeclared_response_file_hashes = 0
    for record in records:
        directory = package_path(root, record.get("artifact_dir", ""))
        if directory is None:
            source_mismatches.append({"run_id": record.get("source_run_id"), "path": record.get("artifact_dir"), "reason": "outside_package"})
            continue
        for name, expected in record.get("source_file_sha256", {}).items():
            file = package_path(directory, name)
            actual = digest(file.read_bytes()) if file and file.is_file() else None
            if actual != expected:
                source_mismatches.append({"run_id": record.get("source_run_id"), "path": name, "expected_sha256": expected, "current_sha256": actual})
        response_path = directory / "response.txt"
        if response_path.is_file():
            response = response_path.read_bytes()
            code_hash = digest(extract_python(response.decode("utf-8")).encode("utf-8"))
            saved_hash = record.get("offline_replay", {}).get("code_sha256")
            if saved_hash is not None:
                if saved_hash == code_hash:
                    code_hash_basis["extracted_code"] += 1
                elif saved_hash == digest(response):
                    code_hash_basis["response_file_bytes"] += 1
                    if record.get("offline_replay", {}).get("code_sha256_basis") != "saved_response_file_bytes_not_extracted_python":
                        undeclared_response_file_hashes += 1
                else:
                    code_hash_basis["unmatched"] += 1
    if source_mismatches:
        issues.append("source_file_hash_mismatch")
    if undeclared_response_file_hashes:
        issues.append("code_hash_describes_response_file_bytes")
    if code_hash_basis["unmatched"]:
        issues.append("code_hash_unmatched")
    return {
        "status": "issues_present" if issues else "internally_consistent_with_unresolved_history" if uncertainties else "internally_consistent",
        "audit_file_sha256": digest(path.read_bytes()), "issues": issues, "uncertainties": uncertainties,
        "recorded_counts": counts,
        "observed_counts": {"grounded_records": len(records), "replayed_statically": static, "functional_tests_executed_records": functional},
        "network_calls": {"recorded_top_level": network_top, "sum_of_recorded_per_record_counts": network_sum,
                          "historical_actual_calls_verified": False},
        "functional_completion_claim_in_limitations": claims_functional,
        "limitations": limitations,
        "config": {"path": recorded.get("config_path"), "recorded_sha256": supplied_config_hash,
                   "current_snapshot_sha256": config_hash, "matches_current_snapshot": supplied_config_hash == config_hash if config_hash else None,
                   "historical_config_verified": False,
                   "hash_basis": "Current file bytes only; this does not establish the configuration used historically."},
        "source_file_hash_mismatches": source_mismatches,
        "code_sha256_matches": dict(code_hash_basis),
        "interpretation": "Saved audit statements are inspected, not re-executed. Zero recorded functional replays cannot independently confirm functional labels.",
    }


def metadata_review(root: Path, metadata_root: Path) -> dict[str, Any]:
    validation_path = metadata_root / "validation.json"
    chart_path = metadata_root / "chart_registry.json"
    validation = read_json(validation_path) if validation_path.exists() else {}
    chart = read_json(chart_path) if chart_path.exists() else {}
    scopes = sorted({item.get("scope", "") for item in chart.get("registry", [])})
    dataset_type = validation.get("dataset_type")
    conflict = any("fixture" in scope.lower() for scope in scopes) and "live" in str(dataset_type).lower()
    return {
        "validation_dataset_type": dataset_type, "chart_scopes": scopes,
        "validation_reported_counts": {key: validation[key] for key in ("planned_slots", "total_run_directories", "complete_latest", "transport_errors", "incomplete_responses", "total_retries") if key in validation},
        "fixture_live_classification_conflict": conflict,
        "replay": replay_review(root, metadata_root / "docground_replay/docground_reuse_audit.json"),
    }


def audit(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    attempts = []
    security = []
    hashes: Counter[str] = Counter({"file_bytes_match": 0, "one_appended_lf_removed_match": 0, "mismatch": 0})
    prompt_hash_mismatches = []
    incomplete_directories = []
    manifest_errors = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    directories = sorted(path for path in (root / "raw").iterdir() if path.is_dir())
    for directory in directories:
        missing = [name for name in EXPECTED_RUN_FILES if not (directory / name).is_file()]
        if missing:
            incomplete_directories.append({"run_directory": directory.name, "missing_files": missing})
        if any(name in missing for name in ("run_manifest.json", "evaluation.json")):
            continue
        manifest = read_json(directory / "run_manifest.json")
        evaluation = read_json(directory / "evaluation.json")
        details = evaluation.get("details", {})
        row = {
            "run_id": manifest["run_id"], "provider": manifest.get("model_provider"),
            "model": manifest["model_id"], "task": manifest["task_id"],
            "condition": manifest["condition"], "replicate": manifest.get("replicate", 1),
            "attempt": manifest["attempt"], "functional": evaluation.get("functional_status"),
            "generation": details.get("generation_status"), "api": evaluation.get("api_conformance_status"),
            "security": evaluation.get("security_status"),
        }
        if row["run_id"] != directory.name:
            manifest_errors.append({"run_directory": directory.name, "recorded_run_id": row["run_id"]})
        attempts.append(row)
        key = (row["provider"], row["model"], row["task"], row["condition"], row["replicate"])
        groups[key].append(row)
        if key not in selected or row["attempt"] > selected[key]["attempt"]:
            selected[key] = row
        prompt_path = directory / "prompt_selected.txt"
        if prompt_path.exists():
            prompt = prompt_path.read_bytes()
            expected = manifest.get("prompt_sha256")
            raw_match = digest(prompt) == expected
            one_lf_match = prompt.endswith(b"\n") and digest(prompt[:-1]) == expected
            hashes["file_bytes_match"] += raw_match
            hashes["one_appended_lf_removed_match"] += one_lf_match
            if not one_lf_match:
                hashes["mismatch"] += 1
                prompt_hash_mismatches.append({"run_id": row["run_id"], "recorded_sha256": expected,
                                               "file_sha256": digest(prompt),
                                               "one_appended_lf_removed_sha256": digest(prompt[:-1]) if prompt.endswith(b"\n") else None})
        security.extend(security_review(row, evaluation, directory))
    retry_issues = []
    for rows in groups.values():
        rows.sort(key=lambda row: row["attempt"])
        for index, row in enumerate(rows):
            if index == 0 and row["attempt"] != 1:
                retry_issues.append({"run_id": row["run_id"], "reason": "first_attempt_missing"})
            if index:
                previous = rows[index - 1]
                if row["attempt"] != previous["attempt"] + 1 or previous["generation"] != "transport_or_parse_error":
                    retry_issues.append({"run_id": row["run_id"], "previous_run_id": previous["run_id"], "reason": "retry_not_after_contiguous_transport_error"})
    conditions = {}
    for condition in CONDITIONS:
        rows = [row for row in selected.values() if row["condition"] == condition]
        conditions[condition] = {"slots": len(rows), "generation": dict(Counter(row["generation"] for row in rows)),
                                 "functional": dict(Counter(row["functional"] for row in rows))}
    comparisons = []
    for before, after in (("baseline", "shifted_baseline"), ("shifted_baseline", "shifted_docground"), ("baseline", "shifted_docground")):
        pairs = []
        task_groups: dict[str, list[int]] = defaultdict(list)
        for (provider, model, task, condition, replicate), row in selected.items():
            if condition != before:
                continue
            other = selected.get((provider, model, task, after, replicate))
            if other and row["functional"] in ("pass", "fail") and other["functional"] in ("pass", "fail") and row["generation"] == other["generation"] == "complete":
                delta = int(other["functional"] == "pass") - int(row["functional"] == "pass")
                pairs.append((row["functional"], other["functional"]))
                task_groups[task].append(delta)
        means = {task: sum(values) / len(values) for task, values in task_groups.items()}
        comparisons.append({
            "before": before, "after": after, "pairs": len(pairs),
            "transitions": {str(key): value for key, value in Counter(pairs).items()},
            "pooled_delta_pp": 100 * sum(sum(values) for values in task_groups.values()) / len(pairs) if pairs else None,
            "task_mean_delta_pp": 100 * sum(means.values()) / len(means) if means else None,
            "task_deltas": means, "task_pair_counts": {task: len(values) for task, values in task_groups.items()},
            "weighting": {"pooled_delta_pp": "Each complete pair has equal weight.", "task_mean_delta_pp": "Each represented task has equal weight."},
        })
    inventory = inventory_review(root)
    metadata = metadata_review(root, root)
    historical_root = root / "review/original_metadata"
    historical = metadata_review(root, historical_root) if historical_root.exists() else None
    unavailable = [row for row in selected.values() if row["generation"] != "complete"]
    observed_summary = {
        "planned_slots": len(selected), "total_run_directories": len(directories),
        "complete_latest": len(selected) - len(unavailable),
        "transport_errors": sum(row["generation"] == "transport_or_parse_error" for row in attempts),
        "incomplete_responses": sum(row["generation"] == "incomplete_response" for row in attempts),
        "total_retries": len(attempts) - len(selected),
    }
    for item in (metadata, historical):
        if item is not None:
            item["validation_count_disagreements"] = [
                {"field": key, "reported": value, "observed": observed_summary[key]}
                for key, value in item["validation_reported_counts"].items() if value != observed_summary[key]
            ]
    return {
        "scope": "Arithmetic and static source review only; no provider requests or candidate execution; not independent authentication of historical execution.",
        "attempts": len(attempts), "selected_slots": len(selected), "conditions": conditions,
        "run_file_completeness": {"directories": len(directories), "expected_files_per_directory": len(EXPECTED_RUN_FILES),
                                  "complete_directories": len(directories) - len(incomplete_directories),
                                  "incomplete_directories": incomplete_directories, "run_id_mismatches": manifest_errors},
        "retry_accounting": {"additional_attempts": len(attempts) - len(selected),
                             "slots_with_multiple_attempts": sum(len(rows) > 1 for rows in groups.values()), "policy_issues": retry_issues},
        "selected_unavailable_outputs": {"count": len(unavailable), "generation_statuses": dict(Counter(row["generation"] for row in unavailable)),
                                         "records": unavailable, "interpretation": "Unavailable outputs remain in selected-slot denominators; these are distinct from missing archive files."},
        "comparisons": comparisons, "prompt_hashes": {**dict(hashes), "rule": "Compare bytes after removing exactly one final LF appended by write_run; preserve every other byte.", "mismatches": prompt_hash_mismatches},
        "security_review": security, "inventory_missing": inventory["missing"], "inventory_changed": inventory["changed"],
        "original_inventory_review": inventory, "metadata_review": metadata, "supplied_metadata_review": historical,
        "provenance_status": "Metadata and supplied labels do not independently authenticate live execution. Consult dataset_provenance.json and the preserved original metadata for repair history.",
        "generation_all_attempts": dict(Counter(row["generation"] for row in attempts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.root)
    output = arguments.output or arguments.root / "review/record_reconciliation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": result["attempts"], "selected_slots": result["selected_slots"], "output": str(output)}))


if __name__ == "__main__":
    main()
