import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.audit_docground_reuse import (
    RecordedResponseAdapter, artifact_text, duplicate_responses, file_hash,
    matching_latest, prepare_audit_prompts, replay_record,
    add_functional_replay, replay_environment_issue,
)
from docground.grounding.doc_store import DocStore
from docground.grounding.wrapper import ground
from experiments.schema import Condition, TaskSet


def test_recorded_adapter_rejects_prompt_drift_and_preserves_unclosed_fence():
    raw = "```python\nvalue = 1"
    adapter = RecordedResponseAdapter("provider", "model", "reviewed", raw)
    with pytest.raises(ValueError, match="differs"):
        adapter.generate("changed")
    assert adapter.calls == 0
    assert adapter.generate("reviewed") == raw


def test_matching_selection_never_falls_back_to_wrong_configuration():
    good = {"run_id": "a", "request_matches_current_config": True}
    stale = {"run_id": "b", "request_matches_current_config": False}
    missing = {"run_id": "c"}
    assert matching_latest([good, stale, missing]) == ([good], [stale, missing])


def test_resampling_audit_counts_valid_responses_but_excludes_transports_and_wrong_configs():
    base = {"provider": "p", "model_id": "m", "task_id": "t", "condition": "c", "request_matches_current_config": True, "http_status": 200}
    rows = [
        {**base, "run_id": "first", "attempt": 1, "generation_status": "incomplete_response"},
        {**base, "run_id": "second", "attempt": 2, "generation_status": "complete"},
        {**base, "run_id": "transport", "attempt": 3, "generation_status": "transport_or_parse_error", "http_status": None},
        {**base, "run_id": "old_config", "attempt": 4, "generation_status": "complete", "request_matches_current_config": False},
    ]
    groups = duplicate_responses(rows)
    assert len(groups) == 1
    assert groups[0]["valid_response_count"] == 2
    assert [row["run_id"] for row in groups[0]["responses"]] == ["first", "second"]


def test_artifact_reader_removes_only_writer_newline(tmp_path):
    path = tmp_path / "response.txt"
    path.write_text("content\n\n", encoding="utf-8")
    assert artifact_text(path) == "content\n"


@pytest.mark.parametrize("response, verdict", [("import unknown_package_for_replay_audit\ndef f(): return 1", "warn"), ("def broken(", "block")])
def test_replay_uses_no_network_or_functional_execution(monkeypatch, response, verdict):
    import requests
    import docground.verification.gate as gate

    def forbidden(*args, **kwargs):
        raise AssertionError("Network or functional execution attempted")

    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(gate, "check_correctness", lambda code, test, timeout, runner: None if test is None else forbidden())
    store = DocStore.load()
    prompt = ground("write f", store=store).text
    result = replay_record(
        original_task="write f", selected_prompt=prompt, saved_response=response,
        provider="saved-provider", model="saved-model", store=store,
    )
    assert result["static_verification"]["verdict"] == verdict
    assert result["functional_tests_executed"] is False
    assert result["export_attempted"] is False
    assert result["recorded_adapter_calls"] == 1


def test_reconstruction_preserves_historical_version_and_rejects_prompt_tampering():
    taskset_path = Path("configs/docground_api_pilot_v1_candidate.json")
    review_path = Path("results/protocol_review/docground_api_pilot_v1_candidate_v3.json")
    taskset = TaskSet.from_path(taskset_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    kwargs = {
        "review_sha256": file_hash(review_path), "taskset": taskset,
        "taskset_sha256": file_hash(taskset_path), "tasks": taskset.tasks,
        "conditions": [Condition.SHIFTED_DOCGROUND], "store": DocStore.load(),
    }
    prompts = prepare_audit_prompts(review=review, **kwargs)
    assert all(record["docground_version"] == review["docground_version"] for record in prompts.values())
    changed = copy.deepcopy(review)
    changed["tasks"][0]["suggested_prompt"] += " altered"
    with pytest.raises(ValueError, match="differs from review"):
        prepare_audit_prompts(review=changed, **kwargs)


def test_empty_saved_response_is_not_synthesized():
    store = DocStore.load()
    prompt = ground("write f", store=store).text
    assert replay_record(
        original_task="write f", selected_prompt=prompt, saved_response="",
        provider="saved-provider", model="saved-model", store=store,
    ) == {"status": "no_saved_response", "network_calls": 0}


@pytest.mark.parametrize("functional_status, verdict", [("pass", "pass"), ("fail", "block"), ("unavailable", "warn")])
def test_functional_replay_uses_original_tests_image_and_distinct_results(monkeypatch, functional_status, verdict):
    import analysis.audit_docground_reuse as reuse
    from docground.verification.correctness import CorrectnessResult

    calls = []

    class RecordedRunner:
        def __init__(self, image):
            self.image = image

        def run(self, code, test, timeout):
            calls.append((self.image, code, test, timeout))
            return CorrectnessResult(functional_status)

    monkeypatch.setattr(reuse, "DockerCorrectnessRunner", RecordedRunner)
    store = DocStore.load()
    tests = "from candidate import f\ndef test_f(): assert f() == 1\n"
    code = "def f(): return 1"
    result = replay_record(
        original_task="write f", selected_prompt=ground("write f", store=store).text,
        saved_response=code, provider="saved", model="saved", store=store,
        test_code=tests, evaluator_image="frozen:1",
    )
    assert calls == [("frozen:1", code, tests, 15.0)]
    assert result["verification"]["verdict"] == verdict
    assert result["functional_status"] == functional_status
    assert result["functional_runner_invoked"] is True
    assert result["functional_tests_executed"] is (functional_status != "unavailable")
    assert "static_verification" not in result


def test_failed_reference_fixture_stops_saved_response_execution(monkeypatch):
    import analysis.audit_docground_reuse as reuse

    monkeypatch.setattr(reuse, "inspect_local_image", lambda image: {"available": True})
    monkeypatch.setattr(reuse, "validate_functional_fixtures", lambda *args: [{"task_id": "task", "functional_status": "unavailable"}])
    monkeypatch.setattr(reuse, "replay_record", lambda **kwargs: pytest.fail("Should stop before saved output execution"))
    result = {"errors": [], "audit_status": "pass", "grounded_records": []}
    add_functional_replay(result, tasks=[], store=DocStore.load(), evaluator_image="frozen:1", workers=4)
    assert result["audit_status"] == "incomplete"
    assert result["errors"][0]["stage"] == "functional_fixtures"
    assert result["functional_tests_executed"] is False


def test_docker_preflight_stops_before_any_execution(monkeypatch):
    import analysis.audit_docground_reuse as reuse

    monkeypatch.setattr(reuse, "inspect_local_image", lambda image: {"available": False, "error": "permission denied"})
    monkeypatch.setattr(reuse, "validate_functional_fixtures", lambda *args: pytest.fail("No Docker execution after failed preflight"))
    result = {"errors": [], "audit_status": "pass", "functional_tests_executed": False}
    add_functional_replay(result, tasks=[], store=DocStore.load(), evaluator_image="frozen:1", workers=4)
    assert result["errors"][0]["stage"] == "docker_preflight"
    assert result["audit_status"] == "incomplete"


@pytest.mark.parametrize("status, stderr, expected", [
    ("unavailable", "Docker daemon unavailable", "docker_unavailable"),
    ("timeout", "Sandbox exceeded 15 seconds\nCould not confirm cleanup of Docker container docground-check-abc.", "container_cleanup_unconfirmed"),
    ("pass", "Could not confirm cleanup of Docker container docground-check-abc.", "container_cleanup_unconfirmed"),
    ("timeout", "Sandbox exceeded 15 seconds", None),
    ("fail", "AssertionError", None),
])
def test_cleanup_uncertainty_is_an_environment_issue_not_a_candidate_failure(status, stderr, expected):
    assert replay_environment_issue(status, stderr) == expected


def test_cleanup_uncertainty_makes_functional_audit_incomplete(monkeypatch, tmp_path):
    import analysis.audit_docground_reuse as reuse

    monkeypatch.setattr(reuse, "inspect_local_image", lambda image: {"available": True})
    monkeypatch.setattr(reuse, "validate_functional_fixtures", lambda *args: [{"functional_status": "pass"}])
    monkeypatch.setattr(reuse, "replay_record", lambda **kwargs: {
        "functional_status": "timeout", "functional_runner_invoked": True,
        "functional_environment_issue": "container_cleanup_unconfirmed",
        "verification": {"verdict": "block"},
    })
    (tmp_path / "prompt_selected.txt").write_text("reviewed\n")
    (tmp_path / "response.txt").write_text("pass\n")
    record = {
        "artifact_dir": str(tmp_path), "source_file_sha256": {}, "source_run_id": "saved",
        "task_id": "task", "provider": "provider", "model_id": "model",
        "original_evaluation": {"evaluator_image": "frozen:1", "functional_status": "pass"},
        "offline_replay": {"status": "replayed_and_statically_verified"},
    }
    result = {
        "errors": [], "audit_status": "pass", "grounded_records": [record],
        "counts": {}, "limitations": [""] * 4,
    }
    add_functional_replay(
        result, tasks=[SimpleNamespace(task_id="task", shifted_prompt="task", test_code="assert True")],
        store=DocStore.load(), evaluator_image="frozen:1", workers=1,
    )
    assert result["audit_status"] == "incomplete"
    assert result["functional_comparison_status"] == "incomplete"
    assert result["counts"]["functional_infrastructure_errors"] == 1
    assert result["counts"]["functional_cleanup_uncertainties"] == 1
    assert record["original_evaluation"]["functional_status"] == "pass"


@pytest.mark.parametrize("functional", [False, True])
def test_cli_never_overwrites_an_existing_audit(monkeypatch, tmp_path, functional):
    import analysis.audit_docground_reuse as reuse

    output = tmp_path / "audit.json"
    output.write_text('{"original": true}\n')
    arguments = ["audit_docground_reuse", "--output", str(output)]
    if functional:
        arguments.append("--functional")
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(reuse, "audit", lambda *args, **kwargs: pytest.fail("Existing output must be checked before execution"))
    with pytest.raises(SystemExit) as error:
        reuse.main()
    assert error.value.code == 2
    assert output.read_text() == '{"original": true}\n'
