from experiments.evaluate import evaluate_python, evaluate_python_isolated
from experiments.schema import TaskSet
from pathlib import Path


def task():
    return TaskSet.from_path(Path("configs/pilot_tasks.json")).tasks[0]


def test_valid_solution_passes_fixture():
    result = evaluate_python(task(), "def sum_even_values(values):\n    return sum(v for v in values if v % 2 == 0)\n")
    assert result.syntax_status == "pass"
    assert result.functional_status == "pass"
    assert result.security_status == "pass"


def test_syntax_error_is_not_a_pass():
    result = evaluate_python(task(), "def sum_even_values(:\n")
    assert result.syntax_status == "fail"
    assert result.functional_status == "not_run"


def test_security_finding_is_preserved():
    result = evaluate_python(task(), "def sum_even_values(values):\n    eval('1 + 1')\n    return sum(values)\n")
    assert result.security_status == "findings"


def test_missing_docker_daemon_is_an_evaluator_error_not_model_failure(monkeypatch):
    import subprocess

    def fake_run(command, **kwargs):
        if "bandit" in command:
            return subprocess.CompletedProcess(command, 0, "{\"results\": []}", "")
        return subprocess.CompletedProcess(command, 1, "", "Cannot connect to the Docker daemon")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = evaluate_python_isolated(task(), task().reference_solution)
    assert result.functional_status == "evaluator_error"
    assert result.details["evaluator_error"] == "docker_runtime_unavailable"
    assert result.details.get("candidate_test_failure") is not True
