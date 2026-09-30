from pathlib import Path

from experiments.artefacts import write_run
from experiments.evaluate import evaluate_python
from experiments.schema import Condition, RunManifest, TaskSet


def test_writer_redacts_and_refuses_overwrite(tmp_path, monkeypatch):
    taskset = TaskSet.from_path(Path("configs/pilot_tasks.json"))
    task = taskset.tasks[0]
    manifest = RunManifest.create(
        taskset_id=taskset.taskset_id, task=task, condition=Condition.BASELINE,
        provider="fixture", model_id="fixture-model", replicate=1, temperature=0,
        top_p=1, max_tokens=100, seed=1,
    )
    evaluation = evaluate_python(task, "def sum_even_values(values):\n    return sum(v for v in values if v % 2 == 0)\n")
    monkeypatch.setenv("TEST_API_KEY", "a-test-secret-value")
    destination = write_run(tmp_path, manifest, task, "a-test-secret-value", evaluation)
    assert "a-test-secret-value" not in (destination / "response.txt").read_text()
    assert "[REDACTED:TEST_API_KEY]" in (destination / "response.txt").read_text()
    assert (destination / "metrics.csv").exists()
