from pathlib import Path

from experiments.schema import Condition, RunManifest, TaskSet


def test_pilot_taskset_is_valid():
    taskset = TaskSet.from_path(Path("configs/pilot_tasks.json"))
    assert len(taskset.tasks) == 2
    assert taskset.tasks[0].prompt_for(Condition.SHIFTED_BASELINE) == taskset.tasks[0].shifted_prompt


def test_manifest_hashes_selected_prompt():
    taskset = TaskSet.from_path(Path("configs/pilot_tasks.json"))
    manifest = RunManifest.create(
        taskset_id=taskset.taskset_id, task=taskset.tasks[0], condition=Condition.BASELINE,
        provider="fixture", model_id="fixture-model", replicate=1, temperature=0,
        top_p=1, max_tokens=100, seed=1,
    )
    assert len(manifest.prompt_sha256) == 64
    assert manifest.condition is Condition.BASELINE


def test_docground_api_candidate_tasks_have_explicit_api_targets():
    taskset = TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json"))
    assert len(taskset.tasks) == 6
    assert all(task.shift_proxy.value == "api" for task in taskset.tasks)
    assert all(task.required_apis for task in taskset.tasks)
    assert all(task.status == "candidate_review_required" for task in taskset.tasks)
