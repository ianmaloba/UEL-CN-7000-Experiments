import json
from pathlib import Path

import pytest

from experiments.run_calibration import _configured_conditions, _configured_scope, _prepare_prompt_map
from experiments.schema import Condition, TaskSet


def test_existing_campaign_default_does_not_add_docground_arm():
    assert _configured_conditions({}) == [Condition.BASELINE, Condition.SHIFTED_BASELINE]


def test_exact_slot_filters_only_select_requested_task_and_condition():
    taskset = TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json"))
    config = {
        "task_ids": [task.task_id for task in taskset.tasks],
        "conditions": [condition.value for condition in Condition],
    }

    tasks, conditions = _configured_scope(
        taskset, config, task_id_filter="api-pandas-merge",
        condition_filter="shifted_docground",
    )

    assert [task.task_id for task in tasks] == ["api-pandas-merge"]
    assert conditions == [Condition.SHIFTED_DOCGROUND]


def test_unapproved_docground_review_is_rejected_before_generation(tmp_path):
    taskset_path = Path("configs/docground_api_pilot_v1_candidate.json")
    taskset = TaskSet.from_path(taskset_path)
    review = json.loads(Path(
        "results/protocol_review/docground_api_pilot_v1_candidate_v3.json"
    ).read_text(encoding="utf-8"))
    review["status"] = "review_pending"
    for item in review["tasks"]:
        item["approval"] = "pending"
    review_path = tmp_path / "review_pending.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    config = {
        "taskset_path": str(taskset_path),
        "docground_review_path": str(review_path),
    }

    with pytest.raises(RuntimeError, match="review is not approved"):
        _prepare_prompt_map(
            taskset,
            taskset.tasks,
            config,
            [Condition.SHIFTED_DOCGROUND],
        )
