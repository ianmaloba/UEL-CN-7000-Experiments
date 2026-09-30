import sys
import types
from pathlib import Path

import pytest

from experiments.schema import TaskSet


@pytest.mark.parametrize(
    "task",
    TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json")).tasks,
    ids=lambda task: task.task_id,
)
def test_reference_solution_satisfies_its_local_behavioral_tests(task):
    candidate = types.ModuleType("candidate")
    exec(compile(task.reference_solution or "", task.task_id, "exec"), candidate.__dict__)
    previous = sys.modules.get("candidate")
    sys.modules["candidate"] = candidate
    try:
        namespace = {"__name__": f"tests_{task.task_id}"}
        exec(compile(task.test_code, f"{task.task_id}_tests", "exec"), namespace)
        test_functions = [
            value for name, value in namespace.items()
            if name.startswith("test_") and callable(value)
        ]
        assert test_functions, "task must define at least one behavioral test"
        for test_function in test_functions:
            test_function()
    finally:
        if previous is None:
            sys.modules.pop("candidate", None)
        else:
            sys.modules["candidate"] = previous
