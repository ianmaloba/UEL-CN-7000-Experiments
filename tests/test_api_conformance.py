from pathlib import Path

from experiments.evaluate import check_api_conformance
from experiments.schema import TaskSet


def test_candidate_reference_solutions_use_the_declared_apis():
    taskset = TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json"))
    assert all(
        check_api_conformance(task, task.reference_solution or "") == "pass"
        for task in taskset.tasks
    )


def test_api_conformance_is_separate_from_functional_correctness():
    taskset = TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json"))
    task = next(item for item in taskset.tasks if item.task_id == "api-numpy-norm")
    pure_python = "def vector_length(values):\n    return sum(x*x for x in values) ** 0.5\n"
    assert check_api_conformance(task, pure_python) == "fail"


def test_required_instance_method_is_matched_conservatively():
    taskset = TaskSet.from_path(Path("configs/docground_api_pilot_v1_candidate.json"))
    task = next(item for item in taskset.tasks if item.task_id == "api-pandas-groupby")
    code = "import pandas as pd\nfrom io import StringIO\ndef f(s):\n d=pd.read_csv(StringIO(s))\n return d.groupby('region')['amount'].sum()\n"
    assert check_api_conformance(task, code) == "pass"
