from analysis.aggregate_model_coverage import DEFAULT_CONDITIONS, make_tables
from experiments.schema import Condition


def test_aggregate_defaults_remain_two_arm_for_existing_campaigns():
    assert DEFAULT_CONDITIONS == [
        Condition.BASELINE.value,
        Condition.SHIFTED_BASELINE.value,
    ]


def test_three_arm_summary_keeps_docground_pair_separate(tmp_path):
    conditions = [
        Condition.BASELINE.value,
        Condition.SHIFTED_BASELINE.value,
        Condition.SHIFTED_DOCGROUND.value,
    ]
    config = {
        "conditions": conditions,
        "models": [{"provider": "example", "model_id": "model-1"}],
        "task_ids": ["task-1"],
        "budget_accounting_root": str(tmp_path),
    }
    latest = [
        {
            "provider": "example",
            "model_id": "model-1",
            "task_id": "task-1",
            "condition": condition,
            "generation_status": "complete",
            "functional_status": "pass",
            "syntax_status": "pass",
            "security_status": "pass",
            "hallucination_status": "not_run",
            "api_conformance_status": "pass",
            "request_matches_current_config": True,
        }
        for condition in conditions
    ]

    provider_summary, pairs, _ = make_tables(config, [], latest)

    assert provider_summary.iloc[0]["planned_outputs"] == 3
    assert pairs.iloc[0]["pair_generation_status"] == "complete_pair"
    assert pairs.iloc[0]["grounding_core_pair_generation_status"] == "complete_pair"
    assert pairs.iloc[0]["shifted_docground_functional_status"] == "pass"


def test_functional_timeout_is_counted_separately_from_failure_and_not_run(tmp_path):
    config = {
        "conditions": [Condition.BASELINE.value],
        "models": [{"provider": "example", "model_id": "model-1"}],
        "task_ids": ["task-1"],
        "budget_accounting_root": str(tmp_path),
    }
    latest = [{
        "provider": "example",
        "model_id": "model-1",
        "task_id": "task-1",
        "condition": Condition.BASELINE.value,
        "generation_status": "complete",
        "functional_status": "timeout",
        "api_conformance_status": "pass",
    }]

    summary, _, _ = make_tables(config, [], latest)

    assert summary.iloc[0]["functional_timeout_outputs"] == 1
    assert summary.iloc[0]["functional_fail_outputs"] == 0
    assert summary.iloc[0]["functional_not_run_outputs"] == 0
