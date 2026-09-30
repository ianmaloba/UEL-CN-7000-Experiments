import json
from pathlib import Path

import pytest
import experiments.run_calibration as calibration
from experiments.run_calibration import (
    _attempt_index, _effective_attempt_limit, _find_existing_attempt,
    _generation_status, _may_retry_error, _may_retry_record, _reserve_budget,
    _recorded_request_matches, _response_text, _retry_after_seconds, _retry_blocked,
    record_interrupted_attempt,
    request_parts, route_for, select_specs,
)
from experiments.costs import output_token_reserve, output_tokens_from_usage
from experiments.artefacts import write_run
from experiments.evaluate import Evaluation
from experiments.schema import Condition, RunManifest, TaskSet


def test_openai_route_families():
    assert route_for("openai", "gpt-5.5-pro-2026-04-23") == "responses"
    assert route_for("openai", "o1-pro-2025-03-19") == "responses"
    assert route_for("openai", "gpt-5.3-codex") == "responses"
    assert route_for("openai", "gpt-6-astra") == "responses"
    assert route_for("openai", "gpt-3.5-turbo-instruct-0914") == "completions"
    assert route_for("openai", "gpt-4.1-mini") == "chat_completions"
    assert route_for("xai", "grok-4.20-multi-agent") == "responses"


def test_openai_o_series_response_preserves_provider_reasoning_default():
    _, body, route, params = request_parts("openai", "o3-mini", "write f", 0, 1, 192)
    assert route == "responses"
    assert body["max_output_tokens"] == 192
    assert "temperature" not in body
    assert params["reasoning_effort"] == "provider_default"


def test_openai_responses_request_disables_storage_and_parses_output():
    url, body, route, params = request_parts("openai", "gpt-5-pro", "write f", 0, 1, 192)
    assert url.endswith("/v1/responses")
    assert route == "responses"
    assert body["store"] is False
    assert body["max_output_tokens"] == 192
    assert body["reasoning"] == {"effort": "medium"}
    assert params["reasoning_effort"] == "medium"
    assert _response_text({"output_text": "def f(): return 1"}, route) == "def f(): return 1"


def test_xai_multi_agent_uses_responses_route_and_supported_parameters():
    url, body, route, params = request_parts(
        "xai", "grok-4.20-multi-agent", "write f", 0, 1, 1024,
    )
    assert url == "https://api.x.ai/v1/responses"
    assert route == "responses"
    assert body["max_output_tokens"] == 1024
    assert body["reasoning"] == {"effort": "medium"}
    assert "max_tokens" not in body
    assert params["reasoning_effort"] == "medium"


def test_openai_chat_latest_uses_max_completion_tokens():
    _, body, route, params = request_parts("openai", "chat-latest", "write f", 0, 1, 512)
    assert route == "chat_completions"
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert params["max_completion_tokens"] == 512


def test_openai_pro_model_accepts_high_reasoning_effort():
    _, body, route, params = request_parts(
        "openai", "gpt-5.5-pro", "write f", 0, 1, 256, reasoning_effort="high",
    )
    assert route == "responses"
    assert body["reasoning"] == {"effort": "high"}
    assert params["reasoning_effort"] == "high"


def test_recorded_output_is_reused_only_when_request_parameters_match():
    prompt = "write f"
    _, _, route, params = request_parts(
        "openai", "gpt-5.5-pro", prompt, 0, 1, 360, reasoning_effort="high",
    )
    details = {
        "generation_status": "incomplete_response", "route": route,
        "request_params": params,
    }
    assert _recorded_request_matches(details, "openai", "gpt-5.5-pro", prompt, 0, 1, 360, "high")
    assert not _recorded_request_matches(details, "openai", "gpt-5.5-pro", prompt, 0, 1, 256, "high")


def test_budget_and_resume_reuse_matching_attempts_across_run_dates(tmp_path, capsys):
    taskset = TaskSet.from_path(Path("configs/humaneval_syntax_pilot_v1.json"))
    task = next(task for task in taskset.tasks if task.task_id == "humaneval-0")
    manifest = RunManifest.create(
        taskset_id=taskset.taskset_id, task=task, condition=Condition.BASELINE,
        provider="glm", model_id="glm-4.5-air", replicate=1, temperature=0,
        top_p=1, max_tokens=128, seed=None, protocol_version="2.1",
    )
    _, _, route, params = request_parts(
        "glm", "glm-4.5-air", task.prompt_for(Condition.BASELINE), 0, 1, 128,
    )
    old_manifest = manifest.model_copy(update={
        "run_id": "2020-01-01" + manifest.run_id[10:],
    })
    complete = Evaluation(
        "not_run", "not_run", "not_run", "not_run",
        {
            "generation_status": "complete", "route": route,
            "request_params": params, "estimated_cost_usd": 0.001,
        },
    )
    complete_path = write_run(tmp_path, old_manifest, task, "def solution(): return 1", complete)

    newer_error_manifest = manifest.model_copy(update={
        "run_id": "2099-01-01" + manifest.run_id[10:],
    })
    interrupted = Evaluation(
        "not_run", "not_run", "not_run", "not_run",
        {
            "generation_status": "transport_interrupted", "retryable": False,
            "billing_state": "unknown; interrupted request",
            "route": route, "request_params": params,
        },
    )
    write_run(tmp_path, newer_error_manifest, task, "", interrupted)

    shifted_manifest = RunManifest.create(
        taskset_id=taskset.taskset_id, task=task, condition=Condition.SHIFTED_BASELINE,
        provider="glm", model_id="glm-4.5-air", replicate=1, temperature=0,
        top_p=1, max_tokens=128, seed=None, protocol_version="2.1",
    )
    _, _, shifted_route, shifted_params = request_parts(
        "glm", "glm-4.5-air", task.prompt_for(Condition.SHIFTED_BASELINE), 0, 1, 128,
    )
    shifted_manifest = shifted_manifest.model_copy(update={
        "run_id": "2020-01-01" + shifted_manifest.run_id[10:],
    })
    write_run(
        tmp_path, shifted_manifest, task, "def solution(): return 1",
        Evaluation(
            "not_run", "not_run", "not_run", "not_run",
            {
                "generation_status": "complete", "route": shifted_route,
                "request_params": shifted_params, "estimated_cost_usd": 0.001,
            },
        ),
    )

    index = _attempt_index(tmp_path)
    found = _find_existing_attempt(
        tmp_path, index, manifest, task.prompt_for(Condition.BASELINE),
        0, 1, 128, "medium",
    )
    assert found is not None
    assert found[0] == complete_path

    config = {
        "budget_limit_per_provider_usd": 5.0, "budget_holdback_usd": 0.25,
        "max_attempts": 2, "task_ids": ["humaneval-0"], "temperature": 0,
        "top_p": 1, "max_tokens": 128, "reasoning_effort": "medium",
        "protocol_version": "2.1",
    }
    projected = _reserve_budget(
        config, taskset,
        [{
            "provider": "glm", "model_id": "glm-4.5-air",
            "rates_usd_per_million": {"input": 0.2, "output": 1.1},
        }],
        tmp_path,
    )
    assert projected["glm"] < 0.003
    assert "planned<=$0.0000" in capsys.readouterr().out


def test_legacy_completion_parsing():
    assert _response_text({"choices": [{"text": "def f(): return 1"}]}, "completions") == "def f(): return 1"


def test_non_openai_providers_use_chat_completion_parameters():
    for provider in ("glm", "deepseek", "xai", "mistral"):
        _, body, route, _ = request_parts(provider, "model-id", "write f", 0, 1, 192)
        assert route == "chat_completions"
        assert body["max_tokens"] == 192
        assert body["temperature"] == 0


def test_exact_campaign_subset_selection_and_empty_filter():
    models = [
        {"provider": "openai", "model_id": "gpt-5.6-luna"},
        {"provider": "xai", "model_id": "grok-4.7"},
    ]
    assert select_specs(models, "openai", "gpt-5.6-luna") == [models[0]]
    try:
        select_specs(models, "openai", "missing")
    except RuntimeError as error:
        assert "No campaign targets match" in str(error)
    else:
        raise AssertionError("an unmatched model filter must not silently run nothing")


def test_interrupted_inflight_attempt_is_preserved_with_unknown_billing(tmp_path):
    taskset = TaskSet.from_path(Path("configs/humaneval_syntax_pilot_v1.json"))
    task = next(task for task in taskset.tasks if task.task_id == "humaneval-4")
    manifest = RunManifest.create(
        taskset_id=taskset.taskset_id, task=task, condition=Condition.BASELINE,
        provider="openai", model_id="gpt-5.5-pro", replicate=1,
        temperature=0, top_p=1, max_tokens=256, seed=None,
        protocol_version="2.0", generation_route="responses", reasoning_effort="medium",
    )
    path = record_interrupted_attempt(
        tmp_path, manifest, task, route="responses", endpoint="https://api.openai.com/v1/responses",
        request_params={"reasoning_effort": "medium", "max_output_tokens": 256},
    )
    details = json.loads((path / "evaluation.json").read_text())["details"]
    assert details["generation_status"] == "transport_interrupted"
    assert details["retryable"] is False
    assert "billing_state" in details and "unknown" in details["billing_state"]


def test_exact_repair_run_allows_bounded_append_only_repair_ladder():
    assert _effective_attempt_limit(2, False) == 2
    assert _effective_attempt_limit(2, True) == 12


def test_completed_output_and_unknown_interrupted_request_are_not_resampled():
    assert not _may_retry_record({"generation_status": "complete"})
    assert not _may_retry_record({"generation_status": "incomplete_response"})
    assert not _may_retry_record({"generation_status": "empty_response"})
    assert not _may_retry_record({"generation_status": "transport_interrupted"})
    assert not _may_retry_record({"generation_status": "transport_or_parse_error"})
    assert not _may_retry_record({"generation_status": "provider_error", "http_status": 403})


def test_explicit_recovery_retries_only_429_or_authorized_ambiguous_transport():
    transport = {"generation_status": "transport_or_parse_error"}
    assert not _may_retry_error(transport)
    assert _may_retry_error(transport, allow_uncertain_billing=True)
    assert _may_retry_error({
        "generation_status": "provider_error", "http_status": 429, "retryable": True,
    })
    assert not _may_retry_error({
        "generation_status": "provider_error", "http_status": 400, "retryable": False,
    }, allow_uncertain_billing=True)
    assert not _may_retry_error({
        "generation_status": "incomplete_response", "finish_reason": "length",
    }, allow_uncertain_billing=True)


def test_exact_recovery_may_retry_ambiguous_transport_but_not_provider_failures():
    uncertain = {"generation_status": "transport_or_parse_error"}
    assert _retry_blocked(uncertain)
    assert not _retry_blocked(uncertain, allow_uncertain_billing=True)
    assert _retry_blocked(
        {"generation_status": "provider_error", "http_status": 503},
        allow_uncertain_billing=True,
    )


def test_xai_cost_usage_includes_hidden_reasoning_tokens_and_provider_ticks():
    from experiments.run_calibration import _usage_cost

    usage = {
        "completion_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 654},
        "prompt_tokens": 222,
        "cost_in_usd_ticks": 17_734_000,
    }
    assert output_tokens_from_usage("xai", 25, usage) == 679
    assert output_tokens_from_usage("glm", 25, usage) == 25
    assert output_token_reserve("xai", 512) == 1024
    assert output_token_reserve("glm", 512) == 512
    assert _usage_cost("xai", "grok-4.20", {"usage": usage}) == 0.0017734


def test_runtime_budget_recheck_stops_before_next_provider_call(tmp_path, monkeypatch):
    task_ids = ["sum-even-values", "normalize-whitespace"]
    config = {
        "protocol_version": "budget-runtime-check-test",
        "taskset_path": "configs/pilot_tasks.json",
        "task_ids": task_ids,
        "conditions": ["baseline"],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 32,
        "timeout_seconds": 1,
        "max_attempts": 1,
        "reasoning_effort": "medium",
        "evaluation_mode": "static",
        "evaluator_image": "unused-in-static-test",
        "budget_accounting_root": str(tmp_path / "runs"),
        "budget_limit_per_provider_usd": 0.01,
        "budget_holdback_usd": 0.001,
        "models": [{
            "provider": "glm", "model_id": "glm-4.5",
            "rates_usd_per_million": {"input": 1.0, "output": 1.0},
        }],
    }
    config_path = tmp_path / "runtime-budget-test.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []

    def fake_generate(provider, model, prompt, temperature, top_p, max_tokens, timeout, effort):
        calls.append((provider, model, prompt))
        _, _, route, params = request_parts(
            provider, model, prompt, temperature, top_p, max_tokens, effort,
        )
        return "def placeholder(): return 1", {
            "generation_status": "complete", "route": route,
            "request_params": params,
            "usage": {"cost_in_usd_ticks": 95_000_000},
        }

    monkeypatch.setattr(calibration, "generate", fake_generate)
    monkeypatch.setattr(
        calibration, "evaluate_static_python",
        lambda task, response: Evaluation("pass", "pass", "pass", "pass", {}),
    )

    with pytest.raises(RuntimeError, match="before the next API call"):
        calibration.run(config_path, tmp_path / "runs")

    assert len(calls) == 1
    assert len(list((tmp_path / "runs").glob("*/run_manifest.json"))) == 1


def test_empty_but_truncated_responses_are_classified_as_incomplete():
    assert _generation_status("", "length") == "incomplete_response"
    assert _generation_status("", "incomplete") == "incomplete_response"
    assert _generation_status("", "stop") == "empty_response"
    assert _generation_status("def f(): pass", "stop") == "complete"


def test_retry_after_header_parses_seconds_and_http_date():
    from email.message import Message

    headers = Message()
    headers["Retry-After"] = "7"
    assert _retry_after_seconds(headers) == 7
    headers.replace_header("Retry-After", "not a date or seconds")
    assert _retry_after_seconds(headers) is None
