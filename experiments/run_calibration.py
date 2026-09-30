"""Bounded, resumable live generation with route-aware adapters and audit trails."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .artefacts import write_run
from .costs import output_token_reserve, output_tokens_from_usage, rates_for, token_cost
from .evaluate import Evaluation, evaluate_python_isolated, evaluate_static_python
from .providers import PROVIDERS, load_dotenv
from .schema import Condition, RunManifest, Task, TaskSet

CHAT_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "glm": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
}
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
OPENAI_COMPLETIONS_URL = "https://api.openai.com/v1/completions"
COMPLETION_ONLY = re.compile(r"^(babbage-002|davinci-002|gpt-3\.5-turbo-instruct(?:-.+)?)$")
RESPONSES_ONLY = re.compile(r"(?:^o1-pro(?:-|$)|^gpt-5(?:\.[0-9]+)?-pro(?:-|$))", re.I)
REASONING_MODEL = re.compile(r"^(?:o[134](?:-|$)|gpt-[56](?:\.|-|$))", re.I)
CHAT_COMPLETION_REASONING_MODEL = re.compile(r"^(?:o[134](?:-|$)|gpt-[56](?:\.|-|$)|chat-latest$)", re.I)


def route_for(provider: str, model: str) -> str:
    if provider == "xai" and "multi-agent" in model.lower():
        return "responses"
    if provider != "openai":
        return "chat_completions"
    if COMPLETION_ONLY.match(model):
        return "completions"
    if RESPONSES_ONLY.search(model) or REASONING_MODEL.search(model):
        return "responses"
    return "chat_completions"


def request_parts(
    provider: str, model: str, prompt: str, temperature: float, top_p: float,
    max_tokens: int, reasoning_effort: str = "medium",
) -> tuple[str, dict[str, object], str, dict[str, object]]:
    """Build a vendor request without credentials; route and parameter policy are testable."""
    user_prompt = prompt + "\nReturn only Python code in one fenced block."
    route = route_for(provider, model)
    if route == "completions":
        return OPENAI_COMPLETIONS_URL, {
            "model": model, "prompt": user_prompt, "temperature": temperature,
            "top_p": top_p, "max_tokens": max_tokens,
        }, route, {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if route == "responses":
        body: dict[str, object] = {
            "model": model, "input": user_prompt, "max_output_tokens": max_tokens,
        }
        params: dict[str, object] = {"max_output_tokens": max_tokens}
        if model.lower().startswith(("gpt-5", "gpt-6")):
            body["reasoning"] = {"effort": reasoning_effort}
            params["reasoning_effort"] = reasoning_effort
        elif provider == "xai" and "multi-agent" in model.lower():
            body["reasoning"] = {"effort": reasoning_effort}
            params["reasoning_effort"] = reasoning_effort
        else:
            params["reasoning_effort"] = "provider_default"
        if provider == "openai":
            body["store"] = False
            params["store"] = False
            endpoint = OPENAI_RESPONSES_URL
        else:
            endpoint = XAI_RESPONSES_URL
        return endpoint, body, route, params

    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if provider == "openai" and CHAT_COMPLETION_REASONING_MODEL.search(model):
        body["max_completion_tokens"] = max_tokens
        params: dict[str, object] = {"max_completion_tokens": max_tokens, "sampling": "provider_default_for_reasoning_model"}
    elif provider == "openai":
        body.update({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    else:
        body.update({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})
        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    return CHAT_URLS[provider], body, route, params


def _response_text(data: dict[str, object], route: str) -> str:
    if route == "completions":
        choices = data.get("choices", [])
        return str(choices[0].get("text") or "") if choices else ""
    if route == "responses":
        direct = data.get("output_text")
        if isinstance(direct, str):
            return direct
        chunks: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        return "\n".join(chunks)
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _generation_status(text: str, finish_reason: object) -> str:
    if finish_reason in {"incomplete", "length", "max_tokens"}:
        return "incomplete_response"
    return "complete" if text.strip() else "empty_response"


def _retry_after_seconds(headers: object) -> float | None:
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            reset_at = parsedate_to_datetime(value)
            now = datetime.now(reset_at.tzinfo or UTC)
            return max(0.0, (reset_at - now).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def generate(
    provider: str, model: str, prompt: str, temperature: float, top_p: float,
    max_tokens: int, timeout_seconds: int = 300, reasoning_effort: str = "medium",
) -> tuple[str, dict[str, object]]:
    configured = next((p for p in PROVIDERS if p.name == provider), None)
    if configured is None:
        return "", {"generation_status": "configuration_error", "provider_error": "unknown_provider"}
    token = os.environ.get(configured.environment_key)
    if not token:
        return "", {"generation_status": "configuration_error", "provider_error": "missing_api_key"}
    url, body, route, request_params = request_parts(
        provider, model, prompt, temperature, top_p, max_tokens, reasoning_effort,
    )
    request = Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    base = {"route": route, "endpoint": url, "request_params": request_params}
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
            text = _response_text(data, route)
            usage = data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {}
            finish_reason = (
                data.get("status") if route == "responses"
                else (data.get("choices") or [{}])[0].get("finish_reason")
            )
            status = _generation_status(text, finish_reason)
            metadata = {
                **base, "generation_status": status, "http_status": response.status,
                "request_id": response.headers.get("x-request-id") or response.headers.get("request-id"),
                "actual_model_id": data.get("model"), "provider_response_id": data.get("id"),
                "usage": usage, "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "finish_reason": finish_reason,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            return text, metadata
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        from .artefacts import redact_secrets
        retryable = error.code == 429
        retry_after = _retry_after_seconds(error.headers)
        rate_headers = {
            name.lower(): error.headers.get(name)
            for name in (
                "Retry-After", "X-RateLimit-Remaining", "X-RateLimit-Limit",
                "X-RateLimit-Reset", "RateLimit-Remaining", "RateLimit-Reset",
            ) if error.headers.get(name) is not None
        }
        return "", {
            **base, "generation_status": "provider_error", "provider_error": "http_error",
            "http_status": error.code, "provider_error_detail": redact_secrets(raw)[:1500],
            "request_id": error.headers.get("x-request-id") or error.headers.get("request-id"),
            "retry_after_seconds": retry_after, "response_rate_limit_headers": rate_headers,
            "retryable": retryable, "retry_policy": "429 only; other errors may have uncertain billing",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except (URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as error:
        return "", {
            **base, "generation_status": "transport_or_parse_error",
            "provider_error": type(error).__name__, "retryable": False,
            "billing_state": "unknown; not automatically resent",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def _tokens(prompt: str) -> int:
    # For a short English/code prompt, three UTF-8 characters/token is a
    # deliberately conservative reservation; actual usage replaces it later.
    return max(1, math.ceil(len(prompt.encode("utf-8")) / 3)) + 12


def _read_details(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    try:
        evaluation = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        return evaluation.get("details", {}), manifest
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}, {}


def _manifest_lookup_key(manifest: dict[str, object]) -> tuple[str, ...] | None:
    condition = manifest.get("condition")
    if isinstance(condition, Condition):
        condition = condition.value
    fields = (
        manifest.get("taskset_id"), manifest.get("task_id"), condition,
        manifest.get("model_provider"), manifest.get("model_id"),
        manifest.get("replicate"), manifest.get("prompt_sha256"),
    )
    if any(value is None for value in fields):
        return None
    return tuple(str(value) for value in fields)


def _attempt_index(result_root: Path) -> dict[tuple[str, ...], dict[int, list[Path]]]:
    """Index immutable attempts by logical run identity, independent of run date."""
    index: dict[tuple[str, ...], dict[int, list[Path]]] = {}
    for manifest_path in result_root.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            key = _manifest_lookup_key(manifest)
            attempt = int(manifest["attempt"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if key is not None:
            index.setdefault(key, {}).setdefault(attempt, []).append(manifest_path.parent)
    return index


def _add_attempt_to_index(
    index: dict[tuple[str, ...], dict[int, list[Path]]], manifest: RunManifest, path: Path,
) -> None:
    key = _manifest_lookup_key(manifest.model_dump(mode="json"))
    if key is not None:
        index.setdefault(key, {}).setdefault(manifest.attempt, []).append(path)


def _usage_cost(provider: str, model: str, details: dict[str, object]) -> float | None:
    estimate = details.get("estimated_cost_usd")
    if isinstance(estimate, (int, float)):
        return float(estimate)
    usage = details.get("usage", {})
    if not isinstance(usage, dict):
        return None
    ticks = usage.get("cost_in_usd_ticks")
    if isinstance(ticks, int):
        return ticks / 10_000_000_000
    input_tokens = details.get("input_tokens", usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = details.get("output_tokens", usage.get("completion_tokens", usage.get("output_tokens")))
    rates = rates_for(provider, model, details.get("catalog_metadata") if isinstance(details.get("catalog_metadata"), dict) else None)
    if rates is None or not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    billable_output_tokens = output_tokens_from_usage(provider, output_tokens, usage)
    return token_cost(input_tokens, billable_output_tokens, rates)


def _historical_spend(result_root: Path) -> dict[str, float]:
    spend: dict[str, float] = {}
    for path in result_root.rglob("evaluation.json"):
        details, manifest = _read_details(path.parent)
        provider, model = manifest.get("model_provider"), manifest.get("model_id")
        if not isinstance(provider, str) or not isinstance(model, str):
            continue
        cost = _usage_cost(provider, model, details)
        if cost is not None:
            spend[provider] = spend.get(provider, 0.0) + cost
    return spend


def _uncertain_billing_reserve(result_root: Path) -> tuple[dict[str, float], list[str]]:
    """Reserve worst-case request cost when a call may have billed but lacks usage."""
    reserve: dict[str, float] = {}
    unknown_rates: list[str] = []
    for path in result_root.rglob("evaluation.json"):
        details, manifest = _read_details(path.parent)
        provider, model = manifest.get("model_provider"), manifest.get("model_id")
        if not isinstance(provider, str) or not isinstance(model, str):
            continue
        status = details.get("generation_status")
        billing_state = str(details.get("billing_state", "")).lower()
        uncertain = status in {"transport_interrupted", "transport_or_parse_error"} or "unknown" in billing_state
        if not uncertain or _usage_cost(provider, model, details) is not None:
            continue
        rates = rates_for(
            provider, model,
            details.get("catalog_metadata") if isinstance(details.get("catalog_metadata"), dict) else None,
        )
        cap = manifest.get("max_tokens")
        condition = manifest.get("condition")
        prompt_name = "prompt_original.txt" if condition == "baseline" else "prompt_shifted.txt"
        prompt_path = path.parent / prompt_name
        if rates is None or not isinstance(cap, int) or not prompt_path.exists():
            unknown_rates.append(f"{provider}:{model}")
            continue
        prompt_tokens = _tokens(prompt_path.read_text(encoding="utf-8"))
        upper_bound = token_cost(prompt_tokens, output_token_reserve(provider, cap), rates)
        reserve[provider] = reserve.get(provider, 0.0) + upper_bound
    return reserve, unknown_rates


def _configured_rates(spec: dict[str, object]) -> tuple[float, float] | None:
    explicit = spec.get("rates_usd_per_million")
    if isinstance(explicit, dict):
        input_rate, output_rate = explicit.get("input"), explicit.get("output")
        if isinstance(input_rate, (int, float)) and isinstance(output_rate, (int, float)):
            return float(input_rate), float(output_rate)
    metadata = spec.get("catalog_metadata")
    return rates_for(str(spec["provider"]), str(spec["model_id"]), metadata if isinstance(metadata, dict) else None)


def _request_parameters_match(
    details: dict[str, object], provider: str, model: str, prompt: str,
    temperature: float, top_p: float, max_tokens: int, reasoning_effort: str,
) -> bool:
    _, _, route, params = request_parts(
        provider, model, prompt, temperature, top_p, max_tokens, reasoning_effort,
    )
    return details.get("route") == route and details.get("request_params") == params


def _recorded_request_matches(
    details: dict[str, object], provider: str, model: str, prompt: str,
    temperature: float, top_p: float, max_tokens: int, reasoning_effort: str,
) -> bool:
    if details.get("generation_status") not in {"complete", "incomplete_response", "empty_response"}:
        return False
    return _request_parameters_match(
        details, provider, model, prompt, temperature, top_p, max_tokens, reasoning_effort,
    )


def _find_existing_attempt(
    result_root: Path, index: dict[tuple[str, ...], dict[int, list[Path]]],
    manifest: RunManifest, prompt: str, temperature: float, top_p: float,
    max_tokens: int, reasoning_effort: str,
) -> tuple[Path, dict[str, object]] | None:
    key = _manifest_lookup_key(manifest.model_dump(mode="json"))
    if key is None:
        return None
    candidates: list[tuple[int, int, Path, dict[str, object]]] = []
    paths = index.get(key, {}).get(manifest.attempt, [])
    current_path = result_root / manifest.run_id
    if current_path.exists() and current_path not in paths:
        paths = [*paths, current_path]
    for path in paths:
        details, stored_manifest = _read_details(path)
        if _manifest_lookup_key(stored_manifest) != key or not _request_parameters_match(
            details, manifest.model_provider, manifest.model_id, prompt, temperature,
            top_p, max_tokens, reasoning_effort,
        ):
            continue
        status = details.get("generation_status")
        if status == "complete":
            quality = 4
        elif status in {"incomplete_response", "empty_response"}:
            quality = 3
        elif status in {"transport_interrupted", "transport_or_parse_error"}:
            quality = 2
        else:
            quality = 1
        candidates.append((quality, path.stat().st_mtime_ns, path, details))
    if not candidates:
        return None
    _, _, path, details = max(candidates, key=lambda item: (item[0], item[1]))
    return path, details


def _effective_attempt_limit(max_attempts: int, retry_recorded_errors: bool) -> int:
    """Permit bounded append-only repair attempts in exact-model recovery runs."""
    return max_attempts + (10 if retry_recorded_errors else 0)


def _retry_blocked(details: dict[str, object], *, allow_uncertain_billing: bool = False) -> bool:
    status = details.get("generation_status")
    if status in {"transport_interrupted", "transport_or_parse_error"}:
        # An exact-model repair run is an explicit authorization to risk a duplicate
        # charge for a response that may have been accepted but not observed. The
        # budget gate separately reserves the worst-case cost of the earlier call.
        return not allow_uncertain_billing
    http_status = details.get("http_status")
    if isinstance(http_status, int):
        if http_status in {403, 404} or http_status >= 500:
            return True
        if status == "provider_error" and http_status not in {400, 429}:
            return True
    return False


def _may_retry_record(details: dict[str, object]) -> bool:
    """Never resample a valid response, including empty or token-truncated output."""
    return False


def _may_retry_error(
    details: dict[str, object], *, allow_uncertain_billing: bool = False,
) -> bool:
    """Retry only an explicit 429 or an authorized ambiguous transport failure."""
    status = details.get("generation_status")
    if status in {"transport_interrupted", "transport_or_parse_error"}:
        return allow_uncertain_billing
    return (
        status == "provider_error"
        and details.get("http_status") == 429
        and details.get("retryable") is True
    )


def record_interrupted_attempt(
    result_root: Path, manifest: RunManifest, task: Task, *, route: str,
    endpoint: str, request_params: dict[str, object], catalog_metadata: object = None,
    selected_prompt: str | None = None,
) -> Path:
    """Persist an in-flight cancellation without claiming a response or safe retry."""
    details: dict[str, object] = {
        "generation_status": "transport_interrupted",
        "provider_error": "KeyboardInterrupt",
        "retryable": False,
        "billing_state": "unknown; interrupted during an in-flight request; not automatically resent",
        "route": route,
        "endpoint": endpoint,
        "request_params": request_params,
        "attempt": manifest.attempt,
        "catalog_metadata": catalog_metadata,
    }
    evaluation = Evaluation("not_run", "not_run", "not_run", "not_run", details)
    return write_run(
        result_root, manifest, task, "", evaluation,
        selected_prompt=selected_prompt,
    )


_DEFAULT_CONDITIONS = (Condition.BASELINE, Condition.SHIFTED_BASELINE)


def _configured_conditions(config: dict[str, object]) -> list[Condition]:
    values = config.get("conditions", [item.value for item in _DEFAULT_CONDITIONS])
    if not isinstance(values, list) or not values:
        raise RuntimeError("Campaign conditions must be a non-empty list")
    try:
        conditions = [Condition(value) for value in values]
    except ValueError as error:
        raise RuntimeError(f"Campaign contains an unsupported condition: {error}") from error
    if len(conditions) != len(set(conditions)):
        raise RuntimeError("Campaign conditions must be unique")
    return conditions


def _configured_scope(
    taskset: TaskSet, config: dict[str, object],
    task_id_filter: str | None = None,
    condition_filter: str | None = None,
) -> tuple[list[Task], list[Condition]]:
    configured_ids = config.get("task_ids")
    tasks = [
        task for task in taskset.tasks
        if not configured_ids or task.task_id in configured_ids
    ]
    if task_id_filter is not None:
        tasks = [task for task in tasks if task.task_id == task_id_filter]
    if not tasks:
        raise RuntimeError("No configured task IDs were present in the selected task set")
    conditions = _configured_conditions(config)
    if condition_filter is not None:
        try:
            selected_condition = Condition(condition_filter)
        except ValueError as error:
            raise RuntimeError(f"Unsupported condition filter: {condition_filter}") from error
        if selected_condition not in conditions:
            raise RuntimeError(f"Condition filter is outside the configured campaign: {condition_filter}")
        conditions = [selected_condition]
    return tasks, conditions


def _prepare_prompt_map(
    taskset: TaskSet,
    tasks: list[Task],
    config: dict[str, object],
    conditions: list[Condition],
) -> dict[tuple[str, Condition], dict[str, object]]:
    """Resolve every prompt before budget reservation or any provider request."""
    prompt_map: dict[tuple[str, Condition], dict[str, object]] = {}
    for task in tasks:
        for condition in conditions:
            prompt_map[(task.task_id, condition)] = {
                "prompt": task.prompt_for(condition),
                "documentation_snapshot_sha256": None,
                "docground_version": None,
                "prompt_decision": None,
                "docground_review_sha256": None,
            }
    if Condition.SHIFTED_DOCGROUND not in conditions:
        return prompt_map

    review_path_value = config.get("docground_review_path")
    if not isinstance(review_path_value, str) or not review_path_value:
        raise RuntimeError("shifted_docground requires docground_review_path")
    review_path = Path(review_path_value)
    review_bytes = review_path.read_bytes()
    try:
        review = json.loads(review_bytes)
    except json.JSONDecodeError as error:
        raise RuntimeError("DocGround prompt-review file is not valid JSON") from error
    if review.get("status") != "approved":
        raise RuntimeError("DocGround prompt review is not approved; no API calls were made")
    taskset_path = Path(str(config["taskset_path"]))
    taskset_sha256 = hashlib.sha256(taskset_path.read_bytes()).hexdigest()
    if review.get("taskset_id") != taskset.taskset_id or review.get("taskset_sha256") != taskset_sha256:
        raise RuntimeError("DocGround review task-set ID or hash does not match the campaign")

    from docground.grounding.doc_store import DocStore
    from docground.grounding.wrapper import ground

    store = DocStore.load()
    docground_version = importlib.metadata.version("docground")
    if review.get("documentation_snapshot_sha256") != store.snapshot_sha256:
        raise RuntimeError("DocGround documentation snapshot differs from the approved review")
    if review.get("docground_version") != docground_version:
        raise RuntimeError("Installed DocGround version differs from the approved review")
    reviewed_tasks = {
        item.get("task_id"): item for item in review.get("tasks", [])
        if isinstance(item, dict)
    }
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    for task in tasks:
        record = reviewed_tasks.get(task.task_id)
        if not isinstance(record, dict) or record.get("approval") != "approve_suggestion":
            raise RuntimeError(f"DocGround prompt for {task.task_id} is not explicitly approved")
        proposal = ground(task.shifted_prompt, store=store)
        hashes = record.get("prompt_hashes", {})
        shifted_hash = hashlib.sha256(task.shifted_prompt.encode("utf-8")).hexdigest()
        suggested_hash = hashlib.sha256(proposal.text.encode("utf-8")).hexdigest()
        if not isinstance(hashes, dict) or hashes.get("shifted") != shifted_hash:
            raise RuntimeError(f"Shifted prompt for {task.task_id} differs from the reviewed prompt")
        if hashes.get("suggested") != suggested_hash:
            raise RuntimeError(f"Grounded prompt for {task.task_id} differs from the reviewed suggestion")
        if record.get("documentation_snapshot_sha256") != proposal.snapshot_sha256:
            raise RuntimeError(f"Documentation snapshot for {task.task_id} differs from review")
        prompt_map[(task.task_id, Condition.SHIFTED_DOCGROUND)] = {
            "prompt": proposal.text,
            "documentation_snapshot_sha256": proposal.snapshot_sha256,
            "docground_version": docground_version,
            "prompt_decision": "approve_suggestion",
            "docground_review_sha256": review_sha256,
        }
    return prompt_map


def _reserve_budget(
    config: dict[str, object], taskset: TaskSet, specs: list[dict[str, object]],
    result_root: Path, retry_recorded_errors: bool = False,
    attempt_index: dict[tuple[str, ...], dict[int, list[Path]]] | None = None,
    prompt_map: dict[tuple[str, Condition], dict[str, object]] | None = None,
    conditions: list[Condition] | None = None,
    tasks: list[Task] | None = None,
    emit: bool = True,
) -> dict[str, float]:
    limit = float(config.get("budget_limit_per_provider_usd", 5.0))
    reserve = float(config.get("budget_holdback_usd", 0.50))
    max_attempts = int(config.get("max_attempts", 2))
    attempt_limit = _effective_attempt_limit(max_attempts, retry_recorded_errors)
    task_ids = config.get("task_ids")
    tasks = tasks or [task for task in taskset.tasks if not task_ids or task.task_id in task_ids]
    selected_conditions = conditions or _configured_conditions(config)
    selected_prompts = prompt_map or _prepare_prompt_map(taskset, tasks, config, selected_conditions)
    if attempt_index is None:
        attempt_index = _attempt_index(result_root)
    planned: dict[str, float] = {}
    unknown: list[str] = []
    for spec in specs:
        provider, model = str(spec["provider"]), str(spec["model_id"])
        token_cap = int(spec.get("max_tokens", config["max_tokens"]))
        reasoning_effort = str(spec.get("reasoning_effort", config.get("reasoning_effort", "medium")))
        rates = _configured_rates(spec)
        if rates is None:
            unknown.append(f"{provider}:{model}")
            continue
        model_cost = 0.0
        for task in tasks:
            for condition in selected_conditions:
                prompt_record = selected_prompts[(task.task_id, condition)]
                prompt = str(prompt_record["prompt"])
                records: dict[int, dict[str, object]] = {}
                for attempt in range(1, attempt_limit + 1):
                    manifest = RunManifest.create(
                        taskset_id=taskset.taskset_id, task=task, condition=condition,
                        provider=provider, model_id=model, replicate=1,
                        temperature=float(config["temperature"]), top_p=config.get("top_p"),
                        max_tokens=token_cap, seed=None, protocol_version=str(config["protocol_version"]),
                        attempt=attempt, generation_route=route_for(provider, model),
                        reasoning_effort=reasoning_effort,
                        prompt_text=prompt,
                        documentation_snapshot_sha256=prompt_record["documentation_snapshot_sha256"],
                        docground_version=prompt_record["docground_version"],
                        prompt_decision=prompt_record["prompt_decision"],
                        docground_review_sha256=prompt_record["docground_review_sha256"],
                        evaluator_image=str(config.get("evaluator_image", "python:3.14-slim")),
                    )
                    existing = _find_existing_attempt(
                        result_root, attempt_index, manifest, prompt,
                        float(config["temperature"]), float(config.get("top_p", 1.0)),
                        token_cap, reasoning_effort,
                    )
                    if existing:
                        records[attempt] = existing[1]

                prompt_tokens = _tokens(prompt + "\nReturn only Python code in one fenced block.")
                unit_cost = token_cost(prompt_tokens, output_token_reserve(provider, token_cap), rates)
                for attempt in range(1, attempt_limit + 1):
                    details = records.get(attempt)
                    if details is not None:
                        if _retry_blocked(
                            details, allow_uncertain_billing=retry_recorded_errors,
                        ):
                            break
                        if details.get("generation_status") == "complete":
                            break
                        if _recorded_request_matches(
                            details, provider, model, prompt, float(config["temperature"]),
                            float(config.get("top_p", 1.0)), token_cap, reasoning_effort,
                        ):
                            if (
                                not retry_recorded_errors
                                or not _may_retry_record(details)
                                or attempt >= attempt_limit
                            ):
                                break
                            continue
                        if attempt < attempt_limit and _may_retry_error(
                            details, allow_uncertain_billing=retry_recorded_errors,
                        ):
                            continue
                        break

                    # A recorded-error recovery run makes at most one new provider
                    # request per logical task. The execution loop stops after that
                    # repair attempt, even though earlier attempt numbers may exist.
                    reserved_attempts = (
                        1 if retry_recorded_errors else attempt_limit - attempt + 1
                    )
                    model_cost += unit_cost * reserved_attempts
                    break
        planned[provider] = planned.get(provider, 0.0) + model_cost
    if unknown:
        raise RuntimeError("No conservative price mapping for candidates: " + ", ".join(unknown))
    accounting_root = Path(str(config.get("budget_accounting_root", result_root)))
    spent = _historical_spend(accounting_root)
    uncertain, unknown_uncertain = _uncertain_billing_reserve(accounting_root)
    if unknown_uncertain:
        raise RuntimeError("Cannot reserve unknown-billing attempts without rates: " + ", ".join(unknown_uncertain))
    projected = {
        provider: spent.get(provider, 0.0) + uncertain.get(provider, 0.0) + amount
        for provider, amount in planned.items()
    }
    over = {provider: total for provider, total in projected.items() if total > limit - reserve}
    if over:
        description = ", ".join(f"{provider} ${amount:.4f} > ${limit - reserve:.2f} available" for provider, amount in over.items())
        raise RuntimeError("Campaign budget gate stopped before the next API call: " + description)
    if emit:
        print("Budget reservation (conservative; historical usage plus missing calls):")
        for provider in sorted(projected):
            print(
                f"  {provider}: historical=${spent.get(provider, 0.0):.4f}, "
                f"unknown-billing reserve<=${uncertain.get(provider, 0.0):.4f}, "
                f"planned<=${planned[provider]:.4f}, projected<=${projected[provider]:.4f} of ${limit:.2f}"
            )
    return projected


def select_specs(
    models: list[dict[str, object]], provider_filter: str | None = None,
    model_filter: str | None = None,
) -> list[dict[str, object]]:
    selected = [
        spec for spec in models
        if (provider_filter is None or spec["provider"] == provider_filter)
        and (model_filter is None or spec["model_id"] == model_filter)
    ]
    if not selected:
        selectors = ", ".join(
            f"{label}={value}" for label, value in (
                ("provider", provider_filter), ("model_id", model_filter),
            ) if value is not None
        ) or "configured model list"
        raise RuntimeError(f"No campaign targets match {selectors}")
    return selected


def run(
    config_path: Path, result_root: Path = Path("results/raw"),
    provider_filter: str | None = None, model_filter: str | None = None,
    retry_recorded_errors: bool = False, task_id_filter: str | None = None,
    condition_filter: str | None = None,
) -> list[Path]:
    load_dotenv()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    taskset = TaskSet.from_path(Path(config["taskset_path"]))
    tasks, conditions = _configured_scope(
        taskset, config, task_id_filter=task_id_filter,
        condition_filter=condition_filter,
    )
    specs = select_specs(config["models"], provider_filter, model_filter)
    prompt_map = _prepare_prompt_map(taskset, tasks, config, conditions)
    max_attempts = int(config.get("max_attempts", 2))
    attempt_limit = _effective_attempt_limit(max_attempts, retry_recorded_errors)
    attempt_index = _attempt_index(result_root)
    _reserve_budget(
        config, taskset, specs, result_root, retry_recorded_errors,
        attempt_index, prompt_map, conditions, tasks,
    )
    timeout_seconds = int(config.get("timeout_seconds", 300))
    isolated = config.get("evaluation_mode") == "docker_isolated"
    evaluator_image = str(config.get("evaluator_image", "python:3.14-slim"))
    total = len(specs) * len(tasks) * len(conditions)
    logical_done = 0
    written: list[Path] = []
    for spec in specs:
        provider, model = str(spec["provider"]), str(spec["model_id"])
        rates = _configured_rates(spec)
        reasoning_effort = str(spec.get("reasoning_effort", config.get("reasoning_effort", "medium")))
        token_cap = int(spec.get("max_tokens", config["max_tokens"]))
        for task in tasks:
            for condition in conditions:
                prompt_record = prompt_map[(task.task_id, condition)]
                selected_prompt = str(prompt_record["prompt"])
                written_before = len(written)
                final_attempt_error: dict[str, object] | None = None
                for attempt in range(1, attempt_limit + 1):
                    manifest = RunManifest.create(
                        taskset_id=taskset.taskset_id, task=task, condition=condition,
                        provider=provider, model_id=model, replicate=1,
                        temperature=float(config["temperature"]), top_p=config.get("top_p"),
                        max_tokens=token_cap, seed=None,
                        protocol_version=str(config["protocol_version"]), attempt=attempt,
                        generation_route=route_for(provider, model),
                        reasoning_effort=reasoning_effort,
                        prompt_text=selected_prompt,
                        documentation_snapshot_sha256=prompt_record["documentation_snapshot_sha256"],
                        docground_version=prompt_record["docground_version"],
                        prompt_decision=prompt_record["prompt_decision"],
                        docground_review_sha256=prompt_record["docground_review_sha256"],
                        evaluator_image=evaluator_image,
                    )
                    destination = result_root / manifest.run_id
                    existing = _find_existing_attempt(
                        result_root, attempt_index, manifest, selected_prompt,
                        float(config["temperature"]), float(config.get("top_p", 1.0)),
                        token_cap, reasoning_effort,
                    )
                    if existing:
                        destination = existing[0]
                    if destination.exists():
                        details, _ = _read_details(destination)
                        if details and _retry_blocked(
                            details, allow_uncertain_billing=retry_recorded_errors,
                        ):
                            final_attempt_error = details
                            break
                        if _recorded_request_matches(
                            details, provider, model, selected_prompt,
                            float(config["temperature"]), float(config.get("top_p", 1.0)),
                            token_cap, reasoning_effort,
                        ):
                            final_attempt_error = None if details.get("generation_status") == "complete" else details
                            if (
                                details.get("generation_status") == "complete"
                                or not retry_recorded_errors
                                or not _may_retry_record(details)
                                or attempt >= attempt_limit
                            ):
                                break
                            continue
                        final_attempt_error = details
                        if attempt < attempt_limit and (
                            details.get("retryable") is True
                            or (
                                retry_recorded_errors
                                and not _retry_blocked(details, allow_uncertain_billing=True)
                            )
                        ):
                            continue
                        if attempt < attempt_limit:
                            for later_attempt in range(attempt + 1, attempt_limit + 1):
                                next_manifest = RunManifest.create(
                                    taskset_id=taskset.taskset_id, task=task, condition=condition,
                                    provider=provider, model_id=model, replicate=1,
                                    temperature=float(config["temperature"]), top_p=config.get("top_p"),
                                    max_tokens=token_cap, seed=None,
                                    protocol_version=str(config["protocol_version"]), attempt=later_attempt,
                                    generation_route=route_for(provider, model),
                                    reasoning_effort=reasoning_effort,
                                    prompt_text=selected_prompt,
                                    documentation_snapshot_sha256=prompt_record["documentation_snapshot_sha256"],
                                    docground_version=prompt_record["docground_version"],
                                    prompt_decision=prompt_record["prompt_decision"],
                                    docground_review_sha256=prompt_record["docground_review_sha256"],
                                    evaluator_image=evaluator_image,
                                )
                                next_details, _ = _read_details(result_root / next_manifest.run_id)
                                if _recorded_request_matches(
                                    next_details, provider, model, selected_prompt,
                                    float(config["temperature"]), float(config.get("top_p", 1.0)),
                                    token_cap, reasoning_effort,
                                ):
                                    final_attempt_error = (
                                        None if next_details.get("generation_status") == "complete"
                                        else next_details
                                    )
                                    break
                        break

                    if retry_recorded_errors and attempt > max_attempts:
                        previous_manifest = RunManifest.create(
                            taskset_id=taskset.taskset_id, task=task, condition=condition,
                            provider=provider, model_id=model, replicate=1,
                            temperature=float(config["temperature"]), top_p=config.get("top_p"),
                            max_tokens=token_cap, seed=None,
                            protocol_version=str(config["protocol_version"]), attempt=attempt - 1,
                            generation_route=route_for(provider, model),
                            reasoning_effort=reasoning_effort,
                            prompt_text=selected_prompt,
                            documentation_snapshot_sha256=prompt_record["documentation_snapshot_sha256"],
                            docground_version=prompt_record["docground_version"],
                            prompt_decision=prompt_record["prompt_decision"],
                            docground_review_sha256=prompt_record["docground_review_sha256"],
                            evaluator_image=evaluator_image,
                        )
                        previous = _find_existing_attempt(
                            result_root, attempt_index, previous_manifest,
                            selected_prompt, float(config["temperature"]),
                            float(config.get("top_p", 1.0)), token_cap, reasoning_effort,
                        )
                        previous_details = previous[1] if previous else {}
                        if previous_details.get("http_status") == 429:
                            min_delay = float(config.get("rate_limit_recovery_delay_seconds", 30))
                            retry_after = previous_details.get("retry_after_seconds")
                            delay = min(120.0, max(min_delay, float(retry_after or 0)))
                            print(f"[rate-limit cooldown] {provider}:{model} {delay:.0f}s", flush=True)
                            time.sleep(delay)

                    try:
                        response, metadata = generate(
                            provider, model, selected_prompt,
                            float(config["temperature"]), float(config.get("top_p", 1.0)),
                            token_cap, timeout_seconds, reasoning_effort,
                        )
                    except KeyboardInterrupt:
                        endpoint, _, route, request_params = request_parts(
                            provider, model, selected_prompt,
                            float(config["temperature"]), float(config.get("top_p", 1.0)),
                            token_cap, reasoning_effort,
                        )
                        written.append(record_interrupted_attempt(
                            result_root, manifest, task, route=route, endpoint=endpoint,
                            request_params=request_params,
                            catalog_metadata=spec.get("catalog_metadata"),
                            selected_prompt=selected_prompt,
                        ))
                        raise
                    metadata["attempt"] = attempt
                    metadata["generation_status"] = metadata.get("generation_status", "provider_error")
                    if response:
                        evaluation = evaluate_python_isolated(
                            task, response, image=evaluator_image,
                        ) if isolated else evaluate_static_python(task, response)
                        usage = metadata.get("usage", {})
                        actual_cost = None
                        if isinstance(usage, dict) and rates is not None:
                            in_tokens = metadata.get("input_tokens")
                            out_tokens = metadata.get("output_tokens")
                            ticks = usage.get("cost_in_usd_ticks")
                            if isinstance(ticks, int):
                                actual_cost = ticks / 10_000_000_000
                            elif isinstance(in_tokens, int) and isinstance(out_tokens, int):
                                actual_cost = token_cost(
                                    in_tokens, output_tokens_from_usage(provider, out_tokens, usage), rates,
                                )
                        metadata["estimated_cost_usd"] = actual_cost
                        metadata["cost_rate_usd_per_million"] = {"input": rates[0], "output": rates[1]} if rates else None
                        metadata["catalog_metadata"] = spec.get("catalog_metadata")
                        evaluation.details.update(metadata)
                        destination = write_run(
                            result_root, manifest, task, response, evaluation,
                            selected_prompt=selected_prompt,
                        )
                        written.append(destination)
                        _add_attempt_to_index(attempt_index, manifest, destination)
                        final_attempt_error = None if metadata.get("generation_status") == "complete" else metadata
                        break
                    evaluation = Evaluation("not_run", "not_run", "not_run", "not_run", dict(metadata))
                    metadata["catalog_metadata"] = spec.get("catalog_metadata")
                    evaluation.details.update(metadata)
                    destination = write_run(
                        result_root, manifest, task, "", evaluation,
                        selected_prompt=selected_prompt,
                    )
                    written.append(destination)
                    _add_attempt_to_index(attempt_index, manifest, destination)
                    final_attempt_error = metadata
                    if attempt >= attempt_limit or attempt > 1 or not metadata.get("retryable"):
                        break
                    retry_after = metadata.get("retry_after_seconds")
                    delay = min(
                        120,
                        max(1, math.ceil(float(retry_after)) if retry_after is not None else 2 * attempt),
                    )
                    time.sleep(delay)
                logical_done += 1
                print(f"[{logical_done}/{total}] {provider}:{model} {task.task_id} {condition.value} "
                      f"{'recorded' if final_attempt_error is not None else 'complete'}", flush=True)
                if len(written) > written_before and logical_done < total:
                    _reserve_budget(
                        config, taskset, specs, result_root, retry_recorded_errors,
                        attempt_index, prompt_map, conditions, tasks, emit=False,
                    )
    return written
