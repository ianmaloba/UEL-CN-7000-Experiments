"""Credential-safe, zero-generation provider discovery used before a live campaign."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Provider:
    name: str
    environment_key: str
    models_url: str


PROVIDERS = (
    Provider("openai", "OPENAI_API_KEY", "https://api.openai.com/v1/models"),
    Provider("glm", "GLM_API_KEY", "https://open.bigmodel.cn/api/paas/v4/models"),
    Provider("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/models"),
    Provider("xai", "XAI_API_KEY", "https://api.x.ai/v1/language-models"),
    Provider("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1/models"),
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without printing or serialising secret values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def discover(provider: Provider, timeout_seconds: int = 20) -> dict[str, object]:
    """List accessible model metadata. It sends no generation request or prompt."""
    token = os.environ.get(provider.environment_key)
    if not token:
        return {"provider": provider.name, "status": "not_configured", "models": []}
    request = Request(provider.models_url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
            records = payload.get("data", payload.get("models", []))
            safe_fields = {
                "id", "model", "name", "object", "created", "owned_by", "root", "root_version",
                "aliases", "capabilities", "max_context_length", "context_length", "pricing",
                "prompt_text_token_price", "cached_prompt_text_token_price", "completion_text_token_price",
                "input_modalities", "output_modalities", "version", "fingerprint", "archived", "type",
            }
            entries = [
                {key: value for key, value in entry.items() if key in safe_fields}
                for entry in records if isinstance(entry, dict)
            ]
            models = sorted(str(entry.get("id", entry.get("model", entry.get("name", "unknown")))) for entry in entries)
            return {"provider": provider.name, "status": "available", "http_status": response.status, "models": models, "model_entries": entries}
    except HTTPError as error:
        return {"provider": provider.name, "status": "http_error", "http_status": error.code, "models": []}
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"provider": provider.name, "status": "connection_error", "error_type": type(error).__name__, "models": []}


def preflight(output: Path) -> dict[str, object]:
    load_dotenv()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "zero-generation model discovery; no prompts sent",
        "providers": [discover(provider) for provider in PROVIDERS],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
