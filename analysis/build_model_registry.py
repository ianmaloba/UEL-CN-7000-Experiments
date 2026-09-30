"""Classify every active provider catalog entry for code-generation eligibility."""
from __future__ import annotations
import json
import re
from datetime import UTC, datetime
from pathlib import Path

CATALOG = Path("artefacts/generated/provider-preflight-2026-09-28.json")
OUTPUT = Path("configs/model_registry_v1.json")
NON_TEXT = re.compile(r"image|vision|audio|speech|transcrib|whisper|realtime|\blive\b|sora|video|embed|moderation|ocr|asr|tts|(?:^|-)v(?:-|$)", re.I)
SEARCH = re.compile(r"search", re.I)
COMPLETION_ONLY = re.compile(r"^(babbage-002|davinci-002|gpt-3\.5-turbo-instruct(?:-.+)?)$")
OPENAI_SHUTDOWN_IDS = {
    "gpt-5-codex": "2026-07-23",
    "gpt-5.1-codex": "2026-07-23",
    "gpt-5.1-codex-max": "2026-07-23",
    "gpt-5.1-codex-mini": "2026-07-23",
    "gpt-5.2-codex": "2026-07-23",
    "gpt-5-chat-latest": "2026-07-23",
    "gpt-5.1-chat-latest": "2026-07-23",
    "gpt-5.2-chat-latest": "2026-08-10",
    "gpt-5.3-chat-latest": "2026-08-10",
}

def classify(provider: str, entry: dict) -> tuple[str, str]:
    model_id = str(entry.get("id", entry.get("model", entry.get("name", "unknown"))))
    if provider == "mistral":
        capabilities = entry.get("capabilities", {})
        if model_id.lower().startswith("labs-"):
            return "exclude", "Mistral API returned labs_not_enabled; organization admin enablement is required"
        if entry.get("archived") is True:
            return "exclude", "provider marks model archived"
        if capabilities.get("completion_chat") is True:
            return "include", "provider explicitly marks chat completion support"
        if capabilities.get("completion_fim") is True:
            return "review", "fill-in-middle model; incompatible with current task prompt protocol"
        return "exclude", "provider does not mark chat completion capability"
    if provider == "xai":
        if "text" in entry.get("input_modalities", []) and "text" in entry.get("output_modalities", []):
            return "include", "provider lists text input and output"
        return "exclude", "provider does not list text-to-text modalities"
    if provider == "deepseek":
        return "include", "official provider catalog contains text chat models"
    if provider == "glm":
        if NON_TEXT.search(model_id):
            return "exclude", "specialized non-code modality inferred from model ID"
        return "include", "GLM text-generation model; verify endpoint support during live smoke"
    if provider == "openai":
        if model_id in OPENAI_SHUTDOWN_IDS:
            return "exclude", f"official API shutdown on {OPENAI_SHUTDOWN_IDS[model_id]}"
        if NON_TEXT.search(model_id):
            return "exclude", "specialized non-code modality inferred from model ID"
        if SEARCH.search(model_id):
            return "separate", "search-specialized model; isolate from the no-tools primary code-generation cohort"
        if COMPLETION_ONLY.match(model_id):
            return "include_completion", "text completion model; use legacy completions endpoint"
        return "include", "text-generation model; verify endpoint support during live smoke"
    return "review", "provider is not active in this protocol"

def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    records = []
    for provider in catalog["providers"]:
        if provider["provider"] not in {"openai", "glm", "deepseek", "xai", "mistral"}:
            continue
        for entry in provider.get("model_entries", []):
            disposition, reason = classify(provider["provider"], entry)
            model_id = str(entry.get("id", entry.get("model", entry.get("name", "unknown"))))
            if model_id == "unknown":
                continue
            records.append({"provider": provider["provider"], "model_id": model_id, "canonical_id": model_id,
                            "variant_kind": "catalog_entry", "disposition": disposition,
                            "reason": reason, "catalog_metadata": entry})
            aliases = entry.get("aliases", [])
            if disposition in {"include", "include_completion", "separate"} and isinstance(aliases, list):
                for alias in aliases:
                    if not isinstance(alias, str) or alias == model_id:
                        continue
                    records.append({"provider": provider["provider"], "model_id": alias, "canonical_id": model_id,
                                    "variant_kind": "alias", "disposition": disposition,
                                    "reason": f"provider alias for {model_id}", "catalog_metadata": entry})
    seen = set()
    deduped = []
    for record in records:
        key = (record["provider"], record["model_id"])
        if key in seen:
            continue
        seen.add(key); deduped.append(record)
    eligible = [r for r in deduped if r["disposition"].startswith("include")]
    separate = [r for r in deduped if r["disposition"] == "separate"]
    report = {"protocol_version": "1.0", "generated_at": datetime.now(UTC).isoformat(),
              "catalogue_snapshot": str(CATALOG), "selection_rule": "include every account-listed text-to-text/chat model, every provider-listed alias, and text-completion-only OpenAI base models; preserve search-specialized models as a separate no-tools exploratory cohort; exclude official API shutdowns, audio, image, video, embedding, moderation, OCR/ASR/TTS specializations, archived, and non-chat models",
              "active_providers": ["openai", "glm", "deepseek", "xai", "mistral"],
              "model_count": len(deduped), "eligible_target_count": len(eligible),
              "separate_exploratory_target_count": len(separate), "models": deduped}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for provider_name in report["active_providers"]:
        p = [r for r in deduped if r["provider"] == provider_name]
        print(f"{provider_name}: {sum(r['disposition'].startswith('include') for r in p)} generation targets from {len(p)} catalog IDs/aliases")
    print(f"total eligible targets: {len(eligible)} / {len(deduped)} inventoried IDs")

if __name__ == "__main__":
    main()
