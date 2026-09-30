from experiments.providers import PROVIDERS, load_dotenv


def test_active_five_provider_slots_are_declared():
    assert {provider.environment_key for provider in PROVIDERS} == {
        "OPENAI_API_KEY", "GLM_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY",
    }


def test_dotenv_loader_does_not_replace_existing_value(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("SAMPLE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SAMPLE_API_KEY", "already-set")
    load_dotenv(dotenv)
    assert __import__("os").environ["SAMPLE_API_KEY"] == "already-set"
