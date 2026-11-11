import sys
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

LAB2_DIR = Path(__file__).resolve().parent.parent
if str(LAB2_DIR) not in sys.path:
    sys.path.insert(0, str(LAB2_DIR))

import cli


class FakeCompletions:
    def __init__(self, content="Local response", choices=True):
        self.content = content
        self.choices = choices
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.choices:
            return SimpleNamespace(choices=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def fake_sdk(model_ids=(cli.DEFAULT_MODEL,), content="Local response", choices=True):
    completions = FakeCompletions(content, choices)
    sdk = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model_id) for model_id in model_ids]
            )
        ),
        chat=SimpleNamespace(completions=completions),
    )
    return sdk, completions


def test_default_config():
    config = cli.parse_config([])
    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.model == cli.DEFAULT_MODEL
    assert config.temperature == 0.7
    assert config.max_tokens == 200


def test_parser_accepts_and_normalizes_all_options():
    config = cli.parse_config(
        [
            "--base-url", "http://localhost:9000/v1/",
            "--model", " custom/model ",
            "--system-prompt", " Be exact. ",
            "--temperature", "1.2",
            "--max-tokens", "30",
        ]
    )
    assert config == cli.AppConfig(
        base_url="http://localhost:9000/v1",
        model="custom/model",
        system_prompt="Be exact.",
        temperature=1.2,
        max_tokens=30,
    )


@pytest.mark.parametrize("temperature", [-0.1, 2.1, True, "0.7"])
def test_temperature_validation(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        cli.AppConfig(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_max_tokens_validation(max_tokens):
    with pytest.raises(ValueError, match="Max tokens"):
        cli.AppConfig(max_tokens=max_tokens)


@pytest.mark.parametrize("field", ["base_url", "model"])
def test_required_config_strings(field):
    with pytest.raises(ValueError):
        cli.AppConfig(**{field: " "})


def test_client_verifies_exact_model():
    config = cli.AppConfig()
    sdk, _ = fake_sdk()
    assert cli.VLLMClient(config, sdk).verify_model() == [cli.DEFAULT_MODEL]


def test_client_reports_available_models():
    config = cli.AppConfig(model="missing/model")
    sdk, _ = fake_sdk(("loaded/model",))
    with pytest.raises(ValueError, match="loaded/model"):
        cli.VLLMClient(config, sdk).verify_model()


def test_completion_contains_history_and_config():
    config = cli.AppConfig(temperature=0.2, max_tokens=80)
    sdk, completions = fake_sdk()
    client = cli.VLLMClient(config, sdk)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Question"},
    ]
    assert client.complete(messages) == "Local response"
    assert completions.calls == [{
        "model": cli.DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 80,
    }]


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_completion_is_rejected(content):
    config = cli.AppConfig()
    sdk, _ = fake_sdk(content=content)
    with pytest.raises(RuntimeError, match="empty assistant response"):
        cli.VLLMClient(config, sdk).complete([])


def test_missing_completion_choices_are_rejected():
    config = cli.AppConfig()
    sdk, _ = fake_sdk(choices=False)
    with pytest.raises(RuntimeError, match="no completion choices"):
        cli.VLLMClient(config, sdk).complete([])


def test_session_commits_only_successful_turns():
    config = cli.AppConfig(system_prompt="System")
    sdk, _ = fake_sdk(content="Answer")
    session = cli.ChatSession(config, cli.VLLMClient(config, sdk))
    assert session.ask(" Question ") == "Answer"
    assert session.history == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_session_keeps_history_when_request_fails():
    class BrokenClient:
        def complete(self, messages):
            raise RuntimeError("failed")

    config = cli.AppConfig(system_prompt="System")
    session = cli.ChatSession(config, BrokenClient())
    with pytest.raises(RuntimeError, match="failed"):
        session.ask("Question")
    assert session.history == [{"role": "system", "content": "System"}]


def test_transcript_is_utf8_and_contains_config(tmp_path):
    config = cli.AppConfig(system_prompt="Відповідай стисло.", temperature=0.2)
    sdk, _ = fake_sdk(content="Це відповідь.")
    session = cli.ChatSession(config, cli.VLLMClient(config, sdk))
    session.ask("Що таке LLM?")
    path = session.save(
        tmp_path,
        recorded_at=datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),
    )
    assert path is not None
    content = path.read_bytes().decode("utf-8")
    assert "Що таке LLM?" in content
    assert "Temperature: 0.2" in content


def test_empty_session_writes_nothing(tmp_path):
    config = cli.AppConfig()
    sdk, _ = fake_sdk()
    session = cli.ChatSession(config, cli.VLLMClient(config, sdk))
    assert session.save(tmp_path) is None


def test_console_handles_unavailable_server(capsys):
    class OfflineClient:
        def verify_model(self):
            raise cli.APIConnectionError(request=SimpleNamespace())

    config = cli.AppConfig()
    app = cli.ConsoleApp(config, OfflineClient())
    assert app.run() == 1
    assert "Cannot connect" in capsys.readouterr().out


def test_console_exit_does_not_call_completion(monkeypatch):
    config = cli.AppConfig()
    sdk, completions = fake_sdk()
    app = cli.ConsoleApp(config, cli.VLLMClient(config, sdk), read=lambda _: "exit")
    monkeypatch.setattr(app.session, "save", lambda: None)
    assert app.run() == 0
    assert completions.calls == []
