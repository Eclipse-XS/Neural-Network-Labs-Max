from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and concise assistant."
DEFAULT_DIALOGS_DIR = Path(__file__).resolve().parent / "outputs" / "dialogs"
EXIT_COMMANDS = {"exit", "quit"}
Message = dict[str, str]


@dataclass(frozen=True)
class AppConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 200

    def __post_init__(self) -> None:
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        system_prompt = self.system_prompt.strip()
        if not base_url:
            raise ValueError("Base URL must not be empty.")
        if not model:
            raise ValueError("Model id must not be empty.")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("Temperature must be a number between 0 and 2.")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("Max tokens must be a positive integer.")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "system_prompt", system_prompt)


class VLLMClient:
    def __init__(self, config: AppConfig, sdk: Any | None = None) -> None:
        self.config = config
        self._sdk = sdk or OpenAI(base_url=config.base_url, api_key="vllm-local")

    def verify_model(self) -> list[str]:
        response = self._sdk.models.list()
        model_ids = [item.id for item in response.data if getattr(item, "id", None)]
        if self.config.model not in model_ids:
            available = ", ".join(model_ids) if model_ids else "none"
            raise ValueError(
                f"Model '{self.config.model}' is not available in vLLM. "
                f"Available model ids: {available}."
            )
        return model_ids

    def complete(self, messages: Sequence[Message]) -> str:
        response = self._sdk.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if not response.choices:
            raise RuntimeError("vLLM returned no completion choices.")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise RuntimeError("vLLM returned an empty assistant response.")
        return content.strip()


class ChatSession:
    def __init__(self, config: AppConfig, client: VLLMClient) -> None:
        self.config = config
        self.client = client
        self.history: list[Message] = []
        if config.system_prompt:
            self.history.append({"role": "system", "content": config.system_prompt})

    @property
    def has_dialogue(self) -> bool:
        return any(message["role"] == "assistant" for message in self.history)

    def ask(self, prompt: str) -> str:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt must not be empty.")
        request_messages = [
            *self.history,
            {"role": "user", "content": cleaned_prompt},
        ]
        answer = self.client.complete(request_messages)
        self.history.extend(
            [
                {"role": "user", "content": cleaned_prompt},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    def save(
        self,
        output_dir: Path = DEFAULT_DIALOGS_DIR,
        *,
        recorded_at: datetime | None = None,
    ) -> Path | None:
        if not self.has_dialogue:
            return None
        timestamp = recorded_at or datetime.now().astimezone()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"dialog_{timestamp:%Y%m%d_%H%M%S_%f}.md"
        lines = [
            "# Dialogue",
            "",
            "## Configuration",
            f"- Timestamp: {timestamp.isoformat(timespec='seconds')}",
            f"- Base URL: {self.config.base_url}",
            f"- Model: {self.config.model}",
            f"- Temperature: {self.config.temperature}",
            f"- Max tokens: {self.config.max_tokens}",
            f"- System prompt: {self.config.system_prompt or '(none)'}",
        ]
        for message in self.history:
            if message["role"] not in {"user", "assistant"}:
                continue
            heading = "User" if message["role"] == "user" else "Assistant"
            lines.extend(["", f"## {heading}", "", message["content"]])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class ConsoleApp:
    def __init__(
        self,
        config: AppConfig,
        client: VLLMClient | None = None,
        *,
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.client = client or VLLMClient(config)
        self.session = ChatSession(config, self.client)
        self.read = read
        self.write = write

    def run(self) -> int:
        if not self._check_server():
            return 1
        self.write(f"vLLM API: {self.config.base_url}")
        self.write(f"Model: {self.config.model}")
        self.write(f"Temperature: {self.config.temperature}")
        self.write(f"Max tokens: {self.config.max_tokens}")
        self.write("Type 'exit' or 'quit' to finish.")
        try:
            self._chat_loop()
        except KeyboardInterrupt:
            self.write("\nSession interrupted.")
        finally:
            transcript = self.session.save()
            if transcript:
                self.write(f"Transcript saved to: {transcript}")
        return 0

    def _check_server(self) -> bool:
        try:
            self.client.verify_model()
            return True
        except APIConnectionError:
            self.write(
                f"Cannot connect to vLLM at {self.config.base_url}. "
                "Check that the server is running."
            )
        except APIError as error:
            self.write(f"vLLM rejected the model check: {error}")
        except ValueError as error:
            self.write(str(error))
        return False

    def _chat_loop(self) -> None:
        while True:
            try:
                prompt = self.read("\nYou: ").strip()
            except EOFError:
                self.write("")
                return
            if prompt.lower() in EXIT_COMMANDS:
                return
            if not prompt:
                continue
            try:
                answer = self.session.ask(prompt)
            except APIConnectionError:
                self.write(f"Cannot connect to vLLM at {self.config.base_url}.")
                continue
            except APIError as error:
                self.write(f"vLLM did not complete the request: {error}")
                continue
            except RuntimeError as error:
                self.write(str(error))
                continue
            self.write(f"\nAssistant: {answer}")


def parse_config(argv: Sequence[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Interactive OpenAI-compatible client for a local vLLM server."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        return AppConfig(**vars(args))
    except ValueError as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    return ConsoleApp(parse_config(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
