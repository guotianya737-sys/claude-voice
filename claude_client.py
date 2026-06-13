from __future__ import annotations

import asyncio
import concurrent.futures
import json
import subprocess
from collections.abc import Awaitable, Callable
from typing import Optional, Union

import config


TokenCallback = Callable[[str], Union[Awaitable[None], None]]


class ClaudeClient:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen[str]] = None
        self._lock = asyncio.Lock()
        self._start()

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def send(self, user_text: str, on_token: Optional[TokenCallback] = None) -> str:
        async with self._lock:
            if not self.is_alive():
                self._start()
            assert self.proc is not None
            payload = {
                "type": "user",
                "message": {"role": "user", "content": self._prepare_user_text(user_text)},
            }
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            reply, token_futures = await asyncio.to_thread(self._read_response, on_token)
            if token_futures:
                await asyncio.gather(*(asyncio.wrap_future(future) for future in token_futures))
            return reply

    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _start(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            [config.CLAUDE_BIN, *config.CLAUDE_FLAGS],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _read_response(
        self,
        on_token: Optional[TokenCallback],
    ) -> tuple[str, list[concurrent.futures.Future]]:
        assert self.proc is not None and self.proc.stdout is not None
        chunks: list[str] = []
        token_futures: list[concurrent.futures.Future] = []
        loop = _get_running_loop()

        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            for text in self._extract_text(event):
                chunks.append(text)
                if on_token is not None:
                    result = on_token(text)
                    if asyncio.iscoroutine(result):
                        token_futures.append(asyncio.run_coroutine_threadsafe(result, loop))

            if self._is_done_event(event):
                break

        return "".join(chunks).strip(), token_futures

    @staticmethod
    def _extract_text(event: dict) -> list[str]:
        if event.get("type") != "assistant":
            return []
        message = event.get("message", event)
        content = message.get("content", [])
        if isinstance(content, str):
            return [content]
        texts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
        return texts

    @staticmethod
    def _is_done_event(event: dict) -> bool:
        event_type = event.get("type")
        if event_type in {"result", "done", "error"}:
            return True
        if event_type == "assistant":
            message = event.get("message", event)
            return bool(message.get("stop_reason") or message.get("stop_sequence"))
        return False

    @staticmethod
    def _prepare_user_text(user_text: str) -> str:
        prefix = config.DEEP_THINK_PREFIX if _needs_deep_think(user_text) else config.FAST_REPLY_PREFIX
        return prefix + user_text


def _needs_deep_think(user_text: str) -> bool:
    normalized = user_text.replace(" ", "")
    return any(trigger in normalized for trigger in config.DEEP_THINK_TRIGGERS)


def _get_running_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # Called inside asyncio.to_thread; recover the main loop captured by MainLoopHolder.
        return MainLoopHolder.loop


class MainLoopHolder:
    loop: asyncio.AbstractEventLoop


def capture_loop() -> None:
    MainLoopHolder.loop = asyncio.get_running_loop()
