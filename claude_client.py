from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import subprocess
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Optional, Union

import config


TokenCallback = Callable[[str], Union[Awaitable[None], None]]


class ReplyPolicy(str, Enum):
    BACKCHANNEL = "backchannel"
    DIRECT = "direct"
    TASK = "task"
    CHAT = "chat"
    DEEP = "deep"


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
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        chunks: list[str] = []
        token_futures: list[concurrent.futures.Future] = []
        loop = _get_running_loop()

        for line in proc.stdout:
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
        prefix = REPLY_POLICY_PREFIXES[classify_reply_policy(user_text)]
        return prefix + user_text


REPLY_POLICY_PREFIXES = {
    ReplyPolicy.BACKCHANNEL: config.BACKCHANNEL_REPLY_PREFIX,
    ReplyPolicy.DIRECT: config.DIRECT_REPLY_PREFIX,
    ReplyPolicy.TASK: config.TASK_REPLY_PREFIX,
    ReplyPolicy.CHAT: config.CHAT_REPLY_PREFIX,
    ReplyPolicy.DEEP: config.DEEP_THINK_PREFIX,
}


BACKCHANNEL_WORDS = frozenset({
    "嗯", "哦", "好", "行", "对", "是", "懂", "知道了", "好的", "行吧",
    "是吗", "真的吗", "假的吧", "不是吧", "天哪", "卧槽", "厉害",
    "ok", "嗯嗯", "哦哦", "呵呵", "哈哈", "嘿嘿",
    "谢了", "谢谢", "辛苦了", "拜拜", "再见", "晚安", "早",
    "yes", "no", "sure", "thanks", "thankyou",
})


TASK_MARKERS = (
    "帮我", "给我", "替我", "为我", "帮忙", "查一下", "搜一下", "找一下",
    "搜索", "查找", "打开", "启动", "关闭", "重启", "新建", "创建",
    "总结", "整理", "写一", "写个", "写一个", "改一下", "修改", "修复", "生成", "翻译",
    "review", "运行", "执行", "测试", "部署", "安装",
)


DIRECT_QUESTION_MARKERS = (
    "？", "?", "谁", "哪", "几", "怎", "咋", "啥", "什么", "吗", "呢",
    "如何", "天气", "时间", "几点", "多久", "多少", "哪里", "哪个",
    "要不要", "能不能", "可以不", "可不可以", "为什么", "怎么办",
    "是不是", "有没有", "该不该", "讲讲", "解释", "介绍",
)
NORMALIZE_PUNCTUATION_RE = re.compile(r"[，。！？、…—\-！,.?~～\s]")


def classify_reply_policy(user_text: str) -> ReplyPolicy:
    stripped = user_text.strip()
    cleaned = _normalize_input(stripped)
    if not cleaned:
        return ReplyPolicy.BACKCHANNEL
    if _needs_deep_think(cleaned):
        return ReplyPolicy.DEEP
    if cleaned in BACKCHANNEL_WORDS:
        return ReplyPolicy.BACKCHANNEL
    if any(marker in cleaned for marker in TASK_MARKERS):
        return ReplyPolicy.TASK
    if any(marker in cleaned for marker in DIRECT_QUESTION_MARKERS):
        return ReplyPolicy.DIRECT
    return ReplyPolicy.CHAT


def _normalize_input(user_text: str) -> str:
    return NORMALIZE_PUNCTUATION_RE.sub("", user_text).lower()


def _needs_deep_think(user_text: str) -> bool:
    normalized = _normalize_input(user_text)
    return any(_normalize_input(trigger) in normalized for trigger in config.DEEP_THINK_TRIGGERS)


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
