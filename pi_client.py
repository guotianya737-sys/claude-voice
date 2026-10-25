"""pi 子进程客户端 —— 通过 `pi --mode rpc` 的 JSONL 协议对话。

协议要点（见 pi 的 docs/rpc.md）：
  stdin  : {"id":..., "type":"prompt", "message":"..."} / {"type":"abort"} / {"type":"new_session"} ...
  stdout : response 记录 + 会话事件流（message_update.assistantMessageEvent.text_delta 为增量文本，
           agent_settled 表示这一轮不会再自动继续）

与 claude-voice 原始实现的差异：
  - 用二进制管道按 LF 切行，避免 Unicode 行分隔符（U+2028/U+2029）被误判为记录边界；
  - 常驻读取线程持续消费 stdout（协议要求），事件经 asyncio 队列投递给会话层；
  - 工具调用通过 on_event 上报，前端可以显示"正在跑 bash"这类状态。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import subprocess
import threading
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Optional, Union

import config


TokenCallback = Callable[[str], Union[Awaitable[None], None]]
EventCallback = Callable[[str, dict], Union[Awaitable[None], None]]


class ReplyPolicy(str, Enum):
    BACKCHANNEL = "backchannel"
    DIRECT = "direct"
    TASK = "task"
    CHAT = "chat"
    DEEP = "deep"


class PiClient:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._counter = 0
        self._stderr_tail: list[str] = []
        self._loop = asyncio.get_running_loop()
        self._start()

    # -- 生命周期 -----------------------------------------------------------
    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _command(self) -> list[str]:
        command = [
            config.PI_NODE,
            config.PI_LAUNCHER,
            "--mode",
            "rpc",
            "--session-dir",
            config.PI_SESSION_DIR,
            "--append-system-prompt",
            config.VOICE_APPEND_SYSTEM_PROMPT,
        ]
        if config.PI_THINKING:
            command += ["--thinking", config.PI_THINKING]
        if config.PI_MODEL:
            command += ["--model", config.PI_MODEL]
        command += list(config.PI_EXTRA_ARGS)
        return command

    def _start(self) -> None:
        self.close()
        self._inbox = asyncio.Queue()
        self._stderr_tail = []
        self.proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(config.BASE_DIR),
        )
        threading.Thread(target=self._read_loop, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._stderr_loop, args=(self.proc,), daemon=True).start()

    def close(self) -> None:
        proc = self.proc
        if proc is None:
            return
        self.proc = None
        with contextlib.suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=3)

    async def reset(self) -> None:
        """开新会话（保留常驻进程，省掉重启开销）。"""
        async with self._lock:
            if not self.is_alive():
                self._start()
            self._drain_inbox()
            request_id = self._next_id()
            self._write({"id": request_id, "type": "new_session"})
            deadline = self._loop.time() + 10.0
            try:
                while True:
                    remaining = deadline - self._loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    record = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
                    kind = record.get("type")
                    if kind == "__eof__":
                        self._start()
                        return
                    if kind == "response" and record.get("id") == request_id:
                        if record.get("success") is False:
                            raise RuntimeError(f"new_session 失败: {record.get('error', '')}")
                        break
            except (asyncio.TimeoutError, RuntimeError) as exc:
                # 协议层面没应⿵也不影响使用：重启子进程拿一个干净会话
                print(f"pi new_session 未确认（{type(exc).__name__}），重启进程")
                self.close()
                self._start()
                return
            self._drain_inbox()

    async def abort(self) -> None:
        if self.is_alive():
            with contextlib.suppress(Exception):
                self._write({"id": self._next_id(), "type": "abort"})

    # -- 主流程 -------------------------------------------------------------
    async def send(
        self,
        user_text: str,
        on_token: Optional[TokenCallback] = None,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        async with self._lock:
            if not self.is_alive():
                self._start()
            message = REPLY_POLICY_PREFIXES[classify_reply_policy(user_text)] + user_text
            self._drain_inbox()
            self._write({"id": self._next_id(), "type": "prompt", "message": message})
            return await self._await_settled(on_token=on_token, on_event=on_event)

    async def _await_settled(
        self,
        timeout: float = config.PI_RESPONSE_TIMEOUT,
        on_token: Optional[TokenCallback] = None,
        on_event: Optional[EventCallback] = None,
    ) -> str:
        chunks: list[str] = []
        futures: list[asyncio.Future] = []
        deadline = self._loop.time() + timeout

        def dispatch(result):
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                futures.append(asyncio.ensure_future(result))

        while True:
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError("pi response timeout")
            try:
                record = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("pi response timeout") from None

            kind = record.get("type")
            if kind == "__eof__":
                self.close()
                detail = "".join(self._stderr_tail[-5:]).strip()
                raise RuntimeError(f"pi 进程已退出 {detail}".strip())

            if kind == "response":
                if record.get("success") is False and record.get("command") == "prompt":
                    raise RuntimeError(f"pi 拒绝了这次输入: {record.get('error', '')}")
                continue

            if kind == "message_update":
                event = record.get("assistantMessageEvent") or {}
                if event.get("type") == "text_delta":
                    delta = event.get("delta") or ""
                    if delta:
                        chunks.append(delta)
                        if on_token is not None:
                            dispatch(on_token(delta))
                continue

            if kind in {"tool_execution_start", "tool_execution_end"}:
                if on_event is not None:
                    dispatch(on_event(kind, record))
                continue

            if kind == "agent_settled":
                if futures:
                    await asyncio.gather(*futures, return_exceptions=True)
                return "".join(chunks).strip()

            if kind == "error":
                message = record.get("error") or record.get("message") or "unknown error"
                raise RuntimeError(f"pi 报错: {message}")

    # -- 管道细节 -----------------------------------------------------------
    def _next_id(self) -> str:
        self._counter += 1
        return f"v{self._counter}"

    def _write(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("pi 进程未启动")
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        proc.stdin.write(data)
        proc.stdin.flush()

    def _read_loop(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip(b"\r\n")
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._inbox.put_nowait, record)
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._inbox.put_nowait, {"type": "__eof__"})

    def _stderr_loop(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                self._stderr_tail.append(text)
                del self._stderr_tail[:-20]

    def _drain_inbox(self) -> None:
        while True:
            try:
                self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                return


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
    cleaned = _normalize_input(user_text.strip())
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
