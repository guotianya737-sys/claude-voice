"""全局按住说话热键（Windows，纯 ctypes，不依赖第三方包）。

用 GetAsyncKeyState 轮询而不是键盘钩子：不需要管理员权限，不会触发杀软，
按下 / 松开两个边沿都可以拿到，正好对上"按住说话"。
"""

from __future__ import annotations

import asyncio
import ctypes
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Optional


_VK = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "PAUSE": 0x13,
    "ESC": 0x1B,
    "SPACE": 0x20,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "END": 0x23,
    "HOME": 0x24,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
    "LSHIFT": 0xA0,
    "RSHIFT": 0xA1,
    "LCTRL": 0xA2,
    "RCTRL": 0xA3,
    "LALT": 0xA4,
    "RALT": 0xA5,
}
_VK.update({f"F{index}": 0x70 + index - 1 for index in range(1, 25)})

VK_NAME = {value: name for name, value in _VK.items()}


def resolve_vk(name: str) -> Optional[int]:
    key = name.strip().upper().replace("-", "")
    if key in _VK:
        return _VK[key]
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        return ord(key)
    return None


class HoldHotkey:
    """名字 → 按下/松开回调的轮询式热键。"""

    def __init__(
        self,
        key: str,
        on_press: Callable[[], Awaitable[None]],
        on_release: Callable[[], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
        poll_interval: float = 0.02,
    ) -> None:
        self.key = key
        self.vk = resolve_vk(key)
        self.on_press = on_press
        self.on_release = on_release
        self.loop = loop
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._down = False

    @property
    def available(self) -> bool:
        return self.vk is not None

    def describe(self) -> str:
        return self.key.upper() if self.vk is not None else f"未识别({self.key})"

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def _run(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        while not self._stop.is_set():
            pressed = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
            if pressed and not self._down:
                self._down = True
                self._dispatch(self.on_press)
            elif not pressed and self._down:
                self._down = False
                self._dispatch(self.on_release)
            time.sleep(self.poll_interval)

    def _dispatch(self, callback: Callable[[], Awaitable[None]]) -> None:
        try:
            asyncio.run_coroutine_threadsafe(callback(), self.loop)
        except RuntimeError:
            pass
