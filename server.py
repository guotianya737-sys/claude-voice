from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Optional

from aiohttp import WSMsgType, web

import config


CommandHandler = Callable[[dict], Awaitable[None]]
StateProvider = Callable[[], list[dict]]


class VoiceServer:
    def __init__(
        self,
        command_handler: CommandHandler,
        state_provider: Optional[StateProvider] = None,
        static_dir: Optional[Path] = None,
    ) -> None:
        self.command_handler = command_handler
        self.state_provider = state_provider
        self.static_dir = static_dir or (config.BASE_DIR / "static")
        self.app = web.Application()
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/ws", self._ws)
        self.websockets: set[web.WebSocketResponse] = set()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

    async def start(self, host: str, port: int) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()

    async def stop(self) -> None:
        for ws in list(self.websockets):
            await ws.close()
        if self.runner is not None:
            await self.runner.cleanup()

    async def broadcast(self, payload: dict) -> None:
        dead: list[web.WebSocketResponse] = []
        data = json.dumps(payload, ensure_ascii=False)
        for ws in self.websockets:
            if ws.closed:
                dead.append(ws)
                continue
            await ws.send_str(data)
        for ws in dead:
            self.websockets.discard(ws)

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.static_dir / "index.html")

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.websockets.add(ws)
        if self.state_provider is not None:
            for payload in self.state_provider():
                await ws.send_str(json.dumps(payload, ensure_ascii=False))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    await self.command_handler(payload)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.websockets.discard(ws)
        return ws
