from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

import config


@dataclass
class PreparedSpeech:
    provider: str
    text: str
    audio_path: Optional[Path] = None


class TTSEngine:
    def __init__(self, provider: str = config.TTS_PROVIDER) -> None:
        self.provider = provider
        self.say = SayTTS()
        self.siliconflow = SiliconFlowTTS.from_config()

    async def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self.provider == "siliconflow":
            ok = await self.siliconflow.speak(text)
            if ok:
                return
        await self.say.speak(text)

    def can_prefetch(self) -> bool:
        return self.provider == "siliconflow" and self.siliconflow.configured

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = text.strip()
        if not text:
            return None
        if self.provider == "siliconflow":
            prepared = await self.siliconflow.prepare(text)
            if prepared is not None:
                return prepared
        return PreparedSpeech(provider="say", text=text)

    async def play_prepared(self, prepared: PreparedSpeech) -> None:
        if prepared.provider == "siliconflow" and prepared.audio_path is not None:
            ok = await self.siliconflow.play_prepared(prepared)
            if ok:
                return
        await self.say.speak(prepared.text)

    def discard_prepared(self, prepared: PreparedSpeech) -> None:
        if prepared.audio_path is not None:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def stop(self) -> None:
        self.say.stop()
        self.siliconflow.stop()

    def set_provider(self, provider: str) -> None:
        if provider not in {"say", "siliconflow"}:
            return
        self.stop()
        self.provider = provider


class SayTTS:
    def __init__(self, voice: str = config.SAY_VOICE, rate: int = config.SAY_RATE) -> None:
        self.voice = voice
        self.rate = rate
        self._proc: Optional[subprocess.Popen[str]] = None

    async def speak(self, text: str) -> bool:
        await asyncio.to_thread(self.stop)
        await asyncio.to_thread(self._run_say, text)
        return True

    def stop(self) -> None:
        _stop_proc(self._proc)
        self._proc = None

    def _run_say(self, text: str) -> None:
        self._proc = subprocess.Popen(["say", "-v", self.voice, "-r", str(self.rate), text])
        self._proc.wait()
        self._proc = None


class SiliconFlowTTS:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        default_model: str,
        default_voice: str,
        sample_rate: int,
        response_format: str,
        stream: bool,
        speed: float,
        gain: int,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.default_model = default_model
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.response_format = response_format
        self.stream = stream
        self.speed = speed
        self.gain = gain
        self._proc: Optional[subprocess.Popen[str]] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_url and self.api_key)

    @classmethod
    def from_config(cls) -> "SiliconFlowTTS":
        siliconflow_config = _load_siliconflow_config()
        return cls(
            api_url=siliconflow_config.get("api_url", config.SILICONFLOW_API_URL),
            api_key=os.environ.get(config.SILICONFLOW_API_KEY_ENV, siliconflow_config.get("api_key", "")),
            default_model=siliconflow_config.get("default_model", config.SILICONFLOW_DEFAULT_MODEL),
            default_voice=siliconflow_config.get("default_voice", config.SILICONFLOW_DEFAULT_VOICE),
            sample_rate=int(siliconflow_config.get("sample_rate", config.SILICONFLOW_SAMPLE_RATE)),
            response_format=siliconflow_config.get("response_format", config.SILICONFLOW_RESPONSE_FORMAT),
            stream=_to_bool(siliconflow_config.get("stream", config.SILICONFLOW_STREAM)),
            speed=float(siliconflow_config.get("speed", config.SILICONFLOW_SPEED)),
            gain=int(siliconflow_config.get("gain", config.SILICONFLOW_GAIN)),
        )

    async def speak(self, text: str) -> bool:
        prepared = await self.prepare(text)
        if prepared is None:
            return False
        return await self.play_prepared(prepared)

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        if not self.configured:
            return None
        audio_path = await self._generate_audio(text)
        if audio_path is None:
            return None
        return PreparedSpeech(provider="siliconflow", text=text, audio_path=audio_path)

    async def play_prepared(self, prepared: PreparedSpeech) -> bool:
        if prepared.audio_path is None:
            return False
        try:
            await asyncio.to_thread(self._play_audio, prepared.audio_path)
            return True
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def stop(self) -> None:
        _stop_proc(self._proc)
        self._proc = None

    async def _generate_audio(self, text: str) -> Optional[Path]:
        payload = {
            "input": text,
            "response_format": self.response_format,
            "sample_rate": self.sample_rate,
            "stream": self.stream,
            "speed": self.speed,
            "gain": self.gain,
            "model": self.default_model,
            "voice": self.default_voice,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=config.SILICONFLOW_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        body = await response.text()
                        print(f"siliconflow tts failed: http {response.status}: {body[:200]}")
                        return None
                    suffix = f".{self.response_format}"
                    path = Path(tempfile.gettempdir()) / f"claude-voice-tts-{uuid.uuid4().hex}{suffix}"
                    path.write_bytes(await response.read())
                    return path
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            print(f"siliconflow tts failed: {type(exc).__name__}: {exc}")
            return None

    def _play_audio(self, audio_path: Path) -> None:
        self._proc = subprocess.Popen(["afplay", str(audio_path)])
        self._proc.wait()
        self._proc = None


def _stop_proc(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)


def _load_siliconflow_config() -> dict:
    if not config.TTS_CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(config.TTS_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    siliconflow = data.get("siliconflow", {})
    return siliconflow if isinstance(siliconflow, dict) else {}


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}
