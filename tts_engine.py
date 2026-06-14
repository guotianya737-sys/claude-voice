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
from tts_text import normalize_for_tts


@dataclass
class PreparedSpeech:
    provider: str
    text: str
    audio_path: Optional[Path] = None


class TTSEngine:
    def __init__(self, provider: str = config.TTS_PROVIDER) -> None:
        self.provider = provider
        self.say = SayTTS()
        self.siliconflow = SiliconFlowTTS.from_config("siliconflow")
        self.siliconflow_moss = SiliconFlowTTS.from_config("siliconflow_moss")
        self.bailian = BailianTTS.from_config()

    async def speak(self, text: str) -> None:
        text = normalize_for_tts(text)
        if not text:
            return
        cloud_tts = self._cloud_tts()
        if cloud_tts is not None:
            ok = await cloud_tts.speak(text)
            if ok:
                return
        await self.say.speak(text)

    def can_prefetch(self) -> bool:
        cloud_tts = self._cloud_tts()
        return cloud_tts is not None and cloud_tts.configured

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = normalize_for_tts(text)
        if not text:
            return None
        cloud_tts = self._cloud_tts()
        if cloud_tts is not None:
            prepared = await cloud_tts.prepare(text)
            if prepared is not None:
                return prepared
        return PreparedSpeech(provider="say", text=text)

    async def play_prepared(self, prepared: PreparedSpeech) -> None:
        cloud_tts = self._provider_tts(prepared.provider)
        if cloud_tts is not None and prepared.audio_path is not None:
            ok = await cloud_tts.play_prepared(prepared)
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
        self.siliconflow_moss.stop()
        self.bailian.stop()

    def set_provider(self, provider: str) -> None:
        if provider not in {"say", "siliconflow", "siliconflow_moss", "bailian"}:
            return
        self.stop()
        self.provider = provider

    def _cloud_tts(self):
        return self._provider_tts(self.provider)

    def _provider_tts(self, provider: str):
        if provider == "siliconflow":
            return self.siliconflow
        if provider == "siliconflow_moss":
            return self.siliconflow_moss
        if provider == "bailian":
            return self.bailian
        return None


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
        provider: str,
        api_url: str,
        api_key: str,
        default_model: str,
        default_voice: str,
        sample_rate: int,
        response_format: str,
        stream: bool,
        speed: float,
        gain: float,
    ) -> None:
        self.provider = provider
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
    def from_config(cls, section: str) -> "SiliconFlowTTS":
        siliconflow_config = _load_siliconflow_config()
        section_config = siliconflow_config if section == "siliconflow" else siliconflow_config.get(section, {})
        if not isinstance(section_config, dict):
            section_config = {}
        common_api_key = siliconflow_config.get("api_key", "")
        if section == "siliconflow_moss":
            default_model = config.SILICONFLOW_MOSS_DEFAULT_MODEL
            default_voice = config.SILICONFLOW_MOSS_DEFAULT_VOICE
            sample_rate = config.SILICONFLOW_MOSS_SAMPLE_RATE
            response_format = config.SILICONFLOW_MOSS_RESPONSE_FORMAT
            stream = config.SILICONFLOW_MOSS_STREAM
            speed = config.SILICONFLOW_MOSS_SPEED
            gain = config.SILICONFLOW_MOSS_GAIN
        else:
            default_model = config.SILICONFLOW_DEFAULT_MODEL
            default_voice = config.SILICONFLOW_DEFAULT_VOICE
            sample_rate = config.SILICONFLOW_SAMPLE_RATE
            response_format = config.SILICONFLOW_RESPONSE_FORMAT
            stream = config.SILICONFLOW_STREAM
            speed = config.SILICONFLOW_SPEED
            gain = config.SILICONFLOW_GAIN
        selected_model = section_config.get("default_model", section_config.get("model", default_model))
        default_voice = section_config.get("default_voice", section_config.get("voice", default_voice))
        if section == "siliconflow_moss":
            default_voice = _normalize_moss_voice(str(default_voice), str(selected_model))
        return cls(
            provider=section,
            api_url=section_config.get("api_url", siliconflow_config.get("api_url", config.SILICONFLOW_API_URL)),
            api_key=os.environ.get(
                config.SILICONFLOW_API_KEY_ENV,
                section_config.get("api_key", common_api_key),
            ),
            default_model=selected_model,
            default_voice=default_voice,
            sample_rate=int(section_config.get("sample_rate", sample_rate)),
            response_format=section_config.get("response_format", response_format),
            stream=_to_bool(section_config.get("stream", stream)),
            speed=float(section_config.get("speed", speed)),
            gain=float(section_config.get("gain", gain)),
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
        return PreparedSpeech(provider=self.provider, text=text, audio_path=audio_path)

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


class BailianTTS:
    def __init__(
        self,
        api_key: str,
        websocket_api_url: str,
        model: str,
        voice: str,
        audio_format: str,
        speech_rate: float,
        volume: int,
        pitch_rate: float,
    ) -> None:
        self.provider = "bailian"
        self.api_key = api_key
        self.websocket_api_url = websocket_api_url
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.speech_rate = speech_rate
        self.volume = volume
        self.pitch_rate = pitch_rate
        self._proc: Optional[subprocess.Popen[str]] = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_config(cls) -> "BailianTTS":
        tts_config = _load_tts_config()
        bailian_config = tts_config.get("bailian", {})
        if not isinstance(bailian_config, dict):
            bailian_config = {}
        return cls(
            api_key=os.environ.get(config.BAILIAN_API_KEY_ENV, bailian_config.get("api_key", "")),
            websocket_api_url=bailian_config.get("websocket_api_url", config.BAILIAN_WEBSOCKET_API_URL),
            model=bailian_config.get("model", config.BAILIAN_DEFAULT_MODEL),
            voice=bailian_config.get("voice", config.BAILIAN_DEFAULT_VOICE),
            audio_format=bailian_config.get("audio_format", config.BAILIAN_AUDIO_FORMAT),
            speech_rate=float(bailian_config.get("speech_rate", config.BAILIAN_SPEECH_RATE)),
            volume=int(bailian_config.get("volume", config.BAILIAN_VOLUME)),
            pitch_rate=float(bailian_config.get("pitch_rate", config.BAILIAN_PITCH_RATE)),
        )

    async def speak(self, text: str) -> bool:
        prepared = await self.prepare(text)
        if prepared is None:
            return False
        return await self.play_prepared(prepared)

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        if not self.configured:
            return None
        audio_path = await asyncio.to_thread(self._generate_audio, text)
        if audio_path is None:
            return None
        return PreparedSpeech(provider=self.provider, text=text, audio_path=audio_path)

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

    def _generate_audio(self, text: str) -> Optional[Path]:
        try:
            import dashscope
            from dashscope.audio.tts_v2 import SpeechSynthesizer

            dashscope.api_key = self.api_key
            dashscope.base_websocket_api_url = self.websocket_api_url
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=_bailian_audio_format(self.audio_format),
                volume=self.volume,
                speech_rate=self.speech_rate,
                pitch_rate=self.pitch_rate,
            )
            audio = synthesizer.call(text)
            if not audio:
                print(f"bailian tts failed: empty audio: {synthesizer.get_response()}")
                return None
            path = Path(tempfile.gettempdir()) / f"claude-voice-bailian-{uuid.uuid4().hex}.mp3"
            path.write_bytes(audio)
            return path
        except Exception as exc:
            print(f"bailian tts failed: {type(exc).__name__}: {exc}")
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


def _load_tts_config() -> dict:
    if not config.TTS_CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(config.TTS_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_siliconflow_config() -> dict:
    data = _load_tts_config()
    siliconflow = data.get("siliconflow", {})
    return siliconflow if isinstance(siliconflow, dict) else {}


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _normalize_moss_voice(voice: str, model: str) -> str:
    voice = voice.strip()
    if not voice or ":" in voice or voice.startswith("speech:"):
        return voice
    return f"{model}:{voice}"


def _bailian_audio_format(name: str):
    from dashscope.audio.tts_v2 import AudioFormat

    normalized = str(name).strip().lower()
    mapping = {
        "mp3": AudioFormat.MP3_22050HZ_MONO_256KBPS,
        "mp3_16000": AudioFormat.MP3_16000HZ_MONO_128KBPS,
        "mp3_22050": AudioFormat.MP3_22050HZ_MONO_256KBPS,
        "mp3_24000": AudioFormat.MP3_24000HZ_MONO_256KBPS,
        "mp3_44100": AudioFormat.MP3_44100HZ_MONO_256KBPS,
        "mp3_48000": AudioFormat.MP3_48000HZ_MONO_256KBPS,
    }
    return mapping.get(normalized, AudioFormat.MP3_22050HZ_MONO_256KBPS)
