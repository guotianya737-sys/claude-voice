from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import sounddevice as sd

import config
from tts_text import normalize_for_tts


@dataclass
class PreparedSpeech:
    provider: str
    text: str
    audio_path: Optional[Path] = None


class IndexTTSTTS:
    """本地 IndexTTS-2.5（独立进程，HTTP 调用）：能演情绪、能克隆音色。

    pi-voice 跑 Python 3.12，IndexTTS 要 3.11 + ROCm torch，所以拆成常驻服务
    （index-tts/indextts_server.py），这里只做请求 + 落盘 + 播放。
    """

    provider = "indextts"

    def __init__(self, base_url: str = config.INDEXTTS_URL, timeout: float = config.INDEXTTS_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._stop_requested = threading.Event()
        self._emotion: Optional[str] = None
        self._healthy_until = 0.0
        self._healthy = False
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def set_emotion(self, emotion: str) -> None:
        self._emotion = None if emotion in {None, "", "default", "none"} else emotion

    def can_prefetch(self) -> bool:
        return self.available()

    def available(self) -> bool:
        now = time.time()
        if now < self._healthy_until:
            return self._healthy
        healthy = False
        try:
            request = urllib.request.Request(f"{self.base_url}/health")
            with self._opener.open(request, timeout=config.INDEXTTS_HEALTH_TIMEOUT) as response:
                healthy = response.status == 200
        except Exception:  # noqa: BLE001 - 服务没起就是不可用
            healthy = False
        self._healthy = healthy
        self._healthy_until = now + (10.0 if healthy else 5.0)
        return healthy

    @property
    def configured(self) -> bool:
        return self.available()

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = text.strip()
        if not text or not self.available():
            return None
        self._stop_requested.clear()
        try:
            payload = {"text": text, "emotion": self._emotion, "emo_alpha": 0.85}
            audio = await asyncio.to_thread(self._request_audio, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"indextts failed: {type(exc).__name__}: {exc}")
            self._healthy_until = 0.0
            return None
        if not audio:
            return None
        path = Path(tempfile.gettempdir()) / f"pi-voice-idx-{uuid.uuid4().hex}.wav"
        path.write_bytes(audio)
        return PreparedSpeech(provider=self.provider, text=text, audio_path=path)

    async def play_prepared(self, prepared: PreparedSpeech) -> bool:
        if prepared.audio_path is None:
            return False
        try:
            await asyncio.to_thread(_play_audio_file, prepared.audio_path, self._stop_requested)
            return True
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    async def speak(self, text: str) -> bool:
        prepared = await self.prepare(text)
        if prepared is None:
            return False
        return await self.play_prepared(prepared)

    def stop(self) -> None:
        self._stop_requested.set()

    def _request_audio(self, payload: dict) -> bytes:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        started = time.time()
        request = urllib.request.Request(
            f"{self.base_url}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            data = response.read()
        if not data or data[:4] != b"RIFF":
            raise RuntimeError(f"非 wav 响应: {data[:120]!r}")
        frames = max(1, (len(data) - 44) // 2)
        print(f"[indextts] {len(payload['text'])}字 {time.time() - started:.2f}s -> {frames / 22050:.2f}s 音频")
        return data


class TTSEngine:
    def __init__(self, provider: str = config.TTS_PROVIDER) -> None:
        self.provider = provider
        self.emotion: Optional[str] = None
        self.indextts = IndexTTSTTS()
        self.sapi = SapiTTS()
        self.siliconflow = SiliconFlowTTS.from_config("siliconflow")
        self.siliconflow_moss = SiliconFlowTTS.from_config("siliconflow_moss")
        self.bailian = BailianTTS.from_config()

    async def speak(self, text: str) -> None:
        text = normalize_for_tts(text)
        if not text or not text.strip():
            return
        for provider in self._chain():
            impl = self._provider_tts(provider)
            if impl is None:
                continue
            try:
                if await impl.speak(text.strip()):
                    if provider != self.provider:
                        print(f"tts fell back to {provider}")
                    return
            except Exception as exc:  # noqa: BLE001 - 一个后端挂了换下一个
                print(f"tts {provider} failed: {type(exc).__name__}: {exc}")
        print("tts: 所有后端都失败了")

    def can_prefetch(self) -> bool:
        impl = self._provider_tts(self.provider)
        if impl is not None and hasattr(impl, "can_prefetch"):
            try:
                return bool(impl.can_prefetch())
            except Exception:  # noqa: BLE001
                return False
        return False

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = normalize_for_tts(text)
        if not text or not text.strip():
            return None
        for provider in self._chain():
            impl = self._provider_tts(provider)
            if impl is None:
                continue
            try:
                prepared = await self._prepare_with(provider, impl, text.strip())
            except Exception as exc:  # noqa: BLE001
                print(f"tts {provider} prepare failed: {type(exc).__name__}: {exc}")
                continue
            if prepared is not None:
                return prepared
        return PreparedSpeech(provider="sapi", text=text.strip())

    async def play_prepared(self, prepared: PreparedSpeech) -> None:
        impl = self._provider_tts(prepared.provider)
        if impl is not None and prepared.audio_path is not None:
            try:
                if await impl.play_prepared(prepared):
                    return
            except Exception as exc:  # noqa: BLE001
                print(f"tts {prepared.provider} playback failed: {type(exc).__name__}: {exc}")
        await self.sapi.speak(prepared.text)

    async def _prepare_with(self, provider: str, impl, text: str) -> Optional[PreparedSpeech]:
        if provider == "indextts":
            return await impl.prepare(text)
        return await impl.prepare(text)

    def discard_prepared(self, prepared: PreparedSpeech) -> None:
        if prepared.audio_path is not None:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def stop(self) -> None:
        self.indextts.stop()
        self.sapi.stop()
        self.siliconflow.stop()
        self.siliconflow_moss.stop()
        self.bailian.stop()

    def set_emotion(self, emotion: str) -> None:
        self.emotion = None if emotion in {None, "", "default", "none"} else emotion
        self.indextts.set_emotion(emotion or "")
        self.bailian.set_emotion(emotion or "")

    def clear_emotion(self) -> None:
        self.emotion = None
        self.bailian.clear_emotion()

    def set_provider(self, provider: str) -> None:
        if provider not in {"indextts", "sapi", "siliconflow", "siliconflow_moss", "bailian"}:
            return
        self.stop()
        self.provider = provider

    def _chain(self) -> list[str]:
        order = [self.provider]
        for candidate in config.TTS_FALLBACK_CHAIN:
            if candidate not in order:
                order.append(candidate)
        return order

    def _provider_tts(self, provider: str):
        if provider == "indextts":
            return self.indextts
        if provider == "sapi":
            return self.sapi
        if provider == "siliconflow":
            return self.siliconflow
        if provider == "siliconflow_moss":
            return self.siliconflow_moss
        if provider == "bailian":
            return self.bailian
        return None


class SapiTTS:
    """Windows 本地 SAPI 语音：离线、即时，替代 macOS 的 `say`。

    合成走 SAPI 直接写 wav 文件（pywin32 优先，缺依赖时回退 PowerShell），
    播放走 sounddevice，所以可以随时打断——和云端 TTS 的打断语义一致。
    """

    provider = "sapi"

    def __init__(
        self,
        voice: str = config.SAPI_VOICE,
        rate: int = config.SAPI_RATE,
        volume: int = config.SAPI_VOLUME,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self._stop_requested = threading.Event()
        self._backend: Optional[str] = None
        self._voices: list[str] = []

    @property
    def configured(self) -> bool:
        return self.available()

    def can_prefetch(self) -> bool:
        return self.available()

    def available(self) -> bool:
        if self._backend is None:
            self._backend = self._detect_backend()
        return self._backend is not None

    def list_voices(self) -> list[str]:
        if self._voices:
            return list(self._voices)
        if not self.available():
            return []
        try:
            if self._backend == "pywin32":
                self._voices = _with_com(lambda: _sapi_voice_names(True))
            else:
                self._voices = _sapi_voice_names(False)
        except Exception as exc:  # noqa: BLE001 - 列不出音色不该挡住说话
            print(f"sapi voices failed: {type(exc).__name__}: {exc}")
            self._voices = []
        return list(self._voices)

    async def speak(self, text: str) -> bool:
        prepared = await self.prepare(text)
        if prepared is None:
            return False
        return await self.play_prepared(prepared)

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = text.strip()
        if not text or not self.available():
            return None
        self._stop_requested.clear()
        path = Path(tempfile.gettempdir()) / f"pi-voice-sapi-{uuid.uuid4().hex}.wav"
        try:
            await asyncio.to_thread(self._synthesize, text, path)
        except Exception as exc:  # noqa: BLE001
            print(f"sapi tts failed: {type(exc).__name__}: {exc}")
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            return None
        if self._stop_requested.is_set() or not path.exists() or path.stat().st_size == 0:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            return None
        return PreparedSpeech(provider=self.provider, text=text, audio_path=path)

    async def play_prepared(self, prepared: PreparedSpeech) -> bool:
        if prepared.audio_path is None:
            return False
        try:
            await asyncio.to_thread(_play_audio_file, prepared.audio_path, self._stop_requested)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"sapi playback failed: {type(exc).__name__}: {exc}")
            return False
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def stop(self) -> None:
        self._stop_requested.set()

    # -- 后端实现 ---------------------------------------------------------
    @staticmethod
    def _detect_backend() -> Optional[str]:
        if not config.IS_WINDOWS:
            return None
        try:
            import pythoncom  # noqa: F401
            import win32com.client  # noqa: F401

            return "pywin32"
        except Exception:  # noqa: BLE001
            pass
        if shutil.which("powershell"):
            return "powershell"
        return None

    def _synthesize(self, text: str, path: Path) -> None:
        if not self.available():
            raise RuntimeError("没有可用的 SAPI 后端")
        if self._backend == "pywin32":
            _with_com(lambda: _sapi_synthesize_pywin32(text, path, self.voice, self.rate, self.volume))
            return
        _sapi_synthesize_powershell(text, path, self.voice, self.rate, self.volume)


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
        self._stop_requested = threading.Event()

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
        self._stop_requested.clear()
        prepared = await self.prepare(text)
        if prepared is None:
            return False
        return await self.play_prepared(prepared)

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        if not self.configured:
            return None
        self._stop_requested.clear()
        audio_path = await self._generate_audio(text)
        if audio_path is None:
            return None
        return PreparedSpeech(provider=self.provider, text=text, audio_path=audio_path)

    async def play_prepared(self, prepared: PreparedSpeech) -> bool:
        if prepared.audio_path is None:
            return False
        try:
            await asyncio.to_thread(_play_audio_file, prepared.audio_path, self._stop_requested)
            return True
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def stop(self) -> None:
        self._stop_requested.set()
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
        _play_audio_file(audio_path, self._stop_requested)


class BailianTTS:
    def __init__(
        self,
        api_key: str,
        websocket_api_url: str,
        http_api_url: str,
        qwen_realtime_websocket_api_url: str,
        model: str,
        voice: str,
        audio_format: str,
        speech_rate: float,
        volume: int,
        pitch_rate: float,
        instruction: str = "",
    ) -> None:
        self.provider = "bailian"
        self.api_key = api_key
        self.websocket_api_url = websocket_api_url
        self.http_api_url = http_api_url
        self.qwen_realtime_websocket_api_url = qwen_realtime_websocket_api_url
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.speech_rate = speech_rate
        self.volume = volume
        self.pitch_rate = pitch_rate
        self.instruction = instruction
        self._emotion_instruction: Optional[str] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._active_synth = None
        self._active_stream: Optional[sd.RawOutputStream] = None
        self._stop_requested = threading.Event()
        self._dashscope_configured = False

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def active_instruction(self) -> Optional[str]:
        """Return emotion override if set, otherwise the base instruction.

        实测 cosyvoice-v3-flash 不接受 instruction（引擎返回 428），所以默认关闭：
        关掉时情绪标签被忽略，而不是让整句合成失败、静默降级成 SAPI。
        """
        if not getattr(config, "BAILIAN_INSTRUCTION_ENABLED", False):
            return None
        if self._emotion_instruction is not None:
            return self._emotion_instruction
        return self.instruction or None

    def set_emotion(self, emotion: str) -> None:
        mapping = config.BAILIAN_EMOTION_INSTRUCTIONS
        self._emotion_instruction = mapping.get(emotion)

    def clear_emotion(self) -> None:
        self._emotion_instruction = None

    @staticmethod
    def _pi_auth_key() -> str:
        """从 ~/.pi/agent/auth.json 取 bailian key —— 复用 pi 已有的凭证，不复制不打印。"""
        path = Path.home() / ".pi" / "agent" / "auth.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        entry = data.get("bailian")
        if isinstance(entry, dict):
            return str(entry.get("key") or entry.get("apiKey") or "")
        return ""

    @classmethod
    def from_config(cls) -> "BailianTTS":
        tts_config = _load_tts_config()
        bailian_config = tts_config.get("bailian", {})
        if not isinstance(bailian_config, dict):
            bailian_config = {}
        return cls(
            api_key=os.environ.get(config.BAILIAN_API_KEY_ENV)
            or str(bailian_config.get("api_key", ""))
            or cls._pi_auth_key(),
            websocket_api_url=bailian_config.get("websocket_api_url", config.BAILIAN_WEBSOCKET_API_URL),
            http_api_url=bailian_config.get("http_api_url", config.BAILIAN_HTTP_API_URL),
            qwen_realtime_websocket_api_url=bailian_config.get(
                "qwen_realtime_websocket_api_url",
                config.BAILIAN_QWEN_REALTIME_WEBSOCKET_API_URL,
            ),
            model=bailian_config.get("model", config.BAILIAN_DEFAULT_MODEL),
            voice=bailian_config.get("voice", config.BAILIAN_DEFAULT_VOICE),
            audio_format=bailian_config.get("audio_format", config.BAILIAN_AUDIO_FORMAT),
            speech_rate=float(bailian_config.get("speech_rate", config.BAILIAN_SPEECH_RATE)),
            volume=int(bailian_config.get("volume", config.BAILIAN_VOLUME)),
            pitch_rate=float(bailian_config.get("pitch_rate", config.BAILIAN_PITCH_RATE)),
            instruction=str(bailian_config.get("instruction", config.BAILIAN_INSTRUCTION)),
        )

    def _ensure_dashscope(self) -> None:
        if self._dashscope_configured:
            return
        # 代理只该走 LLM 的 HTTP，不该拖住 TTS 的 WebSocket：实测带代理首音 5.5s，直连 0.57s
        _bypass_proxy_for_dashscope()
        import dashscope

        dashscope.api_key = self.api_key
        dashscope.base_websocket_api_url = self.websocket_api_url
        dashscope.base_http_api_url = self.http_api_url
        self._dashscope_configured = True

    def _create_synthesizer(self, callback=None):
        self._ensure_dashscope()
        from dashscope.audio.tts_v2 import SpeechSynthesizer

        return SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=_bailian_audio_format(self.audio_format),
            volume=self.volume,
            speech_rate=self.speech_rate,
            pitch_rate=self.pitch_rate,
            instruction=self.active_instruction,
            callback=callback,
        )

    def stop(self) -> None:
        self._stop_requested.set()
        synth = self._active_synth
        if synth is not None:
            with contextlib.suppress(Exception):
                synth.streaming_cancel()
            with contextlib.suppress(Exception):
                synth.close()
            self._active_synth = None
        stream = self._active_stream
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()
            self._active_stream = None
        _stop_proc(self._proc)
        self._proc = None

    async def speak(self, text: str) -> bool:
        if not self.configured:
            return False
        text = text.strip()
        if not text:
            return False
        self._stop_requested.clear()
        if _is_qwen_realtime_model(self.model):
            return await asyncio.to_thread(self._stream_qwen_realtime_to_speaker, text)
        if _is_qwen_http_model(self.model):
            prepared = await self.prepare(text)
            if prepared is None:
                return False
            return await self.play_prepared(prepared)
        return await asyncio.to_thread(self._stream_to_speaker, text)

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        if not self.configured or not _is_qwen_http_model(self.model):
            return None
        text = text.strip()
        if not text:
            return None
        self._stop_requested.clear()
        audio_path = await asyncio.to_thread(self._generate_qwen_http_audio, text)
        if audio_path is None:
            return None
        return PreparedSpeech(provider=self.provider, text=text, audio_path=audio_path)

    async def play_prepared(self, prepared: PreparedSpeech) -> bool:
        if prepared.audio_path is None:
            return False
        try:
            await asyncio.to_thread(_play_audio_file, prepared.audio_path, self._stop_requested)
            return True
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.audio_path.unlink()

    def can_prefetch(self) -> bool:
        return self.configured and _is_qwen_http_model(self.model)

    def _generate_qwen_http_audio(self, text: str) -> Optional[Path]:
        self._ensure_dashscope()
        from dashscope.audio.qwen_tts import SpeechSynthesizer

        path = Path(tempfile.gettempdir()) / f"claude-voice-bailian-{uuid.uuid4().hex}.wav"
        try:
            chunks = SpeechSynthesizer.call(
                model=self.model,
                text=text,
                api_key=self.api_key,
                voice=self.voice,
                language_type="Chinese",
                stream=True,
                **({"instruction": self.active_instruction} if self.active_instruction else {}),
            )
            wrote_audio = False
            with path.open("wb") as file:
                for response in chunks:
                    if self._stop_requested.is_set():
                        return None
                    status_code = getattr(response, "status_code", None)
                    code = getattr(response, "code", "")
                    if status_code is not None and int(status_code) >= 400:
                        print(f"bailian qwen tts failed: http {status_code}: {getattr(response, 'message', '')}")
                        return None
                    if code:
                        print(f"bailian qwen tts failed: {code}: {getattr(response, 'message', '')}")
                        return None
                    data = _qwen_response_audio_data(response)
                    if not data:
                        continue
                    file.write(base64.b64decode(data))
                    wrote_audio = True
            if not wrote_audio:
                print("bailian qwen tts failed: empty audio")
                return None
            if self._stop_requested.is_set():
                return None
            return path
        except Exception as exc:
            print(f"bailian qwen tts failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            if self._stop_requested.is_set():
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    def _play_audio(self, audio_path: Path) -> None:
        _play_audio_file(audio_path, self._stop_requested)

    def _stream_qwen_realtime_to_speaker(self, text: str) -> bool:
        self._ensure_dashscope()
        from dashscope.audio.qwen_tts_realtime import AudioFormat, QwenTtsRealtime, QwenTtsRealtimeCallback

        stream: Optional[sd.RawOutputStream] = None
        opened = False
        error: Optional[str] = None
        audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=config.BAILIAN_AUDIO_QUEUE_MAX_CHUNKS)
        stop_playback = threading.Event()
        playback_done = threading.Event()
        playback_thread: Optional[threading.Thread] = None
        sample_rate = _bailian_audio_sample_rate(self.audio_format)
        owner = self

        def set_error(message: str) -> None:
            nonlocal error
            if error is None:
                error = message
            stop_playback.set()

        def playback_loop() -> None:
            nonlocal stream, opened
            try:
                stream = sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                )
                stream.start()
                owner._active_stream = stream
                opened = True
                while not stop_playback.is_set():
                    data = audio_queue.get()
                    if data is None:
                        return
                    stream.write(data)
            except Exception as exc:
                set_error(f"{type(exc).__name__}: {exc}")
            finally:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.stop()
                    with contextlib.suppress(Exception):
                        stream.close()
                if owner._active_stream is stream:
                    owner._active_stream = None
                playback_done.set()

        class Callback(QwenTtsRealtimeCallback):
            def on_open(self) -> None:
                pass

            def on_event(self, message) -> None:
                try:
                    event = json.loads(message) if isinstance(message, str) else message
                    event_type = event.get("type")
                    if event_type == "response.audio.delta":
                        data = base64.b64decode(event.get("delta", ""))
                        if data:
                            audio_queue.put(data, timeout=1.0)
                    elif event_type == "error" or event.get("error"):
                        set_error(str(event.get("error", event)))
                except queue.Full:
                    set_error("audio playback queue full")
                except Exception as exc:
                    set_error(f"{type(exc).__name__}: {exc}")

            def on_close(self, close_status_code, close_msg) -> None:
                pass

        callback = Callback()
        realtime = None
        try:
            playback_thread = threading.Thread(target=playback_loop, daemon=True)
            playback_thread.start()
            realtime = QwenTtsRealtime(
                model=self.model,
                callback=callback,
                url=self.qwen_realtime_websocket_api_url,
            )
            self._active_synth = realtime
            realtime.connect()
            realtime.update_session(
                voice=self.voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                sample_rate=sample_rate,
                volume=self.volume,
                speech_rate=self.speech_rate,
                pitch_rate=self.pitch_rate,
                mode="server_commit",
                **({"instructions": self.active_instruction} if self.active_instruction else {}),
            )
            realtime.append_text(text)
            realtime.finish()
            if error:
                print(f"bailian qwen realtime tts failed: {error}")
                return False
            return opened
        except Exception as exc:
            print(f"bailian qwen realtime tts failed: {type(exc).__name__}: {exc}")
            return False
        finally:
            self._active_synth = None
            if realtime is not None:
                with contextlib.suppress(Exception):
                    realtime.close()
            if error is not None:
                stop_playback.set()
            if playback_thread is not None:
                try:
                    audio_queue.put(None, timeout=1.0)
                except queue.Full:
                    stop_playback.set()
                playback_done.wait(timeout=5.0)

    def _stream_to_speaker(self, text: str) -> bool:
        stream: Optional[sd.RawOutputStream] = None
        opened = False
        error: Optional[str] = None
        audio_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=config.BAILIAN_AUDIO_QUEUE_MAX_CHUNKS)
        stop_playback = threading.Event()
        playback_done = threading.Event()
        playback_thread: Optional[threading.Thread] = None
        sample_rate = _bailian_audio_sample_rate(self.audio_format)

        from dashscope.audio.tts_v2 import ResultCallback

        owner = self

        def set_error(message: str) -> None:
            nonlocal error
            if error is None:
                error = message
            stop_playback.set()

        def playback_loop() -> None:
            nonlocal stream, opened
            try:
                stream = sd.RawOutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                )
                stream.start()
                owner._active_stream = stream
                opened = True
                while not stop_playback.is_set():
                    data = audio_queue.get()
                    if data is None:
                        return
                    stream.write(data)
            except Exception as exc:
                set_error(f"{type(exc).__name__}: {exc}")
            finally:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.stop()
                    with contextlib.suppress(Exception):
                        stream.close()
                if owner._active_stream is stream:
                    owner._active_stream = None
                playback_done.set()

        class Callback(ResultCallback):
            def on_open(self) -> None:
                nonlocal playback_thread
                try:
                    playback_thread = threading.Thread(target=playback_loop, daemon=True)
                    playback_thread.start()
                except Exception as exc:
                    set_error(f"{type(exc).__name__}: {exc}")

            def on_data(self, data: bytes) -> None:
                if error is not None or not data:
                    return
                try:
                    audio_queue.put(data, timeout=1.0)
                except queue.Full:
                    set_error("audio playback queue full")
                except Exception as exc:
                    set_error(f"{type(exc).__name__}: {exc}")

            def on_error(self, message) -> None:
                set_error(str(message))

            def on_close(self) -> None:
                pass

        callback = Callback()
        synth = None
        try:
            synth = self._create_synthesizer(callback=callback)
            self._active_synth = synth
            synth.streaming_call(text)
            synth.streaming_complete(config.BAILIAN_CALL_TIMEOUT_MS)
            if error:
                print(f"bailian tts failed: {error}")
                return False
            return opened
        except Exception as exc:
            print(f"bailian tts failed: {type(exc).__name__}: {exc}")
            if synth is not None:
                with contextlib.suppress(Exception):
                    synth.close()
            return False
        finally:
            self._active_synth = None
            if error is not None:
                stop_playback.set()
            if playback_thread is not None:
                try:
                    audio_queue.put(None, timeout=1.0)
                except queue.Full:
                    stop_playback.set()
                playback_done.wait(timeout=5.0)

def _stop_proc(proc: Optional[subprocess.Popen[str]]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1)


def _play_audio_file(audio_path: Path, stop_event: Optional[threading.Event] = None) -> None:
    """用 sounddevice 播放 wav 文件；stop_event 置位时立即收手（用于打断）。"""
    if stop_event is not None and stop_event.is_set():
        return
    path = Path(audio_path)
    if path.suffix.lower() != ".wav":
        raise RuntimeError(f"只支持 wav 播放（当前文件 {path.name}）；请把 response_format 设为 wav")
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        if width != 2:
            raise RuntimeError(f"不支持的 wav 位深: {width * 8}bit")
        stream = sd.RawOutputStream(samplerate=rate, channels=channels, dtype="int16")
        try:
            stream.start()
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                data = reader.readframes(2048)
                if not data:
                    break
                stream.write(data)
        finally:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()


def _bypass_proxy_for_dashscope() -> None:
    hosts = ("aliyuncs.com", ".aliyuncs.com", "localhost", "127.0.0.1", "::1")
    for var in ("NO_PROXY", "no_proxy"):
        current = [item for item in os.environ.get(var, "").split(",") if item.strip()]
        for host in hosts:
            if host not in current:
                current.append(host)
        os.environ[var] = ",".join(current)


def _with_com(callback):
    """在恰好的位置初始化一次 COM apartment —— 嵌套 CoInitialize/CoUninitialize
    会把调用线程的 COM 状态弄坏（SAPI 之后就会报 DISP_E_EXCEPTION）。"""
    import pythoncom

    pythoncom.CoInitialize()
    try:
        return callback()
    finally:
        pythoncom.CoUninitialize()


def _sapi_voice_names(use_com: bool) -> list[str]:
    if use_com:
        import win32com.client

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        tokens = speaker.GetVoices()
        return [tokens.Item(i).GetDescription() for i in range(tokens.Count)]
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
        "ForEach-Object { $_.VoiceInfo.Name }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _sapi_select_voice(voice: str, voices: list[str], set_voice) -> None:
    target = voice.strip()
    if not target:
        # 自动挑中文音色（Huihui 之类），否则用系统默认
        lowered = [name.lower() for name in voices]
        for marker in ("chinese", "zh-cn", "huihui"):
            index = next((i for i, name in enumerate(lowered) if marker in name), None)
            if index is not None:
                set_voice(voices[index])
                return
        return
    match = next((name for name in voices if target.lower() in name.lower()), None)
    if match:
        set_voice(match)


def _sapi_synthesize_pywin32(text: str, path: Path, voice: str, rate: int, volume: int) -> None:
    """把 text 合成成 wav 文件（调用方负责先 _with_com）。"""
    import win32com.client

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    voices = _sapi_voice_names(use_com=True)

    def apply(name: str) -> None:
        tokens = speaker.GetVoices()
        for index in range(tokens.Count):
            token = tokens.Item(index)
            if token.GetDescription() == name:
                speaker.Voice = token
                return

    _sapi_select_voice(voice, voices, apply)
    speaker.Rate = max(-10, min(10, int(rate)))
    speaker.Volume = max(0, min(100, int(volume)))
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(str(path), 3)  # SSFMCreateForWrite
    try:
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
    finally:
        stream.Close()


def _sapi_synthesize_powershell(text: str, path: Path, voice: str, rate: int, volume: int) -> None:
    """pywin32 不可用时的兜底：文本走 UTF-8 文件，避开命令行编码问题。"""
    work = Path(tempfile.gettempdir()) / f"pi-voice-sapi-{uuid.uuid4().hex}"
    text_path = work.with_suffix(".txt")
    script_path = work.with_suffix(".ps1")
    text_path.write_text(text, encoding="utf-8")
    script_path.write_text(
        "param([string]$TextPath, [string]$OutPath, [int]$Rate, [int]$Volume, [string]$Voice)\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$text = [IO.File]::ReadAllText($TextPath, [Text.Encoding]::UTF8)\n"
        "if ($Voice) { try { $synth.SelectVoice($Voice) } catch {} }\n"
        "$synth.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate))\n"
        "$synth.Volume = [Math]::Max(0, [Math]::Min(100, $Volume))\n"
        "$synth.SetOutputToWaveFile($OutPath)\n"
        "$synth.Speak($text)\n"
        "$synth.Dispose()\n",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                str(text_path),
                str(path),
                str(int(rate)),
                str(int(volume)),
                voice or "",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode != 0 or not path.exists():
            raise RuntimeError((result.stderr or result.stdout or "powershell synth failed").strip()[:300])
    finally:
        for leftover in (text_path, script_path):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()


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


def _is_qwen_http_model(model: str) -> bool:
    normalized = str(model).strip().lower()
    return normalized.startswith(("qwen-tts", "qwen3-tts")) and "realtime" not in normalized


def _is_qwen_realtime_model(model: str) -> bool:
    normalized = str(model).strip().lower()
    return normalized.startswith(("qwen-tts", "qwen3-tts")) and "realtime" in normalized


def _qwen_response_audio_data(response) -> str:
    output = getattr(response, "output", None)
    audio = getattr(output, "audio", None) if output is not None else None
    if isinstance(audio, dict):
        return str(audio.get("data") or "")
    if audio is not None and hasattr(audio, "data"):
        return str(getattr(audio, "data") or "")
    return ""


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
        "pcm": AudioFormat.PCM_24000HZ_MONO_16BIT,
        "pcm_16000": AudioFormat.PCM_16000HZ_MONO_16BIT,
        "pcm_22050": AudioFormat.PCM_22050HZ_MONO_16BIT,
        "pcm_24000": AudioFormat.PCM_24000HZ_MONO_16BIT,
        "pcm_44100": AudioFormat.PCM_44100HZ_MONO_16BIT,
        "pcm_48000": AudioFormat.PCM_48000HZ_MONO_16BIT,
    }
    return mapping.get(normalized, AudioFormat.PCM_24000HZ_MONO_16BIT)


def _bailian_audio_sample_rate(name: str) -> int:
    normalized = str(name).strip().lower()
    if "16000" in normalized:
        return 16000
    if "22050" in normalized:
        return 22050
    if "44100" in normalized:
        return 44100
    if "48000" in normalized:
        return 48000
    return 24000
