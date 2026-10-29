from __future__ import annotations

import asyncio
import contextlib
import json
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

    def set_emotion(self, emotion: str) -> None:
        self.emotion = None if emotion in {None, "", "default", "none"} else emotion
        self.indextts.set_emotion(emotion or "")

    def clear_emotion(self) -> None:
        self.emotion = None

    def set_provider(self, provider: str) -> None:
        if provider not in {"indextts", "sapi"}:
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
