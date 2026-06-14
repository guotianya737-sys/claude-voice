from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import queue
import subprocess
import tempfile
import threading
import uuid
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
        text = text.strip()
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
        if hasattr(cloud_tts, "can_prefetch"):
            return bool(cloud_tts.can_prefetch())
        return cloud_tts is not None and cloud_tts.configured

    async def prepare(self, text: str) -> Optional[PreparedSpeech]:
        text = normalize_for_tts(text)
        if not text:
            return None
        text = text.strip()
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
        http_api_url: str,
        qwen_realtime_websocket_api_url: str,
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
        self.http_api_url = http_api_url
        self.qwen_realtime_websocket_api_url = qwen_realtime_websocket_api_url
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.speech_rate = speech_rate
        self.volume = volume
        self.pitch_rate = pitch_rate
        self._proc: Optional[subprocess.Popen[str]] = None
        self._active_synth = None
        self._active_stream: Optional[sd.RawOutputStream] = None
        self._stop_requested = threading.Event()
        self._dashscope_configured = False

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
        )

    def _ensure_dashscope(self) -> None:
        if self._dashscope_configured:
            return
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
            await asyncio.to_thread(self._play_audio, prepared.audio_path)
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
        if self._stop_requested.is_set():
            return
        self._proc = subprocess.Popen(["afplay", str(audio_path)])
        self._proc.wait()
        self._proc = None

    def _speak_qwen_http(self, text: str) -> bool:
        prepared = self._generate_qwen_http_audio(text)
        if prepared is None:
            return False
        try:
            self._play_audio(prepared)
            return not self._stop_requested.is_set()
        finally:
            with contextlib.suppress(FileNotFoundError):
                prepared.unlink()

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
