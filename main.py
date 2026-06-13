from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np

import claude_client
import config
from asr_engine import ASREngine
from audio_io import AudioCapture
from claude_client import ClaudeClient
from server import VoiceServer
from tts_engine import PreparedSpeech, TTSEngine
from vad import VADEngine


class Mode(str, Enum):
    HANDSFREE = "handsfree"
    PTT = "ptt"


class Status(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class StreamingReplySpeaker:
    END_CHARS = "。！？!?；;\n"
    SOFT_CHARS = "，、,：: "
    BACKTRACK_BREAK_CHARS = "的了是呢吗吧啊呀哦嘛啦喔耶呐么着过和与及"

    def __init__(self, app: "MiruVoiceApp") -> None:
        self.app = app
        self.buffer = ""
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        if self.app.tts.can_prefetch():
            self.worker = asyncio.create_task(self._prefetching_worker())
        else:
            self.worker = asyncio.create_task(self._worker())
        self.interrupted_pcm: Optional[bytes] = None
        self.closed = False
        self.displayed_text = ""

    async def feed(self, token: str) -> None:
        if self.closed or self.interrupted_pcm is not None:
            return
        self.displayed_text += token
        await self.app.server.broadcast({"type": "assistant_token", "text": token})
        self.buffer += token
        for chunk in self._pop_ready_chunks(flush=False):
            await self.queue.put(chunk)

    async def finish(self) -> Optional[bytes]:
        for chunk in self._pop_ready_chunks(flush=True):
            await self.queue.put(chunk)
        await self.queue.put(None)
        await self.worker
        self.closed = True
        return self.interrupted_pcm

    async def _worker(self) -> None:
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                return
            if self.interrupted_pcm is not None:
                continue
            await self.app.set_status(Status.SPEAKING)
            interrupted = await self.app.speak_with_barge_in(chunk)
            if interrupted is not None:
                self.interrupted_pcm = interrupted
                self._drain_pending()
                return

    async def _prefetching_worker(self) -> None:
        prepared_queue: asyncio.Queue[Optional[PreparedSpeech]] = asyncio.Queue()
        producer = asyncio.create_task(self._prepare_worker(prepared_queue))
        try:
            while True:
                prepared = await prepared_queue.get()
                if prepared is None:
                    return
                if self.interrupted_pcm is not None:
                    self.app.tts.discard_prepared(prepared)
                    continue
                await self.app.set_status(Status.SPEAKING)
                interrupted = await self.app.speak_prepared_with_barge_in(prepared)
                if interrupted is not None:
                    self.interrupted_pcm = interrupted
                    self._drain_pending()
                    return
        finally:
            if not producer.done():
                producer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer
            self._drain_prepared(prepared_queue)

    async def _prepare_worker(self, prepared_queue: asyncio.Queue[Optional[PreparedSpeech]]) -> None:
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                await prepared_queue.put(None)
                return
            if self.interrupted_pcm is not None:
                continue
            prepared = await self.app.tts.prepare(chunk)
            if prepared is not None:
                await prepared_queue.put(prepared)

    def _drain_pending(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _drain_prepared(self, queue: asyncio.Queue[Optional[PreparedSpeech]]) -> None:
        while True:
            try:
                prepared = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if prepared is not None:
                self.app.tts.discard_prepared(prepared)

    def _pop_ready_chunks(self, flush: bool) -> list[str]:
        chunks: list[str] = []
        while self.buffer:
            cut = self._find_cut(flush)
            if cut is None:
                break
            chunk = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _find_cut(self, flush: bool) -> Optional[int]:
        if flush:
            return len(self.buffer)

        if self.app.tts.provider == "siliconflow":
            max_chars = config.SILICONFLOW_TTS_CHUNK_MAX_CHARS
            min_chars = config.SILICONFLOW_TTS_CHUNK_MIN_CHARS
        else:
            max_chars = config.TTS_CHUNK_MAX_CHARS
            min_chars = config.TTS_CHUNK_MIN_CHARS
        for index, char in enumerate(self.buffer):
            if char in self.END_CHARS and index + 1 >= min_chars:
                return index + 1

        if len(self.buffer) < max_chars:
            return None

        soft_limit = self.buffer[:max_chars]
        for index in range(len(soft_limit) - 1, min_chars - 1, -1):
            if soft_limit[index] in self.SOFT_CHARS:
                return index + 1
        for index in range(len(soft_limit) - 1, min_chars - 1, -1):
            if soft_limit[index] in self.BACKTRACK_BREAK_CHARS:
                return index + 1
        return max_chars


class MiruVoiceApp:
    def __init__(self, host: str, port: int, verbose: bool = False) -> None:
        self.host = host
        self.port = port
        self.verbose = verbose
        self.mode = Mode.HANDSFREE
        self.status = Status.IDLE
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=config.AUDIO_QUEUE_MAX_CHUNKS)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.audio_processor_ready = asyncio.Event()
        self.ptt_chunks: list[np.ndarray] = []
        self.ptt_recording = False
        self.closed = asyncio.Event()
        self.last_meter_at = 0.0

        self.audio = AudioCapture()
        self.vad = VADEngine()
        self.barge_vad = self.create_barge_vad(self.vad.model)
        self.asr = ASREngine()
        self.tts = TTSEngine()
        self.claude = ClaudeClient()
        self.server = VoiceServer(self.handle_command, self.current_state)

    async def start(self) -> None:
        claude_client.capture_loop()
        self.loop = asyncio.get_running_loop()
        self.audio.set_callback(self._on_audio)
        processor = asyncio.create_task(self.process_audio())
        processor.add_done_callback(self.report_processor_error)
        self.audio.start()
        await self.server.start(self.host, self.port)
        await self.set_status(Status.LISTENING)
        print(f"claude-voice listening at http://{self.host}:{self.port}")
        await self.closed.wait()
        processor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await processor

    async def close(self) -> None:
        self.closed.set()
        self.audio.close()
        self.tts.stop()
        self.claude.close()
        await self.server.stop()

    async def handle_command(self, payload: dict) -> None:
        command = payload.get("type")
        if command == "start_listening":
            if self.mode == Mode.PTT:
                self.audio.stop()
                self.drain_audio_queue()
                self.vad.reset()
                self.ptt_recording = False
                await self.set_status(Status.IDLE)
                await self.server.broadcast({"type": "mode", "mode": self.mode.value})
                return
            self.mode = Mode.HANDSFREE
            self.drain_audio_queue()
            self.vad.reset()
            self.audio.resume()
            await self.set_status(Status.LISTENING)
            await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "stop_listening":
            self.audio.stop()
            self.drain_audio_queue()
            self.vad.reset()
            await self.set_status(Status.IDLE)
        elif command == "switch_mode":
            mode = payload.get("mode")
            if mode in {Mode.HANDSFREE.value, Mode.PTT.value}:
                self.mode = Mode(mode)
                self.vad.reset()
                self.ptt_chunks.clear()
                self.ptt_recording = False
                self.drain_audio_queue()
                if self.mode == Mode.HANDSFREE:
                    self.audio.resume()
                    await self.set_status(Status.LISTENING)
                else:
                    self.audio.stop()
                    await self.set_status(Status.IDLE)
                await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "push_to_talk_start":
            self.mode = Mode.PTT
            self.ptt_chunks.clear()
            self.drain_audio_queue()
            self.ptt_recording = True
            self.audio.resume()
            await self.set_status(Status.LISTENING)
            await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "push_to_talk_end":
            self.ptt_recording = False
            if self.ptt_chunks:
                audio = np.concatenate(self.ptt_chunks)
                self.ptt_chunks.clear()
                await self.handle_segment((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            else:
                self.audio.stop()
                self.drain_audio_queue()
            await self.set_status(Status.IDLE)
        elif command == "switch_tts":
            provider = payload.get("provider")
            self.tts.set_provider(provider)
            await self.server.broadcast({"type": "tts_provider", "provider": self.tts.provider})
        elif command == "refresh_audio_input":
            self.audio.reopen()
            self.drain_audio_queue()
            self.vad.reset()
            self.barge_vad.reset()
            await self.server.broadcast(self.audio.input_state())
            if self.mode == Mode.HANDSFREE and self.status == Status.IDLE:
                self.audio.resume()
                await self.set_status(Status.LISTENING)

    async def process_audio(self) -> None:
        while True:
            await self.audio_processor_ready.wait()
            chunk = await self.audio_queue.get()
            if self.status != Status.LISTENING:
                continue
            if self.mode == Mode.PTT:
                if self.ptt_recording:
                    self.ptt_chunks.append(chunk)
                continue
            segments = list(self.vad.process(chunk))
            await self.maybe_broadcast_audio_meter()
            for segment in segments:
                await self.handle_segment(segment)

    async def handle_segment(self, pcm: bytes) -> None:
        next_pcm: Optional[bytes] = pcm
        while next_pcm is not None:
            next_pcm = await self.handle_single_segment(next_pcm)

    async def handle_single_segment(self, pcm: bytes) -> Optional[bytes]:
        self.audio.stop()
        self.drain_audio_queue()
        self.vad.reset()
        await self.set_status(Status.THINKING)
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self.asr.transcribe, pcm),
                timeout=config.ASR_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            print(f"asr failed: {type(exc).__name__}: {exc}")
            text = ""
        if not text:
            await self.resume_after_turn()
            return None

        await self.server.broadcast({"type": "user_msg", "text": text})
        speaker = StreamingReplySpeaker(self)
        try:
            reply = await asyncio.wait_for(
                self.claude.send(text, speaker.feed),
                timeout=config.CLAUDE_RESPONSE_TIMEOUT,
            )
            interrupted_pcm = await speaker.finish()
        except asyncio.TimeoutError as exc:
            print(f"claude failed: {type(exc).__name__}: {exc}")
            self.claude.close()
            self.tts.stop()
            interrupted_pcm = await speaker.finish()
            await self.server.broadcast({"type": "assistant_done", "text": speaker.displayed_text})
            await self.resume_after_turn()
            return interrupted_pcm
        except Exception as exc:
            print(f"claude failed: {type(exc).__name__}: {exc}")
            self.tts.stop()
            interrupted_pcm = await speaker.finish()
            await self.server.broadcast({"type": "assistant_done", "text": speaker.displayed_text})
            await self.resume_after_turn()
            return interrupted_pcm
        await self.server.broadcast({"type": "assistant_done", "text": speaker.displayed_text or reply})
        if interrupted_pcm is not None:
            await self.server.broadcast({"type": "status", "status": "thinking"})
            return interrupted_pcm
        await asyncio.sleep(config.TTS_TAIL_GUARD)
        await self.resume_after_turn()
        return None

    async def resume_after_turn(self) -> None:
        self.drain_audio_queue()
        self.vad.reset()
        self.barge_vad.reset()
        if self.mode == Mode.HANDSFREE:
            self.audio.resume()
            await self.set_status(Status.LISTENING)
        else:
            await self.set_status(Status.IDLE)

    async def on_assistant_token(self, token: str) -> None:
        await self.server.broadcast({"type": "assistant_token", "text": token})

    async def maybe_broadcast_audio_meter(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now - self.last_meter_at < 0.25:
            return
        self.last_meter_at = now
        await self.server.broadcast(
            {
                "type": "audio_meter",
                "db": float(round(float(self.vad.last_db), 1)),
                "prob": float(round(float(self.vad.last_prob), 3)),
                "speech": bool(self.vad.last_is_speech),
                "db_threshold": float(round(float(self.vad.effective_db_threshold), 1)),
                "prob_threshold": config.PROB_THRESHOLD,
            }
        )

    async def set_status(self, status: Status) -> None:
        self.status = status
        if status == Status.LISTENING:
            self.audio_processor_ready.set()
        else:
            self.audio_processor_ready.clear()
        await self.server.broadcast({"type": "status", "status": status.value})

    def current_state(self) -> list[dict]:
        return [
            {"type": "mode", "mode": self.mode.value},
            {"type": "status", "status": self.status.value},
            {"type": "tts_provider", "provider": self.tts.provider},
            self.audio.input_state(),
        ]

    def _on_audio(self, chunk: np.ndarray) -> None:
        if self.loop is None:
            return
        if not self.should_accept_audio():
            return
        self.loop.call_soon_threadsafe(self._enqueue_audio, chunk)

    def _enqueue_audio(self, chunk: np.ndarray) -> None:
        if not self.should_accept_audio():
            return
        try:
            self.audio_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self.drain_audio_queue()

    @staticmethod
    def dbfs(chunk: np.ndarray) -> float:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-10)
        return 20.0 * np.log10(rms) + 100.0

    @staticmethod
    def report_processor_error(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"audio processor failed: {type(exc).__name__}: {exc}")

    def drain_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def speak_with_barge_in(self, text: str) -> Optional[bytes]:
        return await self.play_with_barge_in(lambda: self.tts.speak(text))

    async def speak_prepared_with_barge_in(self, prepared: PreparedSpeech) -> Optional[bytes]:
        return await self.play_with_barge_in(lambda: self.tts.play_prepared(prepared))

    async def play_with_barge_in(self, play) -> Optional[bytes]:
        if self.mode != Mode.HANDSFREE or not config.BARGE_IN_ENABLED:
            await play()
            return None

        self.configure_barge_vad_for_detection()
        self.drain_audio_queue()
        self.audio.resume()
        tts_task = asyncio.create_task(play())
        interrupted = False
        trigger_chunks: deque[np.ndarray] = deque(maxlen=config.BARGE_IN_TRIGGER_BUFFER_CHUNKS)
        hit_count = 0

        try:
            while not tts_task.done():
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue

                trigger_chunks.append(chunk)
                if self.is_barge_in_chunk(chunk):
                    hit_count += 1
                else:
                    hit_count = 0

                if hit_count >= config.BARGE_IN_REQUIRED_HITS:
                    interrupted = True
                    self.tts.stop()
                    break

            with contextlib.suppress(asyncio.CancelledError):
                await tts_task
        finally:
            if not interrupted:
                self.audio.stop()

        if not interrupted:
            self.drain_audio_queue()
            self.barge_vad.reset()
            return None

        await self.server.broadcast({"type": "status", "status": "listening"})
        pcm = await self.capture_interruption_segment(list(trigger_chunks))
        self.audio.stop()
        self.drain_audio_queue()
        self.barge_vad.reset()
        return pcm

    async def capture_interruption_segment(self, initial_chunks: list[np.ndarray]) -> Optional[bytes]:
        self.relax_barge_vad_for_capture()
        for chunk in initial_chunks:
            segments = list(self.barge_vad.process(chunk))
            if segments:
                return segments[0]

        deadline = asyncio.get_running_loop().time() + config.BARGE_IN_MAX_CAPTURE_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            segments = list(self.barge_vad.process(chunk))
            if segments:
                return segments[0]
        return self.barge_vad.finish()

    def should_monitor_barge_in(self) -> bool:
        return self.status == Status.SPEAKING and self.mode == Mode.HANDSFREE and config.BARGE_IN_ENABLED

    def should_accept_audio(self) -> bool:
        if self.should_monitor_barge_in():
            return True
        if self.status != Status.LISTENING:
            return False
        if self.mode == Mode.HANDSFREE:
            return True
        return self.mode == Mode.PTT and self.ptt_recording

    def relax_barge_vad_for_capture(self) -> None:
        self.barge_vad.prob_threshold = config.PROB_THRESHOLD
        self.barge_vad.db_threshold = config.DB_THRESHOLD
        self.barge_vad.required_hits = config.REQUIRED_HITS
        self.barge_vad.required_misses = config.BARGE_IN_CAPTURE_REQUIRED_MISSES
        self.barge_vad.require_prob_and_db = False
        self.barge_vad.reset()

    def configure_barge_vad_for_detection(self) -> None:
        self.barge_vad.prob_threshold = config.BARGE_IN_PROB_THRESHOLD
        self.barge_vad.db_threshold = config.BARGE_IN_DB_THRESHOLD
        self.barge_vad.required_hits = config.BARGE_IN_REQUIRED_HITS
        self.barge_vad.required_misses = config.BARGE_IN_REQUIRED_MISSES
        self.barge_vad.require_prob_and_db = True
        self.barge_vad.reset()

    def is_barge_in_chunk(self, chunk: np.ndarray) -> bool:
        prob, db = self.barge_vad.speech_score(chunk)
        return prob >= config.BARGE_IN_PROB_THRESHOLD and db >= config.BARGE_IN_DB_THRESHOLD

    @staticmethod
    def create_barge_vad(model) -> VADEngine:
        return VADEngine(
            prob_threshold=config.BARGE_IN_PROB_THRESHOLD,
            db_threshold=config.BARGE_IN_DB_THRESHOLD,
            required_hits=config.BARGE_IN_REQUIRED_HITS,
            required_misses=config.BARGE_IN_REQUIRED_MISSES,
            min_speech_duration=config.BARGE_IN_MIN_SPEECH_DURATION,
            require_prob_and_db=True,
            model=model,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="claude-voice — voice I/O frontend for Claude Code")
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--port", default=config.SERVER_PORT, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    app = MiruVoiceApp(args.host, args.port, args.verbose)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.closed.set)
    try:
        await app.start()
    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(amain())
