from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np

import config
import hotkey as hotkey_module
from asr_engine import ASREngine
from audio_io import AudioCapture
from audio_util import dbfs  # noqa: F811 — shared with vad.py, used via PiVoiceApp.dbfs
from pi_client import PiClient
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
    END_CHARS = "。！？!?"
    SOFT_CHARS = "，、,"
    BACKTRACK_BREAK_CHARS = "的了是呢吗吧啊呀哦嘛啦喔耶呐么着过和与及"

    def __init__(self, app: "MiruVoiceApp") -> None:
        self.app = app
        self.conversation_id = app.conversation_id
        self.buffer = ""
        self.queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        if self.app.tts.can_prefetch():
            self.worker = asyncio.create_task(self._prefetching_worker())
        else:
            self.worker = asyncio.create_task(self._worker())
        self.interrupted_pcm: Optional[bytes] = None
        self.closed = False
        self._first_chunk_sent = False
        self.displayed_text = ""
        self._emotion_tag_parsed = False
        self._opening_line = ""
        self.app.tts.clear_emotion()

    async def feed(self, token: str) -> None:
        if self.closed or self.interrupted_pcm is not None or self.conversation_id != self.app.conversation_id:
            return
        if not self._emotion_tag_parsed:
            self._opening_line += token
            if not self._opening_may_be_emotion_tag():
                self._emotion_tag_parsed = True
                text = self._opening_line
                self._opening_line = ""
                await self._append_visible_text(text)
                return
            if "\n" in self._opening_line:
                self._emotion_tag_parsed = True
                line, rest = self._opening_line.split("\n", 1)
                visible = self._parse_emotion_line(line)
                text = visible + ("\n" if visible else "") + rest
                self._opening_line = ""
                await self._append_visible_text(text)
            return
        await self._append_visible_text(token)

    async def finish(self) -> Optional[bytes]:
        # Flush any partial opening-line if response never emitted a newline
        if not self._emotion_tag_parsed and self._opening_line:
            self._emotion_tag_parsed = True
            visible = self._parse_emotion_line(self._opening_line)
            self._opening_line = ""
            await self._append_visible_text(visible)
        for chunk in self._pop_ready_chunks(flush=True):
            await self.queue.put(chunk)
        await self.queue.put(None)
        await self.worker
        self.closed = True
        return self.interrupted_pcm

    def _parse_emotion_line(self, line: str) -> str:
        match = config.EMOTION_TAG_RE.match(line)
        if not match:
            return line
        emotion = match.group(1).lower()
        if emotion in {"none", "default"}:
            self.app.tts.clear_emotion()
        else:
            self.app.tts.set_emotion(emotion)
        return match.group(2).lstrip()

    def _opening_may_be_emotion_tag(self) -> bool:
        stripped = self._opening_line.lstrip().lower()
        return not stripped or "[emotion:".startswith(stripped) or stripped.startswith("[emotion:")

    async def _append_visible_text(self, text: str) -> None:
        if not text:
            return
        self.displayed_text += text
        await self.app.server.broadcast({"type": "assistant_token", "text": text})
        self.buffer += text
        for chunk in self._pop_ready_chunks(flush=False):
            await self.queue.put(chunk)

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
        producer.add_done_callback(lambda task: _report_prepare_failure(task, prepared_queue))
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
        accum = ""
        batch_max = config.TTS_BATCH_MAX_CHARS
        batch_timeout = config.TTS_BATCH_TIMEOUT
        if self.app.tts.provider == "indextts":
            # 本地模型慢于实时（RTF≈0.9）：不合并段，靠预取把播放垫平
            batch_max = 1
            batch_timeout = 0.05
        while True:
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=batch_timeout)
            except asyncio.TimeoutError:
                if accum and self.interrupted_pcm is None:
                    prepared = await self.app.tts.prepare(accum)
                    if prepared is not None:
                        await prepared_queue.put(prepared)
                    accum = ""
                continue
            if chunk is None:
                if accum and self.interrupted_pcm is None:
                    prepared = await self.app.tts.prepare(accum)
                    if prepared is not None:
                        await prepared_queue.put(prepared)
                    accum = ""
                await prepared_queue.put(None)
                return
            if self.interrupted_pcm is not None:
                continue
            accum += chunk
            if len(accum) >= batch_max or batch_max <= 1:
                prepared = await self.app.tts.prepare(accum)
                if prepared is not None:
                    await prepared_queue.put(prepared)
                accum = ""

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
                self._first_chunk_sent = True
        return chunks

    def _find_cut(self, flush: bool) -> Optional[int]:
        if flush:
            return len(self.buffer)

        if self.app.tts.provider == "indextts" and config.INDEXTTS_SINGLE_REQUEST:
            # 本地模型 RTF≈1：切成多段必然出现静音缝（生成第2段时第1段已播完），
            # 所以整段一次合成——延迟高但中途不断。
            return None
        if self.app.tts.provider == "indextts":
            # 第一段尽量短，先把声音放出来；之后放大段减少请求次数
            if self._first_chunk_sent:
                max_chars = config.INDEXTTS_CHUNK_MAX_CHARS
                min_chars = config.INDEXTTS_CHUNK_MIN_CHARS
            else:
                max_chars = config.INDEXTTS_FIRST_CHUNK_MAX_CHARS
                min_chars = config.INDEXTTS_FIRST_CHUNK_MIN_CHARS
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
    def __init__(self, host: str, port: int, verbose: bool = False, mode: str = "ptt") -> None:
        self.host = host
        self.port = port
        self.verbose = verbose
        self.mode = Mode.HANDSFREE if str(mode).lower() == "handsfree" else Mode.PTT
        self.status = Status.IDLE
        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=config.AUDIO_QUEUE_MAX_CHUNKS)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.audio_processor_ready = asyncio.Event()
        self.ptt_chunks: list[np.ndarray] = []
        self.ptt_recording = False
        self.ptt_barge_in_event = asyncio.Event()
        self.ptt_barge_in_future: Optional[asyncio.Future[Optional[bytes]]] = None
        self.conversation_id = 0
        self.closed = asyncio.Event()
        self.background_tasks: set[asyncio.Task] = set()
        self.turn_lock = asyncio.Lock()
        self.last_meter_at = 0.0

        self.audio = AudioCapture()
        self.vad: Optional[VADEngine] = None
        self.barge_vad: Optional[VADEngine] = None
        self.asr = ASREngine()
        self.tts = TTSEngine()
        self.pi = PiClient()
        self.server = VoiceServer(self.handle_command, self.current_state)
        self.hotkey: Optional[hotkey_module.HoldHotkey] = None

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.audio.set_callback(self._on_audio)
        processor = asyncio.create_task(self.process_audio())
        processor.add_done_callback(self.report_processor_error)
        self.audio.start()
        await self.server.start(self.host, self.port)
        self.audio.stop()
        await self.set_status(Status.IDLE)
        self.start_hotkey()
        if self.mode == Mode.HANDSFREE:
            try:
                self._ensure_vad()
                self.audio.resume()
                await self.set_status(Status.LISTENING)
            except RuntimeError as exc:
                print(f"handsfree unavailable, falling back to ptt: {exc}")
                self.mode = Mode.PTT
                self.audio.stop()
        self.print_banner()
        await self.closed.wait()
        processor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await processor

    async def close(self) -> None:
        self.closed.set()
        self.audio.close()
        self.tts.stop()
        self.stop_hotkey()
        self.pi.close()
        for task in list(self.background_tasks):
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        await self.server.stop()

    async def handle_command(self, payload: dict) -> None:
        command = payload.get("type")
        if command == "start_listening":
            if self.mode == Mode.PTT:
                self.audio.stop()
                self.drain_audio_queue()
                self.reset_vad()
                self.ptt_recording = False
                await self.set_status(Status.IDLE)
                await self.server.broadcast({"type": "mode", "mode": self.mode.value})
                return
            self.mode = Mode.HANDSFREE
            try:
                self._ensure_vad()
            except RuntimeError as exc:
                self.mode = Mode.PTT
                print(f"handsfree unavailable: {exc}")
                await self.server.broadcast({"type": "error", "message": str(exc)})
                await self.server.broadcast({"type": "mode", "mode": self.mode.value})
                return
            self.drain_audio_queue()
            self.reset_vad()
            self.audio.resume()
            await self.set_status(Status.LISTENING)
            await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "stop_listening":
            self.audio.stop()
            self.drain_audio_queue()
            self.reset_vad()
            await self.set_status(Status.IDLE)
        elif command == "switch_mode":
            mode = payload.get("mode")
            if mode in {Mode.HANDSFREE.value, Mode.PTT.value}:
                self.mode = Mode(mode)
                self.reset_vad()
                self.ptt_chunks.clear()
                self.ptt_recording = False
                self.drain_audio_queue()
                if self.mode == Mode.HANDSFREE:
                    try:
                        self._ensure_vad()
                    except RuntimeError as exc:
                        self.mode = Mode.PTT
                        print(f"handsfree unavailable: {exc}")
                        await self.server.broadcast({"type": "error", "message": str(exc)})
                        await self.server.broadcast({"type": "mode", "mode": self.mode.value})
                        return
                    self.audio.resume()
                    await self.set_status(Status.LISTENING)
                else:
                    self.audio.stop()
                    await self.set_status(Status.IDLE)
                await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "push_to_talk_start":
            is_barge_in = self.status == Status.SPEAKING
            self.mode = Mode.PTT
            self.ptt_chunks.clear()
            self.drain_audio_queue()
            self.ptt_recording = True
            self.update_audio_processor_ready()
            if is_barge_in:
                if self.ptt_barge_in_future is None or self.ptt_barge_in_future.done():
                    self.ptt_barge_in_future = asyncio.get_running_loop().create_future()
                self.ptt_barge_in_event.set()
                self.tts.stop()
            self.audio.resume()
            if not is_barge_in:
                await self.set_status(Status.LISTENING)
            await self.server.broadcast({"type": "mode", "mode": self.mode.value})
        elif command == "push_to_talk_end":
            self.ptt_recording = False
            if self.ptt_barge_in_future is not None and not self.ptt_barge_in_future.done():
                pcm = None
                if self.ptt_chunks:
                    audio = np.concatenate(self.ptt_chunks)
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                self.ptt_chunks.clear()
                self.ptt_barge_in_future.set_result(pcm)
                self.update_audio_processor_ready()
                return
            if self.ptt_chunks:
                audio = np.concatenate(self.ptt_chunks)
                self.ptt_chunks.clear()
                await self.handle_segment((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            else:
                self.audio.stop()
                self.drain_audio_queue()
            self.update_audio_processor_ready()
            await self.set_status(Status.IDLE)
        elif command == "switch_tts":
            provider = payload.get("provider")
            self.tts.set_provider(provider)
            await self.server.broadcast({"type": "tts_provider", "provider": self.tts.provider})
        elif command == "new_conversation":
            await self.start_new_conversation()
        elif command == "refresh_audio_input":
            self.audio.reopen()
            self.drain_audio_queue()
            self.reset_vad()
            await self.server.broadcast(self.audio.input_state())
            if self.mode == Mode.HANDSFREE and self.status == Status.IDLE:
                self.audio.resume()
                await self.set_status(Status.LISTENING)
        elif command == "text_input":
            text = payload.get("text", "").strip()
            if text and self.has_text_signal(text):
                self.track_background_task(asyncio.create_task(self.handle_text_message(text)))

    async def process_audio(self) -> None:
        while True:
            await self.audio_processor_ready.wait()
            chunk = await self.audio_queue.get()
            if self.mode == Mode.PTT:
                if self.ptt_recording:
                    self.ptt_chunks.append(chunk)
                continue
            if self.status != Status.LISTENING:
                continue
            if self.vad is None:
                continue
            segments = list(self.vad.process(chunk))
            await self.maybe_broadcast_audio_meter()
            for segment in segments:
                await self.handle_segment(segment)

    async def cue(self, kind: str) -> None:
        if not config.HANDSFREE_CUE or not config.IS_WINDOWS:
            return
        try:
            import winsound

            alias = "SystemAsterisk" if kind == "heard" else "SystemHand"
            winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception:  # noqa: BLE001
            pass

    async def handle_segment(self, pcm: bytes) -> None:
        next_pcm: Optional[bytes] = pcm
        while next_pcm is not None:
            next_pcm = await self.handle_single_segment(next_pcm)

    async def handle_text_message(self, text: str) -> None:
        next_pcm = await self.handle_user_text(
            text,
            timeout_text="pi 响应超时了，请稍后再说一次。",
        )
        if next_pcm is not None:
            await self.handle_segment(next_pcm)

    async def handle_single_segment(self, pcm: bytes) -> Optional[bytes]:
        conversation_id = self.conversation_id
        print(f"[turn] 收到一段 {len(pcm)/2/config.SAMPLE_RATE:.1f}s，开始识别", flush=True)
        await self.cue("heard")
        self.audio.stop()
        self.drain_audio_queue()
        self.reset_vad()
        await self.set_status(Status.THINKING)
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self.asr.transcribe, pcm),
                timeout=config.ASR_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            print(f"asr failed: {type(exc).__name__}: {exc}")
            text = ""
        print(f"[asr] {text!r} ({len(pcm) / 2 / config.SAMPLE_RATE:.1f}s)")
        if not text:
            await self.resume_after_turn()
            return None
        if not self.should_accept_asr_text(text):
            print(f"ignored asr noise: {text!r}")
            await self.resume_after_turn()
            return None
        if conversation_id != self.conversation_id:
            return None

        return await self.handle_user_text(
            text,
            timeout_text="pi 响应超时了，可能是服务正在重试或暂时拥堵。请稍后再说一次。",
        )

    async def handle_user_text(self, text: str, timeout_text: str) -> Optional[bytes]:
        async with self.turn_lock:
            conversation_id = self.conversation_id
            self.tts.stop()
            self.audio.stop()
            self.drain_audio_queue()
            self.reset_vad()
            await self.set_status(Status.THINKING)
            await self.server.broadcast({"type": "user_msg", "text": text})
            speaker = StreamingReplySpeaker(self)
            try:
                reply = await asyncio.wait_for(
                    self.pi.send(text, speaker.feed, self.on_pi_event),
                    timeout=config.PI_RESPONSE_TIMEOUT,
                )
                interrupted_pcm = await speaker.finish()
            except asyncio.TimeoutError as exc:
                print(f"pi failed: {type(exc).__name__}: {exc}")
                self.pi.close()
                self.tts.stop()
                # Signal speaker to drain immediately — the subprocess is dead,
                # no more tokens will arrive.
                interrupted_pcm = await speaker.finish()
                error_text = speaker.displayed_text or timeout_text
                if conversation_id == self.conversation_id:
                    await self.server.broadcast({"type": "assistant_done", "text": error_text})
                await self.resume_after_turn()
                return interrupted_pcm
            except Exception as exc:
                print(f"pi failed: {type(exc).__name__}: {exc}")
                self.pi.close()
                self.tts.stop()
                interrupted_pcm = await speaker.finish()
                error_text = speaker.displayed_text or f"pi 这次没接上：{type(exc).__name__}。请稍后再试一次。"
                if conversation_id == self.conversation_id:
                    await self.server.broadcast({"type": "assistant_done", "text": error_text})
                await self.resume_after_turn()
                return interrupted_pcm
            if conversation_id != self.conversation_id:
                return None
            await self.server.broadcast({"type": "assistant_done", "text": speaker.displayed_text or reply})
            if interrupted_pcm is not None:
                await self.server.broadcast({"type": "status", "status": "thinking"})
                return interrupted_pcm
            await asyncio.sleep(config.TTS_TAIL_GUARD)
            await self.resume_after_turn()
            return None

    def track_background_task(self, task: asyncio.Task) -> None:
        self.background_tasks.add(task)

        def done_callback(done: asyncio.Task) -> None:
            self.background_tasks.discard(done)
            self.report_processor_error(done)

        task.add_done_callback(done_callback)

    async def start_new_conversation(self) -> None:
        self.conversation_id += 1
        self.tts.stop()
        await self.pi.reset()
        self.ptt_chunks.clear()
        self.ptt_recording = False
        self.ptt_barge_in_event.clear()
        self.ptt_barge_in_future = None
        self.drain_audio_queue()
        self.reset_vad()
        await self.server.broadcast({"type": "conversation_reset"})
        if self.mode == Mode.HANDSFREE:
            self.audio.resume()
            await self.set_status(Status.LISTENING)
        else:
            self.audio.stop()
            await self.set_status(Status.IDLE)

    async def resume_after_turn(self) -> None:
        self.drain_audio_queue()
        self.reset_vad()
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
        self.update_audio_processor_ready()
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
        if not config.BARGE_IN_ENABLED:
            await play()
            return None
        if self.mode == Mode.PTT:
            return await self.play_with_ptt_barge_in(play)
        if self.mode != Mode.HANDSFREE:
            await play()
            return None

        self.configure_barge_vad_for_detection()
        self.drain_audio_queue()
        self.audio.resume()
        print("[play] 开始朗读（免提，监听是否被打断）", flush=True)
        tts_task = asyncio.create_task(play())
        interrupted = False
        trigger_chunks: deque[np.ndarray] = deque(maxlen=config.BARGE_IN_TRIGGER_BUFFER_CHUNKS)
        hit_count = 0
        play_started_at = asyncio.get_running_loop().time()

        try:
            while not tts_task.done():
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue

                trigger_chunks.append(chunk)
                if (
                    asyncio.get_running_loop().time() - play_started_at < config.BARGE_IN_GUARD_SECONDS
                ):
                    hit_count = 0
                    continue
                if self.is_barge_in_chunk(chunk):
                    hit_count += 1
                else:
                    hit_count = 0

                if hit_count >= config.BARGE_IN_REQUIRED_HITS:
                    interrupted = True
                    print(
                        f"[play] 被打断：prob={self.barge_vad.last_prob:.2f} "
                        f"db={self.barge_vad.last_db:.1f}（阈值 {config.BARGE_IN_PROB_THRESHOLD}/"
                        f"{config.BARGE_IN_DB_THRESHOLD}）",
                        flush=True,
                    )
                    self.tts.stop()
                    break

            with contextlib.suppress(asyncio.CancelledError):
                await tts_task
        finally:
            if not interrupted:
                self.audio.stop()

        if not interrupted:
            print("[play] 朗读结束", flush=True)
            self.drain_audio_queue()
            self.reset_vad()
            return None

        await self.server.broadcast({"type": "status", "status": "listening"})
        pcm = await self.capture_interruption_segment(list(trigger_chunks))
        self.audio.stop()
        self.drain_audio_queue()
        self.reset_vad()
        return pcm

    async def play_with_ptt_barge_in(self, play) -> Optional[bytes]:
        if self.ptt_barge_in_event.is_set() and self.ptt_barge_in_future is not None:
            if self.ptt_barge_in_future.done():
                pcm = self.ptt_barge_in_future.result()
                self.ptt_barge_in_event.clear()
                self.ptt_barge_in_future = None
                self.update_audio_processor_ready()
                return pcm
        has_pending_barge_in = (
            self.ptt_barge_in_event.is_set()
            and self.ptt_barge_in_future is not None
            and not self.ptt_barge_in_future.done()
        )
        if not has_pending_barge_in:
            self.ptt_barge_in_event.clear()
            self.ptt_barge_in_future = None
            self.update_audio_processor_ready()
        tts_task = asyncio.create_task(play())
        barge_task = asyncio.create_task(self.ptt_barge_in_event.wait())
        try:
            done, _ = await asyncio.wait(
                {tts_task, barge_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if tts_task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    await tts_task
                return None

            self.tts.stop()
            future = self.ptt_barge_in_future
            if future is None:
                return None
            await self.server.broadcast({"type": "status", "status": "listening"})
            try:
                return await asyncio.wait_for(future, timeout=config.BARGE_IN_MAX_CAPTURE_SECONDS)
            except asyncio.TimeoutError:
                return None
        finally:
            barge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await barge_task
            if not tts_task.done():
                tts_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tts_task
            self.ptt_barge_in_event.clear()
            self.ptt_barge_in_future = None

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
        if self.mode == Mode.PTT and self.ptt_recording:
            return True
        if self.should_monitor_barge_in():
            return True
        return self.status == Status.LISTENING and self.mode == Mode.HANDSFREE

    def relax_barge_vad_for_capture(self) -> None:
        self.barge_vad.prob_threshold = config.PROB_THRESHOLD
        self.barge_vad.db_threshold = config.DB_THRESHOLD
        self.barge_vad.required_hits = config.REQUIRED_HITS
        self.barge_vad.required_misses = config.BARGE_IN_CAPTURE_REQUIRED_MISSES
        self.barge_vad.require_prob_and_db = False
        self.reset_vad()

    def configure_barge_vad_for_detection(self) -> None:
        self.barge_vad.prob_threshold = config.BARGE_IN_PROB_THRESHOLD
        self.barge_vad.db_threshold = config.BARGE_IN_DB_THRESHOLD
        self.barge_vad.required_hits = config.BARGE_IN_REQUIRED_HITS
        self.barge_vad.required_misses = config.BARGE_IN_REQUIRED_MISSES
        self.barge_vad.require_prob_and_db = True
        self.reset_vad()

    def is_barge_in_chunk(self, chunk: np.ndarray) -> bool:
        prob, db = self.barge_vad.speech_score(chunk)
        return prob >= config.BARGE_IN_PROB_THRESHOLD and db >= config.BARGE_IN_DB_THRESHOLD

    @staticmethod
    def should_accept_asr_text(text: str) -> bool:
        normalized = config.ASR_NOISE_TEXT_RE.sub("", text).lower()
        if normalized in config.ASR_ALLOWED_SHORT_TEXTS:
            return True
        if len(normalized) < config.ASR_MIN_NORMALIZED_CHARS:
            return False
        return normalized not in config.ASR_IGNORED_NORMALIZED_TEXTS

    @staticmethod
    def has_text_signal(text: str) -> bool:
        return bool(config.ASR_NOISE_TEXT_RE.sub("", text))

    def update_audio_processor_ready(self) -> None:
        if self.status == Status.LISTENING or (self.mode == Mode.PTT and self.ptt_recording):
            self.audio_processor_ready.set()
        else:
            self.audio_processor_ready.clear()

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

    # -- 免提 / 热键 / 状态辅助 -------------------------------------------
    def reset_vad(self) -> None:
        """VAD 是懒加载的：只用按键说话时根本没有实例。"""
        if self.vad is not None:
            self.vad.reset()
        if self.barge_vad is not None:
            self.barge_vad.reset()

    def _ensure_vad(self) -> None:
        if self.vad is not None:
            return
        try:
            self.vad = VADEngine()
        except Exception as exc:  # noqa: BLE001 - torch 缺失会给很长的报错
            raise RuntimeError(
                "免提模式需要 silero VAD。在 .venv 里装：pip install torch silero-vad（按键说话不需要）"
            ) from exc
        self.barge_vad = self.create_barge_vad(self.vad.model)

    def start_hotkey(self) -> None:
        if not config.PTT_HOTKEY or self.loop is None:
            return
        hotkey = hotkey_module.HoldHotkey(
            config.PTT_HOTKEY,
            self.ptt_press,
            self.ptt_release,
            self.loop,
        )
        if not hotkey.available:
            print(f"热键未识别：{config.PTT_HOTKEY}（支持 F1-F24 / SPACE / 字母数字 / CTRL+键）；Web 界面上的按钮仍可用")
            return
        hotkey.start()
        self.hotkey = hotkey

    def stop_hotkey(self) -> None:
        if self.hotkey is not None:
            self.hotkey.stop()
            self.hotkey = None

    def print_banner(self) -> None:
        mode_text = "免提常听（VAD 自动断句）" if self.mode == Mode.HANDSFREE else "按键说话"
        lines = [
            f"pi-voice 已启动：http://{self.host}:{self.port}/  [模式：{mode_text}]",
            f"麦克风：{self.audio.input_name}",
            f"TTS：{self.tts.provider}" + self._tts_detail(),
        ]
        if self.mode == Mode.HANDSFREE:
            lines.append("一直听着，说完会自动接话；朗读中按 F9 = 打断")
        elif self.hotkey is not None:
            lines.append(f"按住 [{self.hotkey.describe()}] 说话，松开就发出去；说话中途按住可以打断")
        print("\n".join(lines))

    def _tts_detail(self) -> str:
        if self.tts.provider == "indextts":
            state = "已连接" if self.tts.indextts.available() else "未连上（会回落本地 SAPI）"
            return f"（本地 IndexTTS @ {config.INDEXTTS_URL}，{state}）"
        return f"（{self.tts.sapi.voice or '自动选音色'}）"

    async def ptt_press(self) -> None:
        if self.mode == Mode.HANDSFREE:
            # 常听模式下热键没意义，按一下当"打断"
            if self.status == Status.SPEAKING:
                self.tts.stop()
            return
        if config.PTT_BEEP and config.IS_WINDOWS:
            with contextlib.suppress(Exception):
                import winsound

                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        await self.handle_command({"type": "push_to_talk_start"})

    async def ptt_release(self) -> None:
        if self.mode == Mode.HANDSFREE:
            return
        await self.handle_command({"type": "push_to_talk_end"})

    async def on_pi_event(self, kind: str, record: dict) -> None:
        await self.server.broadcast(
            {
                "type": "tool",
                "name": record.get("toolName", ""),
                "phase": "start" if kind == "tool_execution_start" else "end",
                "is_error": bool(record.get("isError")),
            }
        )


def _report_prepare_failure(task: asyncio.Task, queue: "asyncio.Queue[Optional[PreparedSpeech]]") -> None:
    """TTS 预处理任务要是挂了，得把它报出来并叫醒消费者，否则整轮对话会死等。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    print(f"tts prepare failed: {type(exc).__name__}: {exc}")
    with contextlib.suppress(Exception):
        queue.put_nowait(None)


def _load_wav_as_pcm(path: str) -> bytes:
    """读 wav（任意位深/采样率/声道）→ 16k 单声道 int16 PCM，给自检用。"""
    import wave

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"不支持的 wav 位深: {width * 8}bit")
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    if rate != config.SAMPLE_RATE and audio.size > 1:
        target = int(round(audio.size * config.SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0.0, 1.0, num=target, endpoint=False),
            np.linspace(0.0, 1.0, num=audio.size, endpoint=False),
            audio,
        )
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def run_selftest(app: "MiruVoiceApp", wav_path: str) -> int:
    """不开服务器、不用麦克风，直接跑 ASR → pi → TTS 全链路。"""
    print(f"[自检] 输入音频：{wav_path}")
    pcm = await asyncio.to_thread(_load_wav_as_pcm, wav_path)
    seconds = len(pcm) / 2 / config.SAMPLE_RATE
    print(f"[自检] 音频长度 {seconds:.2f}s")
    text = await asyncio.to_thread(app.asr.transcribe, pcm)
    print(f"[自检] ASR → {text!r}")
    if not text:
        print("[自检] ASR 没出字，链路停在识别这一步")
        return 2
    app.conversation_id += 1
    await app.handle_text_message(text)
    print("[自检] 全链路完成")
    return 0


async def run_miccheck(seconds: float) -> int:
    """录一段，报告音量与 VAD 概率 —— 用来判断"到底有没有收到人声"。"""
    chunks: list[np.ndarray] = []

    def collect(chunk: np.ndarray) -> None:
        if len(chunks) < 100000:
            chunks.append(chunk.copy())

    capture = AudioCapture()
    capture.set_callback(collect)
    capture.start()
    print(f"请对着麦克风连续说话 {seconds:.0f} 秒 ...", flush=True)
    await asyncio.sleep(seconds)
    capture.close()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        print("没采到任何数据：麦克风没打开")
        return 1
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(float(np.mean(np.square(audio)))))
    print(f"麦克风：{capture.input_name}  采集 {audio.size / config.SAMPLE_RATE:.1f}s")
    print(f"峰值 {peak:.4f} | RMS {rms:.5f} | 音量 {dbfs(audio):.1f} dB（判定阈值 {config.DB_THRESHOLD}）")
    print("-> " + ("有声音 ✔" if peak > 0.02 else "几乎静音 ✘（检查耳机静音键 / 系统输入音量 / 换设备）"))
    try:
        vad = VADEngine()
        probs = [
            vad.speech_score(audio[i : i + config.BLOCK_SIZE])[0]
            for i in range(0, max(1, audio.size - config.BLOCK_SIZE), config.BLOCK_SIZE)
        ]
        print(f"VAD 最大 prob {max(probs):.2f}（判定阈值 {config.PROB_THRESHOLD}）")
    except Exception as exc:  # noqa: BLE001
        print(f"VAD 不可用（免提模式会退回按键）：{type(exc).__name__}: {exc}")
    return 0


def print_devices() -> None:
    import sounddevice as sd

    print("可用的音频设备（* 为默认输入）：")
    default_input = None
    with contextlib.suppress(Exception):
        default_input = sd.default.device[0]
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) <= 0:
            continue
        mark = "*" if index == default_input else " "
        print(
            f"{mark} [{index}] {device['name']} "
            f"输入通道={device['max_input_channels']} 默认采样率={int(device['default_samplerate'])}"
        )


def print_check() -> int:
    from pathlib import Path

    problems = 0
    node = Path(config.PI_NODE)
    launcher = Path(config.PI_LAUNCHER)
    print(f"pi node     : {'OK' if node.exists() else '缺失'} {node}")
    print(f"pi launcher : {'OK' if launcher.exists() else '缺失'} {launcher}")
    if not (node.exists() and launcher.exists()):
        problems += 1
    missing = models_missing()
    print(f"ASR 模型    : {'OK' if not missing else '缺 ' + ', '.join(missing)} {config.MODEL_DIR}")
    if missing:
        problems += 1
    try:
        import sherpa_onnx  # noqa: F401

        print("sherpa-onnx : OK")
    except Exception as exc:  # noqa: BLE001
        print(f"sherpa-onnx : 缺失 ({exc})")
        problems += 1
    tts = TTSEngine()
    voices = tts.sapi.list_voices()
    print(f"SAPI 音色   : {', '.join(voices) if voices else '不可用'}")
    if not voices:
        problems += 1
    try:
        import sounddevice as sd

        info = sd.query_devices(None, "input")
        print(f"默认麦克风  : {info['name']} @ {int(info['default_samplerate'])}Hz")
    except Exception as exc:  # noqa: BLE001
        print(f"默认麦克风  : 打不开 ({exc})")
        problems += 1
    print("结论：" + ("全部就绪" if problems == 0 else f"{problems} 项待处理"))
    return 0 if problems == 0 else 1


def models_missing() -> list[str]:
    import models

    return models.missing_files()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pi-voice — 语音 I/O 前端（Windows / pi）")
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--port", default=config.SERVER_PORT, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--selftest", metavar="WAV", default=None, help="跑一遍 ASR → pi → TTS 自检后退出")
    parser.add_argument("--list-devices", action="store_true", help="列出可用音频输入设备")
    parser.add_argument("--miccheck", type=float, nargs="?", const=5.0, default=None,
                        metavar="SECONDS", help="录一段并报告音量/VAD 概率（默认 5 秒）")
    parser.add_argument("--check", action="store_true", help="检查依赖、模型、音色、麦克风")
    parser.add_argument("--no-hotkey", action="store_true", help="本次不启用全局热键")
    parser.add_argument("--mode", choices=["ptt", "handsfree"], default=None,
                        help="ptt=按住说话；handsfree=一直聆听自动断句（需要 torch + silero-vad）")
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    if args.list_devices:
        print_devices()
        return
    if args.miccheck:
        raise SystemExit(await run_miccheck(args.miccheck))
    if args.check:
        raise SystemExit(print_check())
    app = MiruVoiceApp(args.host, args.port, args.verbose, args.mode or config.VOICE_MODE)
    if args.no_hotkey:
        app.start_hotkey = lambda: None  # type: ignore[method-assign]
    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, app.closed.set)
    try:
        if args.selftest:
            raise SystemExit(await run_selftest(app, args.selftest))
        await app.start()
    except KeyboardInterrupt:
        pass
    finally:
        await app.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain())
