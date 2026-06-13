from __future__ import annotations

from collections import deque
from collections.abc import Generator
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch

import config


class VADState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    INACTIVE = "inactive"


class VADEngine:
    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        prob_threshold: float = config.PROB_THRESHOLD,
        db_threshold: float = config.DB_THRESHOLD,
        required_hits: int = config.REQUIRED_HITS,
        required_misses: int = config.REQUIRED_MISSES,
        min_speech_duration: float = config.MIN_SPEECH_DURATION,
        require_prob_and_db: bool = config.REQUIRE_PROB_AND_DB,
        max_speech_duration: float = config.MAX_SPEECH_SECONDS,
        model: Optional[Any] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.prob_threshold = prob_threshold
        self.db_threshold = db_threshold
        self.required_hits = required_hits
        self.required_misses = required_misses
        self.min_speech_samples = int(sample_rate * min_speech_duration)
        self.max_speech_samples = int(sample_rate * max_speech_duration)
        self.require_prob_and_db = require_prob_and_db
        self.model = model or self._load_model()
        self.last_prob = 0.0
        self.last_db = 0.0
        self.last_is_speech = False
        self.noise_floor_db = config.NOISE_FLOOR_INITIAL_DB
        self.effective_db_threshold = db_threshold
        self.reset()

    def reset(self) -> None:
        self.state = VADState.IDLE
        self.hit_count = 0
        self.miss_count = 0
        self.pre_buffer: deque[np.ndarray] = deque(maxlen=config.PRE_BUFFER_CHUNKS)
        self.speech_chunks: list[np.ndarray] = []

    def process(self, audio_chunk: np.ndarray) -> Generator[bytes, None, None]:
        chunk = self._normalize(audio_chunk)
        is_speech = self._is_speech(chunk)

        if self.state == VADState.IDLE:
            self.pre_buffer.append(chunk)
            if is_speech:
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = VADState.ACTIVE
                    self.speech_chunks = list(self.pre_buffer)
                    self.pre_buffer.clear()
                    self.miss_count = 0
            else:
                self.hit_count = 0
            return

        self.speech_chunks.append(chunk)
        if self._speech_samples() >= self.max_speech_samples:
            segment = self._finish_segment()
            self.reset()
            if segment is not None:
                yield segment
            return

        if is_speech:
            self.miss_count = 0
            if self.state == VADState.INACTIVE:
                self.state = VADState.ACTIVE
            return

        self.miss_count += 1
        if self.state == VADState.ACTIVE and self.miss_count >= self.required_misses:
            self.state = VADState.INACTIVE
            self.miss_count = 0
            return

        if self.state == VADState.INACTIVE and self.miss_count >= self.required_misses:
            segment = self._finish_segment()
            self.reset()
            if segment is not None:
                yield segment

    def _finish_segment(self) -> Optional[bytes]:
        if not self.speech_chunks:
            return None
        audio = np.concatenate(self.speech_chunks)
        if audio.size < self.min_speech_samples:
            return None
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767.0).astype(np.int16).tobytes()

    def finish(self) -> Optional[bytes]:
        segment = self._finish_segment()
        self.reset()
        return segment

    def _is_speech(self, chunk: np.ndarray) -> bool:
        return self.is_speech(chunk)

    def is_speech(self, chunk: np.ndarray) -> bool:
        prob, db = self.speech_score(chunk)
        self.effective_db_threshold = max(self.db_threshold, self.noise_floor_db + config.SPEECH_DB_MARGIN)
        if self.require_prob_and_db:
            result = prob >= self.prob_threshold and db >= self.effective_db_threshold
        else:
            result = prob >= self.prob_threshold or db >= self.effective_db_threshold
        if not result and self.state == VADState.IDLE:
            self.noise_floor_db = (
                config.NOISE_FLOOR_ALPHA * self.noise_floor_db
                + (1.0 - config.NOISE_FLOOR_ALPHA) * db
            )
        self.last_is_speech = bool(result)
        return result

    def speech_score(self, chunk: np.ndarray) -> tuple[float, float]:
        normalized = self._normalize(chunk)
        self.last_prob = self._speech_probability(normalized)
        self.last_db = self._dbfs(normalized)
        return self.last_prob, self.last_db

    def _speech_samples(self) -> int:
        return sum(chunk.size for chunk in self.speech_chunks)

    def _speech_probability(self, chunk: np.ndarray) -> float:
        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            value = self.model(tensor, self.sample_rate)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @staticmethod
    def _dbfs(chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-10)
        return 20.0 * np.log10(rms) + 100.0

    @staticmethod
    def _normalize(audio_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        return np.clip(chunk, -1.0, 1.0)

    @staticmethod
    def _load_model():
        try:
            from silero_vad import load_silero_vad

            return load_silero_vad()
        except Exception:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            return model
