from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd

import config


AudioCallback = Callable[[np.ndarray], None]


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        block_size: int = config.BLOCK_SIZE,
        input_device: Optional[int | str] = config.AUDIO_INPUT_DEVICE,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.input_device = input_device
        self._callback: Optional[AudioCallback] = None
        self._stream: Optional[sd.InputStream] = None
        self._paused = threading.Event()
        self._input_name = "未启动"
        self._stream_sample_rate = sample_rate
        self._stream_block_size = block_size

    def set_callback(self, fn: AudioCallback) -> None:
        self._callback = fn

    def start(self) -> None:
        if self._stream is not None:
            return
        device = sd.query_devices(self.input_device, "input")
        self._input_name = str(device["name"])
        self._stream_sample_rate = int(round(float(device["default_samplerate"])))
        self._stream_block_size = max(
            1,
            int(round(self.block_size * self._stream_sample_rate / self.sample_rate)),
        )
        print(
            "audio input:"
            f" {self._input_name} @ {self._stream_sample_rate}Hz"
            f" -> {self.sample_rate}Hz"
        )
        self._stream = sd.InputStream(
            device=self.input_device,
            samplerate=self._stream_sample_rate,
            channels=self.channels,
            blocksize=self._stream_block_size,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def reopen(self, refresh_devices: bool = True) -> None:
        was_paused = self._paused.is_set()
        self.close()
        if refresh_devices:
            self._refresh_device_cache()
        self.start()
        if was_paused:
            self.stop()
        else:
            self.resume()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def input_sample_rate(self) -> int:
        return self._stream_sample_rate

    @property
    def target_sample_rate(self) -> int:
        return self.sample_rate

    def input_state(self) -> dict:
        return {
            "type": "audio_input",
            "name": self.input_name,
            "sample_rate": self.input_sample_rate,
            "target_sample_rate": self.target_sample_rate,
        }

    @staticmethod
    def _refresh_device_cache() -> None:
        with contextlib.suppress(Exception):
            sd._terminate()
            sd._initialize()

    def _on_audio(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            # sounddevice status objects are warnings; the main loop can keep running.
            pass
        if self._paused.is_set() or self._callback is None:
            return
        audio = np.asarray(indata, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[:, 0]
        self._callback(self._resample_to_target(audio))

    def _resample_to_target(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self._stream_sample_rate == self.sample_rate:
            return audio.copy()
        target_size = int(round(audio.size * self.sample_rate / self._stream_sample_rate))
        if target_size <= 0:
            return np.empty(0, dtype=np.float32)
        if audio.size == 1:
            return np.full(target_size, audio[0], dtype=np.float32)

        source_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
        return np.interp(target_x, source_x, audio).astype(np.float32)
