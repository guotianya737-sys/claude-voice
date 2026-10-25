from __future__ import annotations

from typing import Union

import numpy as np

import config
import models


class ASREngine:
    def __init__(self, ensure_model: bool = True) -> None:
        if ensure_model:
            models.ensure_model()
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(config.SENSE_VOICE_MODEL),
            tokens=str(config.SENSE_VOICE_TOKENS),
            num_threads=config.ASR_NUM_THREADS,
            use_itn=True,
            provider="cpu",
        )

    def transcribe(self, audio: Union[np.ndarray, bytes]) -> str:
        audio_np = self._to_float32(audio)
        if audio_np.size == 0:
            return ""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, audio_np)
        self.recognizer.decode_streams([stream])
        return stream.result.text.strip()

    @staticmethod
    def _to_float32(audio: Union[np.ndarray, bytes]) -> np.ndarray:
        if isinstance(audio, bytes):
            return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        arr = np.asarray(audio)
        if arr.dtype == np.int16:
            return arr.astype(np.float32) / 32768.0
        return arr.astype(np.float32).reshape(-1)
