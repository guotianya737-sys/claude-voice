from __future__ import annotations

import site
import sys
from pathlib import Path
from typing import Union

import numpy as np

import config


def _ensure_onnxruntime_dylib() -> None:
    if sys.platform != "darwin":
        return

    dylib_name = "libonnxruntime.1.24.4.dylib"
    site_dirs = [Path(p) for p in site.getsitepackages()]
    site_dirs.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")

    for site_dir in site_dirs:
        source = site_dir / "onnxruntime" / "capi" / dylib_name
        target_dir = site_dir / "sherpa_onnx" / "lib"
        target = target_dir / dylib_name
        if not source.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()  # remove broken symlink from previous run
        if target.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError:
            import shutil

            shutil.copy2(source, target)
        return


_ensure_onnxruntime_dylib()
import sherpa_onnx


class ASREngine:
    def __init__(self) -> None:
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(config.SENSE_VOICE_MODEL),
            tokens=str(config.SENSE_VOICE_TOKENS),
            num_threads=4,
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
