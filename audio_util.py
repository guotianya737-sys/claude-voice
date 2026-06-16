from __future__ import annotations

import numpy as np


def dbfs(chunk: np.ndarray) -> float:
    """返回音频块的分贝值（满量程）。"""
    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(float(np.mean(np.square(audio)))) + 1e-10)
    return 20.0 * float(np.log10(rms)) + 100.0
