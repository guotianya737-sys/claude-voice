"""SenseVoice ASR 模型下载（首次运行自动准备，约 230 MB）。

优先 hf-mirror.com，失败回退 huggingface.co；文件级下载，支持断点续传。
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _url_for(mirror: str, filename: str) -> str:
    return f"{mirror.rstrip('/')}/{_model_repo()}/resolve/main/{filename}"


def _model_repo() -> str:
    import config

    return config.MODEL_HF_REPO


def missing_files() -> list[str]:
    import config

    return [name for name in config.MODEL_FILES if not (config.MODEL_DIR / name).exists()]


def ensure_model(quiet: bool = False) -> None:
    """确保模型文件齐全，缺什么下什么。"""
    import config

    missing = missing_files()
    if not missing:
        return
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for mirror in config.MODEL_MIRRORS:
        try:
            for filename in list(missing):
                _download(_url_for(mirror, filename), config.MODEL_DIR / filename, quiet=quiet)
            if not missing_files():
                return
        except Exception as exc:  # noqa: BLE001 - 换镜像重试
            last_error = exc
            if not quiet:
                print(f"模型下载失败（{mirror}）：{type(exc).__name__}: {exc}")
            continue
    raise RuntimeError(
        f"无法下载 SenseVoice 模型：{last_error}。可手动把 {config.MODEL_FILES} 放到 {config.MODEL_DIR}"
    )


def _download(url: str, target: Path, quiet: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0

    request = urllib.request.Request(url, headers={"User-Agent": "pi-voice/0.1"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # 已经下完
            partial.replace(target)
            return
        raise

    total = int(response.headers.get("Content-Length") or 0) + existing
    mode = "ab" if response.status == 206 and existing else "wb"
    if mode == "wb":
        existing = 0

    if not quiet:
        print(f"下载 {target.name} …（{_human(total) if total else '大小未知'}）")
    downloaded = existing
    last_report = 0.0
    import time

    with partial.open(mode) as handle:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if not quiet and total:
                now = time.time()
                if now - last_report > 0.5:
                    last_report = now
                    percent = downloaded * 100 / total
                    sys.stdout.write(f"\r  {percent:5.1f}%  {_human(downloaded)}/{_human(total)}")
                    sys.stdout.flush()
    if not quiet:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    if total and downloaded < total:
        raise IOError(f"{target.name} 下载不完整（{downloaded}/{total}）")
    shutil.move(str(partial), str(target))


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"
