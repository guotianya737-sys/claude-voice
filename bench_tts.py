"""Benchmark TTS latency for all four engines."""
from __future__ import annotations

import asyncio
import time

import config
from tts_engine import BailianTTS, SiliconFlowTTS, _bailian_audio_format

TEST_TEXT = "你好，今天天气真不错，适合出去走走。"

SAY_TEXT = "测试文本"  # say is fast, just test process spawn


async def bench_say() -> dict:
    """Test macOS say TTS latency."""
    import subprocess

    start = time.perf_counter()
    proc = subprocess.Popen(
        ["say", "-v", config.SAY_VOICE, "-r", str(config.SAY_RATE), SAY_TEXT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time_to_spawn = time.perf_counter() - start
    proc.wait()
    total = time.perf_counter() - start
    return {
        "engine": "macOS say",
        "time_to_spawn_s": round(time_to_spawn, 4),
        "total_s": round(total, 4),
        "note": "本地进程，无网络",
    }


async def bench_siliconflow(sf: SiliconFlowTTS, label: str) -> dict:
    """Test SiliconFlow HTTP API TTS latency."""
    if not sf.configured:
        return {"engine": label, "status": "未配置 API Key"}

    start = time.perf_counter()

    payload = {
        "input": TEST_TEXT,
        "response_format": sf.response_format,
        "sample_rate": sf.sample_rate,
        "stream": sf.stream,
        "speed": sf.speed,
        "gain": sf.gain,
        "model": sf.default_model,
        "voice": sf.default_voice,
    }
    headers = {
        "Authorization": f"Bearer {sf.api_key}",
        "Content-Type": "application/json",
    }

    import aiohttp
    timeout = aiohttp.ClientTimeout(total=30)
    ttfb = None
    total = None

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(sf.api_url, json=payload, headers=headers) as resp:
                ttfb = time.perf_counter() - start
                if resp.status >= 400:
                    body = await resp.text()
                    return {"engine": label, "status": f"HTTP {resp.status}: {body[:100]}"}
                data = await resp.read()
                total = time.perf_counter() - start
    except Exception as e:
        return {"engine": label, "status": f"异常: {e}"}

    return {
        "engine": label,
        "ttfb_s": round(ttfb, 3),
        "total_s": round(total, 3),
        "audio_size_bytes": len(data),
        "note": "HTTP POST → 流式返回",
    }


async def bench_bailian() -> dict:
    """Test Bailian streaming TTS first-audio and total latency."""
    bt = BailianTTS.from_config()
    if not bt.configured:
        return {"engine": "Bailian CosyVoice v3", "status": "未配置 API Key"}

    start = time.perf_counter()
    first_audio_at = None
    total_audio_bytes = 0
    error = None

    try:
        import dashscope
        from dashscope.audio.tts_v2 import ResultCallback, SpeechSynthesizer

        dashscope.api_key = bt.api_key
        dashscope.base_websocket_api_url = bt.websocket_api_url

        class Callback(ResultCallback):
            def on_data(self, data: bytes) -> None:
                nonlocal first_audio_at, total_audio_bytes
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                total_audio_bytes += len(data)

            def on_error(self, message) -> None:
                nonlocal error
                error = str(message)

        synth = SpeechSynthesizer(
            model=bt.model,
            voice=bt.voice,
            format=_bailian_audio_format(bt.audio_format),
            volume=bt.volume,
            speech_rate=bt.speech_rate,
            pitch_rate=bt.pitch_rate,
            callback=Callback(),
        )
        synth.streaming_call(TEST_TEXT)
        synth.streaming_complete(config.BAILIAN_CALL_TIMEOUT_MS)
    except Exception as e:
        return {"engine": "Bailian CosyVoice v3 Flash", "status": f"stream 异常: {e}"}
    total = time.perf_counter() - start
    if error:
        return {"engine": "Bailian CosyVoice v3 Flash", "status": f"stream 错误: {error[:120]}"}
    if first_audio_at is None or total_audio_bytes == 0:
        return {"engine": "Bailian CosyVoice v3 Flash", "status": "未收到音频数据"}
    return {
        "engine": "Bailian CosyVoice v3 Flash",
        "ttfb_s": round(first_audio_at - start, 3),
        "total_s": round(total, 3),
        "audio_size_bytes": total_audio_bytes,
        "note": "回调流式输出",
    }


async def main():
    print("=" * 60)
    print("  claude-voice TTS 延迟测试")
    print(f"  测试文本: \"{TEST_TEXT}\"")
    print("=" * 60)
    print()

    # Test each engine 3 times for stable results
    sf = SiliconFlowTTS.from_config("siliconflow")
    sf_moss = SiliconFlowTTS.from_config("siliconflow_moss")

    engines = [
        ("macOS say", lambda: bench_say()),
        ("SiliconFlow CosyVoice2", lambda: bench_siliconflow(sf, "SiliconFlow CosyVoice2")),
        ("SiliconFlow MOSS", lambda: bench_siliconflow(sf_moss, "SiliconFlow MOSS")),
        ("Bailian CosyVoice v3 Flash", lambda: bench_bailian()),
    ]

    results = {}
    for name, fn in engines:
        print(f"\n--- {name} (warm-up) ---")
        result = await fn()
        print(f"  warm-up: {result}")

        runs = []
        ttfb_runs = []
        for i in range(3):
            print(f"  run {i + 1}/3...", end=" ", flush=True)
            result = await fn()
            if "total_s" in result:
                runs.append(result["total_s"])
                if "ttfb_s" in result:
                    ttfb_runs.append(result["ttfb_s"])
                    print(f"首包 {result['ttfb_s']:.3f}s / 总 {result['total_s']:.3f}s")
                else:
                    print(f"{result['total_s']:.3f}s")
            else:
                print(f"FAILED: {result.get('status', '?')}")

        if runs:
            avg = sum(runs) / len(runs)
            best = min(runs)
            if ttfb_runs:
                ttfb_avg = sum(ttfb_runs) / len(ttfb_runs)
                ttfb_best = min(ttfb_runs)
                print(
                    f"  → 首包平均: {ttfb_avg:.3f}s  首包最快: {ttfb_best:.3f}s"
                    f"  总平均: {avg:.3f}s"
                )
            else:
                ttfb_avg = None
                ttfb_best = None
                print(f"  → 平均: {avg:.3f}s  最快: {best:.3f}s  最慢: {max(runs):.3f}s")
            results[name] = {
                "avg": avg,
                "best": best,
                "runs": runs,
                "ttfb_avg": ttfb_avg,
                "ttfb_best": ttfb_best,
            }
        else:
            results[name] = {"avg": None, "best": None, "runs": [], "ttfb_avg": None, "ttfb_best": None}

    # Summary
    print("\n" + "=" * 60)
    print("  总结：TTFB / 延迟排行")
    print("=" * 60)

    for name, data in results.items():
        if data["avg"] is None:
            print(f"  ❌ {name}: 未配置或测试失败")
        else:
            print(f"  {'✅' if data['best'] < 2 else '⚠️'} {name}:")
            if data["ttfb_best"] is not None:
                print(
                    f"       首包最快 {data['ttfb_best']:.3f}s / 首包平均 {data['ttfb_avg']:.3f}s"
                    f" / 总平均 {data['avg']:.3f}s"
                )
            else:
                print(f"       最快 {data['best']:.3f}s / 平均 {data['avg']:.3f}s")

    print()


if __name__ == "__main__":
    asyncio.run(main())
