# claude-voice

Voice I/O frontend for Claude Code — hear and speak without touching the API.

A thin shell that adds voice interaction to [Claude Code](https://claude.ai/code). It captures your microphone, transcribes with offline ASR, feeds text to a persistent `claude` process via stdin/stdout JSON, and reads the response aloud with TTS.

Claude Code handles *all* the brain work — persona, memory, tools, context. This tool only handles ears and mouth.

## Features

- **Hands-free mode** — continuous listening with voice activity detection (VAD) and auto-reply
- **Push-to-talk mode** — hold spacebar or button to speak
- **Barge-in** — interrupt Claude mid-sentence and speak again
- **Offline ASR** — sherpa-onnx SenseVoice int8, local, private, ~186 MB
- **Dual TTS** — macOS `say` (zero-network, instant) or SiliconFlow CosyVoice2 (cloud)
- **Web UI** — dialogue bubbles, real-time status, microphone meter, live speech probability
- **Adaptive noise floor** — automatically adjusts to ambient noise
- **Multi-mic support** — works with built-in mic, AirPods, Bluetooth headsets; auto resampling
- **Low-effort by default** — fast, short replies for voice; deep think only when you ask for it
- **Timeout protection** — ASR / Claude calls won't hang forever

## Architecture

```
Microphone → VAD → ASR → text → claude subprocess (stdin)
                                    │
                              Claude thinks
                                    │
Ear / speaker ← TTS ← text ← claude subprocess (stdout)
```

The Web UI connects over WebSocket to display status, transcriptions, and streaming replies.

## Prerequisites

- **macOS** (for `say` TTS; ASR/server/UI work on Linux with minor tweaks)
- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Claude Code](https://claude.ai/code)** CLI installed as `claude`

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start
uv run python main.py

# 3. Open browser
open http://localhost:12394/
```

On first run, the SenseVoice ASR model (~186 MB) will be auto-downloaded to `models/`.

## Configuration

Edit `config.py` for VAD thresholds, TTS settings, timeouts, and port.

### SiliconFlow TTS (optional)

Copy `tts_config.example.json` to `tts_config.local.json` and fill in your key:

```json
{
  "siliconflow": {
    "api_key": "sk-your-key-here",
    "voice": "FunAudioLLM/CosyVoice2-0.5B:anna"
  }
}
```

Set the env var `SILICONFLOW_API_KEY` as an alternative to the config file.  
When unavailable, falls back to macOS `say`.

Switch TTS live via the Web UI button.

## Project Structure

```
claude-voice/
├── main.py             # Entry point, orchestration loop
├── config.py           # All tunable constants
├── audio_io.py         # Microphone capture (sounddevice)
├── vad.py              # Silero VAD + 3-state state machine
├── asr_engine.py       # sherpa-onnx SenseVoice wrapper
├── tts_engine.py       # macOS say + SiliconFlow TTS
├── claude_client.py    # claude subprocess stdin/stdout JSON pipe
├── server.py           # aiohttp HTTP + WebSocket
├── static/index.html   # Web UI (vanilla HTML/CSS/JS)
├── pyproject.toml      # Dependencies (uv)
├── tts_config.example.json
└── models/             # ASR model (auto-downloaded, gitignored)
```

## License

MIT
