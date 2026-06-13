# claude-voice

Voice I/O frontend for Claude Code — add speech input and output without touching the API.

A thin shell that adds voice interaction to [Claude Code](https://claude.ai/code). It captures microphone input, transcribes via offline ASR, feeds text to a persistent `claude` subprocess over stdin/stdout JSON, and reads responses aloud with TTS.

Claude Code handles all reasoning — persona, memory, tools, context. This tool only handles audio I/O.

---

# claude-voice

Claude Code 的语音 I/O 前端，在 API 层之上增加语音输入输出能力。

一个薄层语音外壳，为 [Claude Code](https://claude.ai/code) 提供语音交互。采集麦克风输入 → 离线 ASR 转写 → 写入长驻 `claude` 子进程的标准输入（stream-json）→ 从标准输出读取回复 → TTS 朗读。

Claude Code 负责全部推理——角色、记忆、工具调用、上下文管理。本工具仅处理音频输入输出。

## Features / 功能

- **Hands-free mode / 免提模式** — continuous listening with VAD-based auto-reply
- **Push-to-talk mode / 按讲模式** — hold a key or button to speak, release to send
- **Barge-in / 中途打断** — interrupt the assistant mid-speech; captures the full interruption before submitting
- **Offline ASR / 离线语音识别** — sherpa-onnx SenseVoice int8, local, private, ~186 MB
- **Dual TTS / 双引擎语音合成** — macOS `say` (zero-network, instant) or SiliconFlow CosyVoice2 (cloud)
- **Web UI** — dialogue bubbles, real-time status indicators, microphone level meter, speech probability
- **Adaptive noise floor / 自适应底噪** — dynamically tracks ambient noise to adjust VAD sensitivity
- **Multi-device support / 多设备支持** — built-in mic, AirPods, Bluetooth headsets with automatic sample rate conversion
- **Low-effort by default / 默认快速短答** — voice mode prioritizes short replies; switches to full reasoning only on explicit triggers
- **Timeout protection / 超时保护** — ASR and Claude calls will not hang indefinitely

## Architecture / 架构

```
Microphone → VAD → ASR → text → claude subprocess (stdin)
                                    │
                              Claude processes
                                    │
Speaker ← TTS ← text ← claude subprocess (stdout)
```

The Web UI connects over WebSocket to stream status updates, transcriptions, and responses in real time.

## Prerequisites / 环境要求

- **macOS** (required for `say` TTS; ASR, server, and UI work on Linux with minor adjustments)
- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Claude Code](https://claude.ai/code)** CLI, available as the `claude` command

## Quick Start / 快速开始

```bash
# 1. Install dependencies / 安装依赖
uv sync

# 2. Start / 启动
uv run python main.py

# 3. Open browser / 打开浏览器
open http://localhost:12394/
```

On first run, the SenseVoice ASR model (~186 MB) is auto-downloaded to `models/`.

## Configuration / 配置

Edit `config.py` for VAD thresholds, TTS settings, timeouts, and server port.

### SiliconFlow TTS (optional / 可选)

Copy `tts_config.example.json` to `tts_config.local.json` and fill in your API key:

```json
{
  "siliconflow": {
    "api_key": "sk-your-key-here",
    "voice": "FunAudioLLM/CosyVoice2-0.5B:anna"
  }
}
```

Alternatively, set the `SILICONFLOW_API_KEY` environment variable.  
When unavailable, falls back to macOS `say`.

Switch TTS engines at runtime via the Web UI button.

## Project Structure / 项目结构

```
claude-voice/
├── main.py             # Entry point and orchestration loop
├── config.py           # All tunable constants
├── audio_io.py         # Microphone capture (sounddevice)
├── vad.py              # Silero VAD with 3-state state machine
├── asr_engine.py       # sherpa-onnx SenseVoice wrapper
├── tts_engine.py       # macOS say + SiliconFlow TTS
├── claude_client.py    # claude subprocess stdin/stdout JSON pipe
├── server.py           # aiohttp HTTP + WebSocket server
├── static/index.html   # Web UI (vanilla HTML/CSS/JS)
├── pyproject.toml      # Dependencies (uv)
├── tts_config.example.json
└── models/             # ASR model (auto-downloaded, gitignored)
```

## License / 许可证

MIT
