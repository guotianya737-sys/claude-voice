# claude-voice

Voice I/O frontend for Claude Code — add speech input and output without touching the API.

A thin shell that adds voice interaction to [Claude Code](https://claude.ai/code). It captures microphone input, transcribes via offline ASR, feeds text to a persistent `claude` subprocess over stdin/stdout JSON, and reads responses aloud with TTS.

Claude Code handles all reasoning — persona, memory, tools, and context. This tool only handles audio I/O.

## Features

- **Hands-free mode** — continuous listening with VAD-based auto-reply
- **Push-to-talk mode** — hold a key or button to speak, release to send
- **Barge-in** — interrupt the assistant mid-speech in both hands-free and PTT modes; captures the full interruption before submitting
- **Offline ASR** — sherpa-onnx SenseVoice int8, local, private, ~186 MB
- **Three TTS engines** — macOS `say` (zero-network, instant), SiliconFlow CosyVoice2 / MOSS (cloud), Bailian CosyVoice v3 Flash / Qwen TTS (cloud, streaming)
- **Web UI** — dialogue bubbles, real-time status indicators, microphone level meter, speech probability
- **New conversation** — one-click reset of the Claude session from the Web UI
- **Adaptive noise floor** — dynamically tracks ambient noise to adjust VAD sensitivity
- **Multi-device support** — built-in mic, AirPods, Bluetooth headsets with automatic sample rate conversion
- **Low-effort by default** — voice mode prioritizes short replies; switches to full reasoning only on explicit triggers
- **Timeout protection** — ASR and Claude calls will not hang indefinitely
- **TTS benchmark tool** — measure first-audio and total latency across all TTS engines

## Architecture

```
Microphone → VAD → ASR → text → claude subprocess (stdin)
                                    │
                              Claude processes
                                    │
Speaker ← TTS ← text ← claude subprocess (stdout)
```

The Web UI connects over WebSocket to stream status updates, transcriptions, and responses in real time.

## Prerequisites

- **macOS** (required for `say` TTS; ASR, server, and UI work on Linux with minor adjustments)
- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Claude Code](https://claude.ai/code)** CLI, available as the `claude` command

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start
uv run python main.py

# 3. Open browser
open http://localhost:12394/
```

On first run, the SenseVoice ASR model (~186 MB) is auto-downloaded to `models/`.

## Configuration

Edit `config.py` for VAD thresholds, TTS settings, timeouts, and server port.

### Cloud TTS (optional)

Copy `tts_config.example.json` to `tts_config.local.json` and fill in your API keys:

```json
{
  "siliconflow": {
    "api_key": "sk-your-key-here",
    "default_model": "FunAudioLLM/CosyVoice2-0.5B",
    "default_voice": "FunAudioLLM/CosyVoice2-0.5B:anna"
  },
  "bailian": {
    "api_key": "sk-your-key-here",
    "model": "cosyvoice-v3-flash",
    "voice": "longanli_v3"
  }
}
```

Alternatively, set environment variables `SILICONFLOW_API_KEY` and `DASHSCOPE_API_KEY`.
When no cloud TTS is configured, falls back to macOS `say`.

Switch TTS engines at runtime via the Web UI dropdown.

## TTS Engines

| Engine | Type | Latency | Quality | Requires |
|--------|------|---------|---------|----------|
| macOS `say` | local | instant | good | nothing |
| SiliconFlow CosyVoice2 | cloud (HTTP) | ~1–3s | high | API key |
| SiliconFlow MOSS | cloud (HTTP) | ~1–3s | high | API key |
| Bailian CosyVoice v3 Flash | cloud (WebSocket streaming) | low | high | API key |
| Bailian Qwen TTS | cloud (HTTP, batch) | ~2–5s | high | API key |
| Bailian Qwen TTS Realtime | cloud (WebSocket streaming) | low | high | API key |

Run `uv run python bench_tts.py` to benchmark TTS latency for all configured engines.

## Project Structure

```
claude-voice/
├── main.py               # Entry point and orchestration loop
├── config.py             # All tunable constants
├── audio_io.py           # Microphone capture (sounddevice)
├── vad.py                # Silero VAD with 3-state state machine
├── asr_engine.py         # sherpa-onnx SenseVoice wrapper
├── tts_engine.py         # macOS say + SiliconFlow + Bailian TTS
├── claude_client.py      # claude subprocess stdin/stdout JSON pipe
├── server.py             # aiohttp HTTP + WebSocket server
├── static/index.html     # Web UI (vanilla HTML/CSS/JS)
├── bench_tts.py          # TTS latency benchmark tool
├── pyproject.toml        # Dependencies (uv)
├── tts_config.example.json
├── LICENSE               # MIT
└── models/               # ASR model (auto-downloaded, gitignored)
```

## License

MIT

---

# claude-voice（中文）

Claude Code 的语音 I/O 前端，在 API 层之上增加语音输入输出能力。

一个薄层语音外壳，为 [Claude Code](https://claude.ai/code) 提供语音交互。采集麦克风输入 → 离线 ASR 转写 → 写入长驻 `claude` 子进程的标准输入（stream-json）→ 从标准输出读取回复 → TTS 朗读。Claude Code 负责全部推理（角色、记忆、工具调用、上下文管理），本工具仅处理音频输入输出。

## 功能

- **免提模式** — 持续监听，VAD 自动检测说话结束并触发回复
- **按讲模式** — 按住按键或按钮说话，松开发送
- **中途打断** — 免提和按讲模式均支持打断正在播放的 TTS；打断后完整收音再提交，不会截断语音
- **离线语音识别** — sherpa-onnx SenseVoice int8，本地运行，隐私安全，约 186 MB
- **三种 TTS 引擎** — macOS `say`（零网络、即时响应）、硅基流动 CosyVoice2 / MOSS（云端）、阿里百炼 CosyVoice v3 Flash / Qwen TTS（云端流式）
- **Web 界面** — 对话气泡、实时状态指示、麦克风电平表、语音概率显示
- **新对话** — Web 界面一键重置 Claude 会话上下文
- **自适应底噪** — 动态追踪环境噪音，自动调整 VAD 灵敏度
- **多设备支持** — 内置麦克风、AirPods、蓝牙耳机，自动采样率转换
- **默认快速短答** — 语音模式优先短回复，仅在明确要求时才进入完整推理
- **超时保护** — ASR 和 Claude 调用不会无限挂起
- **TTS 延迟测试** — 内置测试脚本，对比各引擎首包延迟和总延迟

## 架构

```
麦克风 → VAD → ASR → 文字 → claude 子进程(stdin)
                               │
                         Claude 处理
                               │
扬声器 ← TTS ← 文字 ← claude 子进程(stdout)
```

Web 界面通过 WebSocket 连接，实时推送状态、识别结果和流式回复。

## 环境要求

- **macOS**（`say` TTS 需要；ASR、服务、UI 在 Linux 上稍作调整也可运行）
- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** 包管理器
- **[Claude Code](https://claude.ai/code)** CLI，安装后可用 `claude` 命令

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 启动
uv run python main.py

# 3. 打开浏览器
open http://localhost:12394/
```

首次运行时 SenseVoice ASR 模型（~186 MB）会自动下载到 `models/` 目录。

## 配置

编辑 `config.py` 调整 VAD 阈值、TTS 设置、超时和端口。

### 云端 TTS（可选）

将 `tts_config.example.json` 复制为 `tts_config.local.json`，填入 API Key：

```json
{
  "siliconflow": {
    "api_key": "sk-你的密钥",
    "default_model": "FunAudioLLM/CosyVoice2-0.5B",
    "default_voice": "FunAudioLLM/CosyVoice2-0.5B:anna"
  },
  "bailian": {
    "api_key": "sk-你的密钥",
    "model": "cosyvoice-v3-flash",
    "voice": "longanli_v3"
  }
}
```

也可通过环境变量 `SILICONFLOW_API_KEY` 和 `DASHSCOPE_API_KEY` 设置。未配置云端 TTS 时自动回退到 macOS `say`。

在 Web 界面下拉菜单即可实时切换 TTS 引擎。

## TTS 引擎

| 引擎 | 类型 | 延迟 | 音质 | 需要 |
|------|------|------|------|------|
| macOS `say` | 本地 | 即时 | 良好 | 无需配置 |
| 硅基流动 CosyVoice2 | 云端 (HTTP) | ~1–3s | 高 | API Key |
| 硅基流动 MOSS | 云端 (HTTP) | ~1–3s | 高 | API Key |
| 百炼 CosyVoice v3 Flash | 云端 (WebSocket 流式) | 低 | 高 | API Key |
| 百炼 Qwen TTS | 云端 (HTTP，批量) | ~2–5s | 高 | API Key |
| 百炼 Qwen TTS Realtime | 云端 (WebSocket 流式) | 低 | 高 | API Key |

运行 `uv run python bench_tts.py` 可测试各引擎 TTS 延迟。

## 项目结构

```
claude-voice/
├── main.py               # 入口，编排主循环
├── config.py             # 所有可调常量
├── audio_io.py           # 麦克风采集（sounddevice）
├── vad.py                # Silero VAD + 三段式状态机
├── asr_engine.py         # sherpa-onnx SenseVoice 封装
├── tts_engine.py         # macOS say + 硅基流动 + 百炼 TTS
├── claude_client.py      # claude 子进程 stdin/stdout JSON 管道
├── server.py             # aiohttp HTTP + WebSocket 服务
├── static/index.html     # Web 界面（纯 HTML/CSS/JS）
├── bench_tts.py          # TTS 延迟测试脚本
├── pyproject.toml        # 依赖声明（uv）
├── tts_config.example.json
├── LICENSE               # MIT
└── models/               # ASR 模型（自动下载，gitignore）
```

## 许可证

MIT
