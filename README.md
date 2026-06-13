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

---

# claude-voice

Claude Code 的语音 I/O 前端 —— 不动 API，只管听和说。

一个薄薄的语音外壳，为 [Claude Code](https://claude.ai/code) 加上语音交互能力。采集麦克风 → 离线 ASR 转文字 → 喂给常驻 `claude` 子进程（stdin/stdout JSON）→ TTS 把回复念出来。

Claude Code 负责**所有脑力活**——人设、记忆、工具、上下文。这个工具只管耳朵和嘴。

## 功能

- **免提模式** — 持续监听，VAD 自动检测说话结束并触发回复
- **按讲模式** — 按住空格或按钮说话，松开发送
- **中途打断** — miru 说到一半时直接插话，打断后完整收音再提交
- **离线语音识别** — sherpa-onnx SenseVoice int8，本地运行，隐私安全，约 186 MB
- **双 TTS 引擎** — macOS `say`（零网络、即时）或硅基流动 CosyVoice2（云端高音质）
- **Web 界面** — 对话气泡、实时状态、麦克风收音表、语音概率
- **自适应底噪** — 自动追踪环境噪音，动态调整 VAD 阈值
- **多设备支持** — 内置麦克风、AirPods、蓝牙耳机自动适配采样率
- **默认快速短答** — 语音模式优先短回复；主人说「好好想想」等触发词时才深入思考
- **超时保护** — ASR / Claude 调用不会永久卡死

## 架构

```
麦克风 → VAD → ASR → 文字 → claude 子进程(stdin)
                               │
                         Claude 思考
                               │
耳朵 / 扬声器 ← TTS ← 文字 ← claude 子进程(stdout)
```

Web 界面通过 WebSocket 连接，实时显示状态、识别结果和流式回复。

## 环境要求

- **macOS**（`say` TTS 需要；ASR/服务/UI 在 Linux 上稍作调整也可运行）
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

### 硅基流动 TTS（可选）

将 `tts_config.example.json` 复制为 `tts_config.local.json`，填入 API Key：

```json
{
  "siliconflow": {
    "api_key": "sk-你的密钥",
    "voice": "FunAudioLLM/CosyVoice2-0.5B:anna"
  }
}
```

也可以通过环境变量 `SILICONFLOW_API_KEY` 设置密钥。  
不可用时自动回退到 macOS `say`。

在 Web 界面点击按钮即可实时切换 TTS 引擎。

## 项目结构

```
claude-voice/
├── main.py             # 入口，编排主循环
├── config.py           # 所有可调常量
├── audio_io.py         # 麦克风采集（sounddevice）
├── vad.py              # Silero VAD + 三段式状态机
├── asr_engine.py       # sherpa-onnx SenseVoice 封装
├── tts_engine.py       # macOS say + 硅基流动 TTS
├── claude_client.py    # claude 子进程 stdin/stdout JSON 管道
├── server.py           # aiohttp HTTP + WebSocket
├── static/index.html   # Web 界面（纯 HTML/CSS/JS）
├── pyproject.toml      # 依赖声明（uv）
├── tts_config.example.json
└── models/             # ASR 模型（自动下载，gitignore）
```

---

## 许可证

MIT
