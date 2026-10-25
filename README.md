# pi-voice

Windows 上的语音 I/O 前端，把 [claude-voice](https://github.com/guotianya737-sys/claude-voice) 的思路接到 **pi** 上：
麦克风 → VAD/按键 → 离线 ASR → `pi` 子进程（RPC 协议）→ 回复文字 → TTS 朗读。

推理、人格、记忆、工具全部交给 `pi`（它会自动读取 `~/.pi/agent/APPEND_SYSTEM.md` 与 skills/MCP 配置），
本工具只做音频 I/O 和一层"适合朗读"的对话编排。

## 和原版（macOS + Claude Code）的差异

| 部位 | 原版 | 这里 |
|---|---|---|
| 后端 | `claude -p --input-format stream-json` | `pi --mode rpc`（JSONL，事件驱动） |
| 流式取字 | 解析 `assistant` 事件 | `message_update.assistantMessageEvent.text_delta` |
| 一轮结束 | `result` / `stop_reason` | `agent_settled`（含重试/压缩/steer 之后的真正收尾） |
| 本地 TTS | macOS `say` | Windows SAPI（pywin32；缺失回退 PowerShell），合成 wav 后由 sounddevice 播放 |
| 播放/打断 | `afplay` | `sounddevice.RawOutputStream` 分块写，随时可断 |
| 按键说话 | 网页按钮 / 页面内空格 | **全局热键按住说话**（默认 F9，`GetAsyncKeyState` 轮询，不用钩子、不要管理员） |
| 新对话 | 杀进程重建 | `new_session` 命令（0.06s 换新会话，进程不重启） |
| 免提 | 必须装 torch | VAD 懒加载：只用按键说话时不需要 torch / silero-vad |
| 模型下载 | README 说会自动下（实际没有） | `models.py` 首次运行自动下（hf-mirror → huggingface 回退，支持断点续传） |

## 快速开始

```bat
run.bat            :: 首次会自动建 .venv、装依赖、下模型（~230MB）
```

然后浏览器打开 <http://localhost:12394/>。

**按住 `F9` 说话，松开发送。** 说话途中按住 `F9` 可以打断当前朗读，打断后完整收音再提交。
不想用热键就点网页上的「按住说话」，或者用输入框直接打字。

自检与排查：

```bat
run.bat --check                       :: 依赖 / 模型 / 音色 / 麦克风 体检
run.bat --list-devices                :: 列出输入设备（想固定麦克风时用序号填 config.AUDIO_INPUT_DEVICE）
run.bat --selftest <某个.wav>          :: 不用麦克风，跑一遍 ASR -> pi -> TTS 全链路
run.bat --no-hotkey                   :: 本次不启用全局热键
```

## 配置

改 `config.py`：

| 项 | 默认 | 说明 |
|---|---|---|
| `PTT_HOTKEY` | `"F9"` | 全局按住说话键。支持 `F1..F24` / `SPACE` / 字母数字 / `CTRL` `RCTRL` `ALT` `SHIFT` |
| `AUDIO_INPUT_DEVICE` | `None` | 系统默认麦克风；填设备序号或名称可固定 |
| `TTS_PROVIDER` | `"sapi"` | `sapi` 本地 / `siliconflow` / `siliconflow_moss` / `bailian` 云端 |
| `SAPI_VOICE` | `""` | 空 = 自动挑中文音色（本机为 `Microsoft Huihui Desktop`） |
| `SAPI_RATE` | `0` | SAPI 语速，-10..10 |
| `PI_EXTRA_ARGS` | `["--no-mcp"]` | 语音优先低延迟；需要 MCP 工具（rea/comfy 等）就删掉这一项 |
| `PI_MODEL` | `""` | 空 = 用 pi 当前默认模型 |
| `PI_THINKING` | `"low"` | 首字延迟优先；想要深想可改 `medium` |
| `PI_SESSION_DIR` | `./sessions` | 语音端独立会话目录，不污染平时的 pi 会话 |
| `PI_RESPONSE_TIMEOUT` | `180` | 单轮上限（秒） |
| `SERVER_PORT` | `12394` | Web UI 端口 |

云端 TTS：把 `tts_config.example.json` 复制成 `tts_config.local.json` 填 key，
或用 `SILICONFLOW_API_KEY` / `DASHSCOPE_API_KEY` 环境变量。`response_format` 必须是 **wav**（播放端只解 wav）。

免提模式（不需要按键）：`pip install torch silero-vad`，然后在网页上切到「免提模式」。

## 目录

```
pi-voice/
├── main.py          # 编排：按键/免提、ASR→pi→TTS、打断、自检
├── pi_client.py     # pi 子进程 RPC 客户端（常驻读线程 + asyncio 队列）
├── tts_engine.py    # SAPI（pywin32/PowerShell）+ 硅基流动 + 百炼；sounddevice 播放
├── asr_engine.py    # sherpa-onnx SenseVoice int8
├── models.py        # ASR 模型下载（hf-mirror 优先）
├── vad.py           # Silero VAD（懒加载，免提才用）
├── audio_io.py      # sounddevice 采集 + 重采样到 16k
├── hotkey.py        # 全局按住说话（纯 ctypes）
├── server.py        # aiohttp + WebSocket
├── static/index.html
├── config.py
└── run.bat
```

## 踩坑记录（都已修）

1. **pywin32 嵌套 COM 初始化会毁掉线程**：在 `_synthesize` 里 `CoInitialize` 后又在内层 helper 再 `CoInitialize/CoUninitialize`，
   会导致后续 `GetVoices()` 抛 `DISP_E_EXCEPTION (-2147352567)`。改成 `_with_com()` 单一入口管理 apartment。
2. **pi RPC 必须按 LF 切行**：用二进制管道手工切，避免 `readline` 把 U+2028/U+2029 当记录边界。
3. **`new_session` 不会发 `agent_settled`**：等它等于白等 10s。改成等 `response` 记录（id 对应）→ 0.06s。
4. **TTS 预处理任务异常会静默死锁**：producer 挂了没人叫醒消费者，整轮对话会一直挂着。已加 done-callback 上报 + 投喂 `None`。
5. **`.bat` 必须纯 ASCII + CRLF**，中文会直接写成乱码字节。
6. SAPI 合成的 wav 是 22050Hz/16bit/单声道，播放端按 wav 头自适应，不要写死采样率。

## License

MIT（继承原项目）
