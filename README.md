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

**后台常驻（推荐，关掉终端也不停）：**

```bat
voiceon               :: 启动（沿用上次的说话模式；已在跑就只报状态）
voiceon ptt           :: 按键说话：按住 F9 说，松开发送
voiceon handsfree     :: 免提常听：一直听着，说完自动断句接话
voicestatus           :: 看 模式 / PID / 端口 / 日志尾巴
voiceoff              :: 停止（TTS 服务一起停）
voiceoff --force      :: pid 丢了或端口被别人占着时强杀
```

模式会记在 `%LOCALAPPDATA%\pi-voice\mode`，下次 `voiceon` 不带参数就沿用；
**切换模式会自动重启服务**（说的和听的都换掉）。也可以在网页上点「免提模式 / 按讲模式」临时切。

- `ptt`：按住 **F9** 说话，松开发送；朗读中按住 F9 = 打断。
- `handsfree`：不用按键，VAD 自动判断你说完没有；朗读中按一下 **F9** = 打断。
  需要 `silero-vad`（已在 requirements）。⚠️ 用外放会互相触发，建议戴耳机。

免提收不到反应时，按这个顺序查：

1. `run.bat --miccheck 5`，对着麦克风说话 —— 看"有声音 ✔ / 几乎静音 ✘"。
   本机耳麦实测只有 **~47 dB**，所以 `REQUIRE_PROB_AND_DB=False`（只看 silero 概率，不卡音量）。
   如果你换了更小声的麦克风还是不行，把 `DB_THRESHOLD` 往下调，或保持只信概率。
2. 看 `voicestatus` 的日志尾巴：出现 `[turn] 收到一段 …` 说明 VAD 收到了；
   出现 `[asr] '…'` 说明识别到了；`[play] 开始朗读 / 朗读结束` 说明确实放了。
3. 一轮要 **15–20 秒**（pi 思考 ~10s + 本地 TTS RTF≈0.85）。嫌慢先看 `[indextts] …字 …s` 那行，确认是 TTS 在等而不是 pi 在等。

三条命令装在 `%USERPROFILE%\bin`（已在 PATH，cmd 和 PowerShell 通用），
实现在 `ctl.py`；PowerShell 里还会顺带设置会话级环境变量 `PI_VOICE_ON=1` / `PI_VOICE_URL`（`voiceoff` 时清掉），脚本可以用它判断在不在跑。

**前台调试（看得到实时日志，Ctrl+C 停）：**

```bat
run.bat
```

首次运行会自动建 `.venv`、装依赖、下模型（~230MB，几秒到几分钟看网速）。
启动后浏览器打开 <http://localhost:12394/>。

**按住 `F9` 说话，松开发送。** 说话途中按住 `F9` 可以打断当前朗读，打断后完整收音再提交。
不想用热键就点网页上的「按住说话」，或者用输入框直接打字。

自检与排查：

```bat
run.bat --check                       :: 依赖 / 模型 / 音色 / 麦克风 体检
run.bat --list-devices                :: 列出输入设备（想固定麦克风时用序号填 config.AUDIO_INPUT_DEVICE）
run.bat --selftest <某个.wav>          :: 不用麦克风，跑一遍 ASR -> pi -> TTS 全链路
run.bat --no-hotkey                   :: 本次不启用全局热键
run.bat --miccheck 5                  :: 录 5 秒，报告音量/VAD 概率（判"到底有没有收到人声"）
```

## 配置

改 `config.py`：

| 项 | 默认 | 说明 |
|---|---|---|
| `PTT_HOTKEY` | `"F9"` | 全局按住说话键。支持 `F1..F24` / `SPACE` / 字母数字 / `CTRL` `RCTRL` `ALT` `SHIFT` |
| `AUDIO_INPUT_DEVICE` | `None` | 系统默认麦克风；填设备序号或名称可固定 |
| `TTS_PROVIDER` | `"indextts"` | `indextts`=本地 IndexTTS（能演情绪+克隆音色）/ `sapi`=本地 SAPI（机械但离线即时，兜底） |
| `SAPI_VOICE` | `""` | 空 = 自动挑中文音色（本机为 `Microsoft Huihui Desktop`） |
| `SAPI_RATE` | `0` | SAPI 语速，-10..10 |
| `PI_EXTRA_ARGS` | `["--no-mcp"]` | 语音优先低延迟；需要 MCP 工具（rea/comfy 等）就删掉这一项 |
| `PI_MODEL` | `""` | 空 = 用 pi 当前默认模型 |
| `PI_THINKING` | `"low"` | 首字延迟优先；想要深想可改 `medium` |
| `PI_SESSION_DIR` | `./sessions` | 语音端独立会话目录，不污染平时的 pi 会话 |
| `PI_RESPONSE_TIMEOUT` | `180` | 单轮上限（秒） |
| `SERVER_PORT` | `12394` | Web UI 端口 |

### 语音（谁在说话）

**默认：本地 IndexTTS-2.5**（`TTS_PROVIDER="indextts"`）——声音更自然，而且能用 7 种情绪演。
它跑在独立进程里（`~/index-tts/indextts_server.py`，:12395），`voiceon` 会把两个服务一起拉起：

```
voiceon    → ① IndexTTS 服务（加载约 40s，之后常驻） ② pi-voice
voiceoff   → 两个一起停
```

实测（RX 9070 XT / ROCm）：**RTF ≈ 0.85–1.0**（5 秒音频约 4.5–5 秒生成），
每个请求另有 ~2.5s 固定开销，所以**首音 ≈ 2.5s + 0.85×时长**；45 字回复约 7.4s 生成 8.5s 音频。
收益是情感和音色克隆，代价是首音慢——**全本地、不要任何云端 key** 是有意为之。

音色参考是 `~/index-tts/voices/miru_ref.wav`（一段中性御姐音，基频 ~187Hz），
换声音只要换这个文件并重启服务；同一个参考音 + 常驻模型 = 每次出声音色一致，不会漂。

**分档策略**：`INDEXTTS_SINGLE_REQUEST=True` 整段一次合成（连续无接缝）。
本地 RTF≈1 时切成多段反而会出现静音缝（生成第 2 段时第 1 段已播完），所以默认不开分段。

**情绪**：模型回复首行写 `[emotion:teasing]` 这类标签，映射到 IndexTTS 的 8 维情绪向量
（happy / excited / teasing / loving / worried / sad / angry）。

**回退链**：`indextts → sapi`（`config.TTS_FALLBACK_CHAIN`）。
本地 TTS 服务没起来就退回 Windows 自带语音，整条链路都不出网。

> 早期版本接过云端 TTS（百炼 CosyVoice / 硅基流动），现在已整体移除：
> 这套东西的定位是**纯本地**，不需要任何云端 key，也不受断网和代理影响。

免提模式（不需要按键）：`pip install torch silero-vad`，然后在网页上切到「免提模式」。

## 目录

```
pi-voice/
├── main.py          # 编排：按键/免提、ASR→pi→TTS、打断、自检
├── pi_client.py     # pi 子进程 RPC 客户端（常驻读线程 + asyncio 队列）
├── tts_engine.py    # 本地 IndexTTS（HTTP）+ 本地 SAPI（pywin32/PowerShell）；sounddevice 播放
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
7. **开代理时 localhost 也会被代理**：`Invoke-WebRequest http://localhost:...` 走代理会返回 503，就绪探测必须用裸 TCP 连接（`Net.Sockets.TcpClient`），否则 voiceon 会假死等两分钟。
8. **`.bat` 里别用裸 `find`/`tasklist`**：git-bash 的 GNU `find` 会盖掉系统 `find.exe`，一律写 `%SystemRoot%\System32\xxx.exe`。
9. **venv 的 `python.exe` 是转发器**：它 spawn 一个 base python 做真正的活儿，所以端口持有者 PID ≠ 启动时拿到的 PID；杀进程必须 `/T` 杀整棵树，判断"是不是我们的进程"要连父进程命令行一起看。
10. **PowerShell 5.1 两个坑**：profile 里写中文注释必须存成 **UTF-8 with BOM**（否则按 cp936 解码直接语法错，连带 `proxy` 都失效）；没有 `??` 运算符。

## License

MIT（继承原项目）
