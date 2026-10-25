from __future__ import annotations

import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
# None = system default input device. 设为设备序号或名称来固定麦克风（启动日志会打印可用列表）。
AUDIO_INPUT_DEVICE = None

PROB_THRESHOLD = 0.4
DB_THRESHOLD = 64
REQUIRE_PROB_AND_DB = True
NOISE_FLOOR_INITIAL_DB = 58.0
NOISE_FLOOR_ALPHA = 0.95
SPEECH_DB_MARGIN = 3.0
REQUIRED_HITS = 3
REQUIRED_MISSES = 24
MIN_SPEECH_DURATION = 0.3
MAX_SPEECH_SECONDS = 30.0
TTS_TAIL_GUARD = 1.0
PRE_BUFFER_CHUNKS = 12
AUDIO_QUEUE_MAX_CHUNKS = 240
TTS_CHUNK_MIN_CHARS = 20
TTS_CHUNK_MAX_CHARS = 120
TTS_BATCH_TIMEOUT = 0.3
TTS_BATCH_MAX_CHARS = 150
TTS_TEXT_NORMALIZATION = True
TTS_URL_REPLACEMENT = "这里有一个链接"
TTS_CODE_REPLACEMENT = "这里有一段代码"
SILICONFLOW_TTS_CHUNK_MIN_CHARS = 40
SILICONFLOW_TTS_CHUNK_MAX_CHARS = 150
BAILIAN_TTS_CHUNK_MIN_CHARS = 24
BAILIAN_TTS_CHUNK_MAX_CHARS = 90
BAILIAN_PREFETCH_TTS_CHUNK_MIN_CHARS = 48
BAILIAN_PREFETCH_TTS_CHUNK_MAX_CHARS = 160
ASR_TIMEOUT = 20
ASR_MIN_NORMALIZED_CHARS = 2
ASR_ALLOWED_SHORT_TEXTS = frozenset({
    "嗯",
    "哦",
    "好",
    "行",
    "对",
    "是",
    "不",
    "要",
    "喂",
    "啊",
})
ASR_IGNORED_NORMALIZED_TEXTS = frozenset({
    "miru",
    "milu",
    "米露",
    "you",
})
ASR_NOISE_TEXT_RE = re.compile(r"[\s，。！？、…—\-！,.?~～`'\"“”‘’()\[\]{}<>《》【】]+")

# ---------------------------------------------------------------------------
# 按键说话（PTT）
# ---------------------------------------------------------------------------
# 全局按住说话热键。支持 F1..F24 / SPACE / A..Z / RCTRL / RALT / LCTRL / LALT / SHIFT / 等。
PTT_HOTKEY = "F9"
# 蜂鸣提示：按下（起录）一声，松开（提交）不响。
PTT_BEEP = True

# 免提模式需要 torch + silero-vad；只用 PTT 时可以不装（VAD 懒加载）。

# ---------------------------------------------------------------------------
# pi 后端（子进程 RPC 模式）
# ---------------------------------------------------------------------------
PI_NODE = str(Path.home() / "AppData/Local/pi-node/current/node.exe")
PI_LAUNCHER = str(Path.home() / ".pi/agent/bin/pi-launcher.js")
PI_EXTRA_ARGS = [
    "--no-mcp",  # 语音对话优先低延迟；需要 MCP 工具时删掉这一行
]
PI_MODEL = ""  # 留空 = 用 pi 当前默认模型；如 "bailian/qwen3-max"
PI_THINKING = "low"  # off / minimal / low / medium / high
PI_SESSION_DIR = str(BASE_DIR / "sessions")
PI_RESPONSE_TIMEOUT = 180

VOICE_APPEND_SYSTEM_PROMPT = """
你正在通过语音和用户对话，用户说的话来自离线语音识别（可能有错别字），回复会被 TTS 念出来。
默认低思考、快速短答：先直接回应用户的话，不要铺长篇分析，不要念出思考过程。

按用户这一轮的意图决定长度，不要凑字数：
- 附和、确认、感叹、寒暄：一句自然回应即可
- 直接问题：先给答案，通常 1-2 句
- 任务或工具请求：直接执行；执行中不要长篇解说，完成后用 1-3 句总结结果
- 复杂分析：只有用户明确要求深入时才展开

语音朗读约束：不要输出表格、代码块、URL 原文、markdown 装饰；数字和单位用口语表达。
不要输出 `[emotion:...]` 之类的情绪标签。

需要实时信息、文件/项目上下文、MCP 或 skill 能力时，可以自主调用工具，完成后用适合朗读的短句总结。
""".strip()

BACKCHANNEL_REPLY_PREFIX = (
    "【语音回复策略：BACKCHANNEL。对方只是附和、确认、感叹或寒暄；"
    "自然回应一句即可，不展开、不解释、不主动列点。】\n"
)
DIRECT_REPLY_PREFIX = (
    "【语音回复策略：DIRECT。先直接给答案；通常控制在1-2句，"
    "只补最必要的上下文，不要铺陈。】\n"
)
TASK_REPLY_PREFIX = (
    "【语音回复策略：TASK。用户在请求你执行、查询、修改或生成内容；"
    "能做就直接做，必要时简短说明进展，完成后用1-3句总结结果。】\n"
)
CHAT_REPLY_PREFIX = (
    "【语音回复策略：CHAT。自然聊天，直接回应，1-2小段即可；"
    "不要长篇铺陈，不要展示推理过程，可以自然追问一句。】\n"
)
DEEP_THINK_PREFIX = (
    "【语音模式：被要求认真思考。可以更完整地分析，但仍然分成适合语音朗读的短段。】\n"
)
DEEP_THINK_TRIGGERS = (
    "好好想",
    "认真想",
    "仔细想",
    "深入想",
    "认真分析",
    "仔细分析",
    "详细分析",
    "详细讲",
    "深入分析",
    "推理一下",
    "想清楚",
)

SERVER_HOST = "localhost"
SERVER_PORT = 12394

# ---------------------------------------------------------------------------
# ASR（sherpa-onnx SenseVoice int8，离线）
# ---------------------------------------------------------------------------
MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
MODEL_DIR = BASE_DIR / "models" / MODEL_NAME
SENSE_VOICE_MODEL = MODEL_DIR / "model.int8.onnx"
SENSE_VOICE_TOKENS = MODEL_DIR / "tokens.txt"
MODEL_FILES = (
    "model.int8.onnx",
    "tokens.txt",
    "test_wavs/zh.wav",
)
MODEL_HF_REPO = f"csukuangfj/{MODEL_NAME}"
MODEL_MIRRORS = (
    "https://hf-mirror.com",
    "https://huggingface.co",
)
ASR_NUM_THREADS = 4

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
# sapi = Windows 本地 SAPI（离线即时）；siliconflow / siliconflow_moss / bailian = 云端
TTS_PROVIDER = "sapi"

# SAPI 音色。留空 = 自动挑第一个中文音色；也可以是 "Microsoft Huihui Desktop" 等完整名。
SAPI_VOICE = ""
# SAPI 语速 -10..10（0 为正常）
SAPI_RATE = 0
# SAPI 音量 0..100
SAPI_VOLUME = 100

TTS_CONFIG_FILE = BASE_DIR / "tts_config.local.json"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/audio/speech"
SILICONFLOW_API_KEY_ENV = "SILICONFLOW_API_KEY"
SILICONFLOW_DEFAULT_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
SILICONFLOW_DEFAULT_VOICE = "FunAudioLLM/CosyVoice2-0.5B:anna"
SILICONFLOW_SAMPLE_RATE = 32000
SILICONFLOW_RESPONSE_FORMAT = "wav"
SILICONFLOW_STREAM = True
SILICONFLOW_SPEED = 1.0
SILICONFLOW_GAIN = 0
SILICONFLOW_MOSS_DEFAULT_MODEL = "fnlp/MOSS-TTSD-v0.5"
SILICONFLOW_MOSS_DEFAULT_VOICE = "fnlp/MOSS-TTSD-v0.5:anna"
SILICONFLOW_MOSS_SAMPLE_RATE = 32000
SILICONFLOW_MOSS_RESPONSE_FORMAT = "wav"
SILICONFLOW_MOSS_STREAM = True
SILICONFLOW_MOSS_SPEED = 1.1
SILICONFLOW_MOSS_GAIN = 0.0
SILICONFLOW_TIMEOUT = 30

BAILIAN_API_KEY_ENV = "DASHSCOPE_API_KEY"
BAILIAN_WEBSOCKET_API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
BAILIAN_QWEN_REALTIME_WEBSOCKET_API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
BAILIAN_HTTP_API_URL = "https://dashscope.aliyuncs.com/api/v1"
BAILIAN_DEFAULT_MODEL = "cosyvoice-v3-flash"
BAILIAN_DEFAULT_VOICE = "longanli_v3"
BAILIAN_AUDIO_FORMAT = "pcm_24000"
BAILIAN_SPEECH_RATE = 1.0
BAILIAN_VOLUME = 50
BAILIAN_PITCH_RATE = 1.0
BAILIAN_INSTRUCTION = ""
BAILIAN_EMOTION_INSTRUCTIONS = {
    "angry": "你现在很生气，声音带着明显的怒意，语气严厉强势，说话节奏加快。",
    "sad": "你现在有点低落和难过，声音微微发颤，语气带着淡淡的忧伤，说话缓慢。",
    "excited": "你现在非常兴奋开心，声音轻快雀跃，音调上扬充满活力。",
    "teasing": "你现在正在调戏挑逗，声音带着狡黠的笑意，语气轻佻撩人。",
    "worried": "你现在有点担心和关切，声音温柔中带着紧张，语气小心翼翼。",
    "loving": "你现在充满疼爱，声音格外温柔绵软，像在哄人一样。",
    "default": "",
}
EMOTION_TAG_RE = re.compile(r"^\s*\[emotion:(\w+)\]\s*(.*)$", re.DOTALL)
BAILIAN_CALL_TIMEOUT_MS = 30000
BAILIAN_AUDIO_QUEUE_MAX_CHUNKS = 16

BARGE_IN_ENABLED = True
BARGE_IN_PROB_THRESHOLD = 0.65
BARGE_IN_DB_THRESHOLD = 72
BARGE_IN_REQUIRED_HITS = 4
BARGE_IN_REQUIRED_MISSES = 18
BARGE_IN_MIN_SPEECH_DURATION = 0.25
BARGE_IN_TRIGGER_BUFFER_CHUNKS = 16
BARGE_IN_CAPTURE_REQUIRED_MISSES = 40
BARGE_IN_MAX_CAPTURE_SECONDS = 30.0

IS_WINDOWS = sys.platform == "win32"
