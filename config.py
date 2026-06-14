from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
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
TTS_CHUNK_MIN_CHARS = 40
TTS_CHUNK_MAX_CHARS = 150
TTS_TEXT_NORMALIZATION = True
TTS_URL_REPLACEMENT = "这里有一个链接"
TTS_CODE_REPLACEMENT = "这里有一段代码"
SILICONFLOW_TTS_CHUNK_MIN_CHARS = 40
SILICONFLOW_TTS_CHUNK_MAX_CHARS = 150
ASR_TIMEOUT = 20
CLAUDE_RESPONSE_TIMEOUT = 240
TTS_BATCH_TIMEOUT = 0.3
TTS_BATCH_MAX_CHARS = 150

BARGE_IN_ENABLED = True
BARGE_IN_PROB_THRESHOLD = 0.65
BARGE_IN_DB_THRESHOLD = 72
BARGE_IN_REQUIRED_HITS = 4
BARGE_IN_REQUIRED_MISSES = 18
BARGE_IN_MIN_SPEECH_DURATION = 0.25
BARGE_IN_TRIGGER_BUFFER_CHUNKS = 16
BARGE_IN_CAPTURE_REQUIRED_MISSES = 40
BARGE_IN_MAX_CAPTURE_SECONDS = 30.0

SERVER_HOST = "localhost"
SERVER_PORT = 12394

MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSE_VOICE_MODEL = MODEL_DIR / "model.int8.onnx"
SENSE_VOICE_TOKENS = MODEL_DIR / "tokens.txt"

SAY_VOICE = "Tingting"
SAY_RATE = 200
TTS_PROVIDER = "say"

TTS_CONFIG_FILE = BASE_DIR / "tts_config.local.json"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/audio/speech"
SILICONFLOW_API_KEY_ENV = "SILICONFLOW_API_KEY"
SILICONFLOW_DEFAULT_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
SILICONFLOW_DEFAULT_VOICE = "FunAudioLLM/CosyVoice2-0.5B:anna"
SILICONFLOW_SAMPLE_RATE = 32000
SILICONFLOW_RESPONSE_FORMAT = "mp3"
SILICONFLOW_STREAM = True
SILICONFLOW_SPEED = 1.0
SILICONFLOW_GAIN = 0
SILICONFLOW_MOSS_DEFAULT_MODEL = "fnlp/MOSS-TTSD-v0.5"
SILICONFLOW_MOSS_DEFAULT_VOICE = "fnlp/MOSS-TTSD-v0.5:anna"
SILICONFLOW_MOSS_SAMPLE_RATE = 32000
SILICONFLOW_MOSS_RESPONSE_FORMAT = "mp3"
SILICONFLOW_MOSS_STREAM = True
SILICONFLOW_MOSS_SPEED = 1.1
SILICONFLOW_MOSS_GAIN = 0.0
SILICONFLOW_TIMEOUT = 30

BAILIAN_API_KEY_ENV = "DASHSCOPE_API_KEY"
BAILIAN_WEBSOCKET_API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
BAILIAN_DEFAULT_MODEL = "cosyvoice-v3-flash"
BAILIAN_DEFAULT_VOICE = "longanli_v3"
BAILIAN_AUDIO_FORMAT = "mp3"
BAILIAN_SPEECH_RATE = 1.0
BAILIAN_VOLUME = 50
BAILIAN_PITCH_RATE = 1.0

CLAUDE_BIN = "claude"
CLAUDE_MODEL = "sonnet"
CLAUDE_FALLBACK_MODEL = "fable"
VOICE_APPEND_SYSTEM_PROMPT = """
你正在通过语音和用户对话。默认用低思考、快速短答：先直接回应用户的话，不要铺长篇分析，不要输出思考过程。
普通闲聊每次回复控制在 1-3 小段，每段适合一口气念完；如果内容较多，分几次自然追问或分段说，不要一次塞满。
当用户的问题需要实时信息、搜索、文件/项目上下文、MCP 或 skill 能力时，可以自主调用可用工具完成查询；完成后用适合语音朗读的短句总结。
只有当收到明确表达“好好想想”“认真分析”“仔细推理”“详细讲讲”“帮我深入想”等意思时，才展开更完整的分析。
""".strip()
FAST_REPLY_PREFIX = (
    "【语音模式：默认快速短答。请直接回答，控制在1-3小段，每段短一点；"
    "不要长篇铺陈，不要展示推理过程。】\n"
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
CLAUDE_FLAGS = [
    "-p",
    "--verbose",
    "--no-session-persistence",
    "--effort",
    "low",
    "--model",
    CLAUDE_MODEL,
    "--fallback-model",
    CLAUDE_FALLBACK_MODEL,
    "--permission-mode",
    "auto",
    "--append-system-prompt",
    VOICE_APPEND_SYSTEM_PROMPT,
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
]
