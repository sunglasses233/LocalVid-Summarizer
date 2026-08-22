"""项目统一配置。

分享版默认把数据、模型、工具和临时文件放在项目目录中，移动整个目录后
仍可工作。安装器或高级用户也可以通过 ``AI_VIDEO_*`` 环境变量覆盖路径。
"""

import os
from pathlib import Path

# ================= 项目根目录 =================
BASE_DIR = Path(__file__).resolve().parent


def resolve_config_path(env_name: str, default: Path) -> Path:
    """解析可由环境变量覆盖的路径。

    相对路径始终以项目根目录为基准，避免启动命令的当前目录影响结果。
    空白环境变量视为未设置；外层引号会被清理以兼容 Windows 批处理传值。
    """

    raw_value = os.environ.get(env_name, "").strip().strip('"').strip("'")
    if raw_value:
        expanded = os.path.expandvars(os.path.expanduser(raw_value))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
    else:
        candidate = Path(default)
    return candidate.resolve(strict=False)

# ================= 本地任务服务 =================
API_HOST = "127.0.0.1"
API_PORT = 8000
TASK_API_URL = f"http://{API_HOST}:{API_PORT}/api/tasks"
STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_PORT = 8501
STREAMLIT_MAX_UPLOAD_MB = 2048

# ================= 大模型接口 =================
# 兼容 OpenAI API 的本地或在线接口，例如 LM Studio、Ollama 兼容层等。
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
MODEL_NAME = "local_model"
LLM_TEMPERATURE = 0.3

# 首次自动总结和打开视频后的后续对话，统一使用这个字幕字符上限。
# 注意：这是字符数限制，不是模型的精确 Token 上限。
MAX_CHARS_LIMIT = 26000

# 仅用于网页端连续对话的历史消息上限，同样是近似估算值。
MAX_HISTORY_TOKENS = 28000

# ================= 下载设置 =================
# 没有通过任务 options 明确指定时，使用这里的默认值。
AUTO_DOWNLOAD_VIDEO = True
DOWNLOAD_MAX_HEIGHT = 720
DATA_DIR = resolve_config_path("AI_VIDEO_DATA_DIR", BASE_DIR / "data")
RUNTIME_DIR = resolve_config_path("AI_VIDEO_RUNTIME_DIR", BASE_DIR / "runtime")
MODEL_ROOT_DIR = resolve_config_path("AI_VIDEO_MODEL_DIR", BASE_DIR / "models")
TOOLS_DIR = resolve_config_path("AI_VIDEO_TOOLS_DIR", BASE_DIR / "tools")

COOKIES_FILE = DATA_DIR / "cookies.txt"

# 历史任务或异常 options 缺少该字段时的兜底值。
# 新任务统一以 processing_settings.json 中保存的本地设置为准。
DEFAULT_USE_VOCAL_SEPARATION = False

# ================= 数据目录 =================
SRT_VAULT_DIR = DATA_DIR / "srt_vault"
LOCAL_UPLOADS_DIR = DATA_DIR / "local_uploads"
VIDEO_VAULT_DIR = DATA_DIR / "video_downloads"
DB_PATH = DATA_DIR / "tasks.db"
KNOWLEDGE_CARD_DIR = DATA_DIR / "知识卡片"
KNOWLEDGE_CARD_FILES_DIR = KNOWLEDGE_CARD_DIR / "卡片"
KNOWLEDGE_CARD_DRAFTS_DIR = KNOWLEDGE_CARD_DIR / "_drafts"
KNOWLEDGE_CARD_HISTORY_DIR = KNOWLEDGE_CARD_DIR / "_history"
KNOWLEDGE_CARD_INDEX_PATH = KNOWLEDGE_CARD_DIR / "index.json"

# ================= Whisper 与人声分离 =================
WHISPER_MODEL_DIR = resolve_config_path(
    "AI_VIDEO_WHISPER_MODEL_DIR", MODEL_ROOT_DIR / "whisper"
)
VOCAL_MODEL_DIR = resolve_config_path(
    "AI_VIDEO_VOCAL_MODEL_DIR", MODEL_ROOT_DIR / "audio_separator"
)
SPEAKER_MODEL_DIR = resolve_config_path(
    "AI_VIDEO_SPEAKER_MODEL_DIR", MODEL_ROOT_DIR / "speaker_diarization"
)
AUDIO_WORKSPACE_DIR = resolve_config_path(
    "AI_VIDEO_AUDIO_WORKSPACE_DIR", RUNTIME_DIR / "audio_workspace"
)
FFMPEG_BIN_DIR = resolve_config_path(
    "AI_VIDEO_FFMPEG_BIN_DIR", TOOLS_DIR / "ffmpeg" / "bin"
)
FFMPEG_EXECUTABLE = FFMPEG_BIN_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

WHISPER_MODEL_NAME = "large-v3"
WHISPER_MODEL_REPO_ID = "Systran/faster-whisper-large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LOCAL_FILES_ONLY = True
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = True
WHISPER_MIN_SILENCE_MS = 500
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False
WHISPER_WORD_TIMESTAMPS = True

VOCAL_SEPARATOR_MODEL_NAME = "Kim_Vocal_2.onnx"
HF_ENDPOINT = "https://hf-mirror.com"

# ================= 英语/表音文字断句 =================
EN_MAX_CHARS = 70
EN_ABSOLUTE_MAX_CHARS = 90
EN_MAX_GAP = 0.8
EN_TOLERANCE_WORDS = 2

# ================= 统一创建项目数据目录 =================
for directory in (
    DATA_DIR,
    RUNTIME_DIR,
    MODEL_ROOT_DIR,
    SRT_VAULT_DIR,
    LOCAL_UPLOADS_DIR,
    VIDEO_VAULT_DIR,
    KNOWLEDGE_CARD_FILES_DIR,
    KNOWLEDGE_CARD_DRAFTS_DIR,
    KNOWLEDGE_CARD_HISTORY_DIR,
    WHISPER_MODEL_DIR,
    VOCAL_MODEL_DIR,
    SPEAKER_MODEL_DIR,
    AUDIO_WORKSPACE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
