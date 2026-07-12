"""项目统一配置。

普通用户通常只需要修改本文件，不必再分别编辑 app.py、worker.py、
whisper_worker.py、server.py、db.py 和 launcher.py。
"""

from pathlib import Path

# ================= 项目根目录 =================
BASE_DIR = Path(__file__).resolve().parent

# ================= 本地任务服务 =================
API_HOST = "127.0.0.1"
API_PORT = 8000
TASK_API_URL = f"http://{API_HOST}:{API_PORT}/api/tasks"
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
DOWNLOAD_MAX_HEIGHT = 1080
COOKIES_FILE = BASE_DIR / "cookies.txt"

# 本地导入、黑框拖拽等未明确传参的任务，默认是否开启人声分离。
# 油猴 V6.3 会在请求中主动传入 true，因此不受这里 False 的影响。
DEFAULT_USE_VOCAL_SEPARATION = False

# ================= 数据目录 =================
SRT_VAULT_DIR = BASE_DIR / "srt_vault"
LOCAL_UPLOADS_DIR = BASE_DIR / "local_uploads"
VIDEO_VAULT_DIR = BASE_DIR / "video_downloads"
DB_PATH = BASE_DIR / "tasks.db"

# ================= Whisper 与人声分离 =================
WHISPER_MODEL_DIR = Path(r"G:\WhisperModels")
VOCAL_MODEL_DIR = WHISPER_MODEL_DIR / "audio-separator-models"
AUDIO_WORKSPACE_DIR = Path(r"D:\AI_Workspace")

WHISPER_MODEL_NAME = "large-v3"
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
for directory in (SRT_VAULT_DIR, LOCAL_UPLOADS_DIR, VIDEO_VAULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
