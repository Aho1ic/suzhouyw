"""集中管理项目配置，敏感参数从环境变量读取，避免硬编码。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ===== 路径 =====
TASK_DIR = ROOT / "task"
RESULT_DIR = ROOT / "result1"
LOG_DIR = ROOT / "logs"
WEIGHTS_DIR = ROOT / "weights"
CONFIG_DIR = ROOT / "config"

# ===== 模型 =====
DET_MODEL_PATH = WEIGHTS_DIR / _env("DET_MODEL", "antenna_best.pt")
SEG_MODEL_PATH = WEIGHTS_DIR / _env("SEG_MODEL", "blue_seg_best.pt")
TRACKER_CONFIG_PATH = CONFIG_DIR / _env("TRACKER_CFG", "botsort.yaml")

# ===== RTMP 默认地址 =====
DEFAULT_RTMP_URL = _env(
    "DEFAULT_RTMP_URL",
    "rtmp://127.0.0.1:1935/ly/1581F6Q8X253H00G06BQ",
)

# ===== 检测阈值 =====
DET_CONF = _env_float("DET_CONF", 0.5)
DET_IOU = _env_float("DET_IOU", 0.4)
SEG_CONF = _env_float("SEG_CONF", 0.5)
SEG_IOU = _env_float("SEG_IOU", 0.45)
MIN_TRACK_FRAMES = _env_int("MIN_TRACK_FRAMES", 90)
MAX_LOST_FRAMES = _env_int("MAX_LOST_FRAMES", 30)

# ===== 业务字段（可被 task json / 入参覆盖） =====
DEVICE_NAME = _env("DEVICE_NAME", "苏州AI识别")
DEVICE_ID = _env("DEVICE_ID", "e5819ec6-8499-b692-c5ab-db60a2aff753")
APP_NAME = _env("APP_NAME", "天线资产盘点")
APP_ID = _env("APP_ID", "218")
SRC_NAME = _env("SRC_NAME", "相城区机巢rtsp")
SRC_ID = _env("SRC_ID", "603b011a67f6a92c276a9d1259e1a614")
DEFAULT_TASK_NAME = _env("DEFAULT_TASK_NAME", "天线资产盘点")
DEFAULT_TASK_ID = _env("DEFAULT_TASK_ID", "7")
DEFAULT_BOX_ID = _env("DEFAULT_BOX_ID", "20")

# ===== 上传服务（敏感凭证强制从 env 读取，未设置则禁用上传） =====
UPLOAD_URL = _env("UPLOAD_URL", "")
UPLOAD_APP_KEY_ID = _env("UPLOAD_APP_KEY_ID", "")
UPLOAD_APP_KEY_SECRET = _env("UPLOAD_APP_KEY_SECRET", "")
UPLOAD_SYS_ID = _env("UPLOAD_SYS_ID", "11")
UPLOAD_TYPE = _env("UPLOAD_TYPE", "1")
UPLOAD_TIMEOUT = _env_int("UPLOAD_TIMEOUT", 5)


def upload_enabled() -> bool:
    """上传开关：必须同时配置 URL 和凭证。每次调用时检查，避免模块导入时副作用。"""
    url = os.environ.get("UPLOAD_URL", "")
    key_id = os.environ.get("UPLOAD_APP_KEY_ID", "")
    key_secret = os.environ.get("UPLOAD_APP_KEY_SECRET", "")
    return bool(url and key_id and key_secret)


# ===== Flask 服务 =====
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 58081)
STREAM_HOST = _env("STREAM_HOST", "0.0.0.0")
STREAM_PORT = _env_int("STREAM_PORT", 19053)

# ===== 流状态轮询 =====
STREAM_CHECK_INTERVAL = _env_int("STREAM_CHECK_INTERVAL", 30)
STREAM_CHECK_WORKERS = _env_int("STREAM_CHECK_WORKERS", 4)

# ===== CORS（生产建议指定具体源） =====
CORS_ORIGIN = _env("CORS_ORIGIN", "*")
