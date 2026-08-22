"""全局音频处理设置与任务参数快照。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from config import BASE_DIR


SETTINGS_PATH = BASE_DIR / "processing_settings.json"
AUTO_SPEAKER_COUNT = -1
MIN_SPEAKER_COUNT = 2
MAX_SPEAKER_COUNT = 20

DEFAULT_PROCESSING_SETTINGS = {
    "use_vocal_separation": False,
    "use_speaker_diarization": False,
    "speaker_count": 2,
}


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "是", "开启"}:
            return True
        if normalized in {"false", "0", "no", "off", "否", "关闭", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def normalize_speaker_count(value: Any, default: int = 2) -> int:
    if isinstance(value, str) and value.strip().lower() in {
        "auto",
        "automatic",
        "自动",
    }:
        return AUTO_SPEAKER_COUNT

    try:
        count = int(value)
    except (TypeError, ValueError):
        return default

    if count == AUTO_SPEAKER_COUNT:
        return AUTO_SPEAKER_COUNT
    if MIN_SPEAKER_COUNT <= count <= MAX_SPEAKER_COUNT:
        return count
    return default


def normalize_processing_settings(
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw or {}
    return {
        "use_vocal_separation": _normalize_bool(
            raw.get("use_vocal_separation"),
            DEFAULT_PROCESSING_SETTINGS["use_vocal_separation"],
        ),
        "use_speaker_diarization": _normalize_bool(
            raw.get("use_speaker_diarization"),
            DEFAULT_PROCESSING_SETTINGS["use_speaker_diarization"],
        ),
        "speaker_count": normalize_speaker_count(
            raw.get("speaker_count"),
            DEFAULT_PROCESSING_SETTINGS["speaker_count"],
        ),
    }


def load_processing_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or SETTINGS_PATH
    if not settings_path.exists():
        return normalize_processing_settings()

    try:
        with settings_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError):
        return normalize_processing_settings()
    return normalize_processing_settings(data)


def save_processing_settings(
    settings: Mapping[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    normalized = normalize_processing_settings(settings)
    settings_path = path or SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings_path.with_suffix(settings_path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(normalized, file, ensure_ascii=False, indent=2)
            file.flush()
        temp_path.replace(settings_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return normalized


def merge_processing_defaults(
    task_options: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """把当前设置固化到新任务；本机降噪设置拥有最终控制权。"""
    merged = dict(task_options or {})
    defaults = normalize_processing_settings(
        settings if settings is not None else load_processing_settings()
    )
    merged.setdefault(
        "use_speaker_diarization", defaults["use_speaker_diarization"]
    )
    merged.setdefault("speaker_count", defaults["speaker_count"])

    # 降噪是本机算力策略，不允许网页插件或其他任务入口覆盖。
    merged["use_vocal_separation"] = defaults["use_vocal_separation"]

    merged["use_speaker_diarization"] = _normalize_bool(
        merged["use_speaker_diarization"],
        defaults["use_speaker_diarization"],
    )
    merged["speaker_count"] = normalize_speaker_count(
        merged["speaker_count"],
        defaults["speaker_count"],
    )
    return merged
