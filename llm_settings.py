import json
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from config import BASE_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE, MODEL_NAME
from summary_generation import (
    normalize_summary_char_limit,
    recommended_summary_char_limit,
)


SETTINGS_PATH = BASE_DIR / "llm_settings.json"

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "local": {
        "label": "本地 OpenAI 兼容",
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
        "model": MODEL_NAME,
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "model": "deepseek-v4-flash",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容",
        "base_url": "",
        "api_key": "",
        "model": "",
    },
}


def get_provider_options() -> Dict[str, str]:
    return {key: value["label"] for key, value in PROVIDER_PRESETS.items()}


def get_provider_defaults(provider: str) -> Dict[str, Any]:
    preset = PROVIDER_PRESETS.get(provider) or PROVIDER_PRESETS["custom"]
    return {
        "provider": provider if provider in PROVIDER_PRESETS else "custom",
        "base_url": preset["base_url"],
        "api_key": preset["api_key"],
        "model": preset["model"],
        "temperature": LLM_TEMPERATURE,
        "summary_max_chars": recommended_summary_char_limit(
            {"model": preset["model"]}
        ),
    }


def normalize_llm_settings(raw: Dict[str, Any] | None = None) -> Dict[str, Any]:
    raw = raw or {}
    provider = raw.get("provider") or "local"
    if provider not in PROVIDER_PRESETS:
        provider = "custom"

    defaults = get_provider_defaults(provider)
    try:
        temperature = float(raw.get("temperature", defaults["temperature"]))
    except (TypeError, ValueError):
        temperature = defaults["temperature"]
    model = str(raw.get("model") or defaults["model"]).strip()
    summary_max_chars = normalize_summary_char_limit(
        raw.get("summary_max_chars"),
        {"model": model},
    )

    return {
        "provider": provider,
        "base_url": str(raw.get("base_url") or defaults["base_url"]).strip(),
        "api_key": str(raw.get("api_key") or defaults["api_key"]).strip(),
        "model": model,
        "temperature": max(0.0, min(2.0, temperature)),
        "summary_max_chars": summary_max_chars,
    }


def estimate_summary_video_minutes(summary_max_chars: Any) -> tuple[int, int]:
    normalized = normalize_summary_char_limit(summary_max_chars, {})
    minimum_minutes = max(1, round(normalized / 750))
    maximum_minutes = max(minimum_minutes, round(normalized / 450))
    return minimum_minutes, maximum_minutes


def load_llm_settings() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return normalize_llm_settings()

    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            return normalize_llm_settings(json.load(f))
    except (OSError, json.JSONDecodeError):
        return normalize_llm_settings()


def save_llm_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_llm_settings(settings)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = SETTINGS_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    temp_path.replace(SETTINGS_PATH)
    return normalized


def create_llm_client(settings: Dict[str, Any] | None = None) -> OpenAI:
    active = normalize_llm_settings(settings or load_llm_settings())
    if not active["base_url"]:
        raise ValueError("请先填写 Base URL")
    if not active["model"]:
        raise ValueError("请先填写模型名称")
    return OpenAI(base_url=active["base_url"], api_key=active["api_key"] or "EMPTY")


def test_llm_connection(settings: Dict[str, Any]) -> str:
    active = normalize_llm_settings(settings)
    client = create_llm_client(active)
    response = client.chat.completions.create(
        model=active["model"],
        messages=[{"role": "user", "content": "请只回复 OK"}],
        temperature=0,
        max_tokens=8,
    )
    return (response.choices[0].message.content or "").strip()
