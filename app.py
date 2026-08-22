import os
import sqlite3
import json
import copy
import hashlib
import html
import re
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
import streamlit.components.v1 as components

from library_ui import (
    build_library_file_labels,
    build_video_player_fingerprint,
    build_video_subtitle_tracks,
    clean_library_file_label,
)
from llm_settings import (
    create_llm_client,
    estimate_summary_video_minutes,
    get_provider_defaults,
    get_provider_options,
    load_llm_settings,
    save_llm_settings,
    test_llm_connection,
)
from processing_settings import (
    AUTO_SPEAKER_COUNT,
    load_processing_settings,
    save_processing_settings,
)
from knowledge_cards import (
    CARD_TYPES,
    CARD_TYPE_LABELS,
    CardBundle,
    CardSource,
    KnowledgeCardError,
    commit_bundle,
    derive_source_id,
    derive_video_title,
    draft_path,
    load_committed_bundle,
    load_draft,
    load_index,
    now_china_iso,
    revise_bundle,
    revise_card,
    save_draft,
    update_source_path,
)
from subtitle_correction import SubtitleCorrectionError, compare_srt_versions
from summary_generation import (
    MAX_SUMMARY_CHAR_LIMIT,
    MIN_SUMMARY_CHAR_LIMIT,
    SUMMARY_CHAR_LIMIT_STEP,
    SUMMARY_PROFILES,
    build_summary_messages,
    classify_alternative_summary_profile,
    limit_transcript_to_complete_segments,
    resolve_summary_baseline_route,
    resolve_summary_char_limit,
)

# ================= 统一配置 =================
from config import (
    AUTO_DOWNLOAD_VIDEO,
    COOKIES_FILE,
    DB_PATH,
    DOWNLOAD_MAX_HEIGHT,
    MAX_HISTORY_TOKENS,
    SRT_VAULT_DIR,
    TASK_API_URL,
    VIDEO_VAULT_DIR,
)

DELIMITER = "\n\n====================== AI_CHAT_HISTORY ======================\n"
KNOWLEDGE_CARD_JOB_API_URL = (
    TASK_API_URL.rsplit("/api/tasks", 1)[0] + "/api/knowledge-card-jobs"
)

QA_MODE_VIDEO_FIRST = "视频优先"
QA_MODE_FREE_EXPLORE = "自由探索"
QA_MODE_OPTIONS = (QA_MODE_VIDEO_FIRST, QA_MODE_FREE_EXPLORE)
MAIN_VIEW_CHAT = "AI对话"
MAIN_VIEW_KNOWLEDGE = "知识卡片"
MAIN_VIEW_OPTIONS = (MAIN_VIEW_CHAT, MAIN_VIEW_KNOWLEDGE)
KNOWLEDGE_CARD_PRESET = "__OPEN_KNOWLEDGE_CARD_WORKBENCH__"
KNOWLEDGE_CARD_AUTO_TYPE = "自动判断"
KNOWLEDGE_CARD_TYPE_OPTION_LABELS = {
    KNOWLEDGE_CARD_AUTO_TYPE: "自动判断（推荐）",
    "action_guide": "行动指南（解决问题）",
    "resource_index": "资源索引（多个项目）",
    "viewpoint_profile": "观点档案（作者主张）",
    "general_knowledge": "通用知识（单一主题）",
}
KNOWLEDGE_CARD_TYPE_DESCRIPTIONS = {
    KNOWLEDGE_CARD_AUTO_TYPE: "自动判断：根据视频内容选择类型；普通内容生成 1 张，多个项目或工具生成总索引和独立子卡。",
    "action_guide": "行动指南：适合维权、操作教程和决策问题；生成 1 张，突出步骤、证据、升级路径与风险。",
    "resource_index": "资源索引：适合一期介绍多个项目、软件或网站的视频；生成 1 张总索引和每个资源的独立子卡。",
    "viewpoint_profile": "观点档案：适合两性、社会评论和作者观点类视频；生成 1 张，整理主张、论据、前提与反方视角。",
    "general_knowledge": "通用知识：适合围绕一个核心主题的科普或干货；生成 1 张，次要内容作为卡片内章节。",
}

PRESETS = {
    "📝 总结": "请仔细阅读文本，按照以下框架输出：\n1. 🎯 **核心摘要**\n2. 🗺️ **逻辑脉络**\n3. ⚖️ **对比/亮点**（如有对比，【必须】使用标准的换行 Markdown 表格，如：\n| 维度 | A | B |\n|---|---|---|\n| 1 | 2 | 3 |）\n\n【排版纪律】（不要输出规则本身）：\n- 必须使用 Emoji 和**加粗**。\n- 所有要点必须带时间戳，严格使用单圆括号包裹，如 `(01:30)` 或 `(01:30-02:15)`，严禁分开写。",
    "🗺️ 大纲": "请梳理视频的逻辑脉络，输出一份清晰的、带有 Emoji 装饰的思维导图大纲，**必须**在每个节点后标注对应的时间戳 `(MM:SS)`。",
    "💬 大白话": "请用通俗易懂的大白话、类似朋友聊天的口吻总结这个视频，提炼出 3-4 个最接地气的要点，并在关键句后附带时间戳 `(MM:SS)`。",
    "🤫 是暗广吗？": "请像一位资深的内容审核员一样，分析这期视频是否可能是隐性广告。请列出**怀疑点**和**澄清点**的对比表格，并在引用证据时标注时间戳 `(MM:SS)`。",
    "📌 要点": "请提取视频中的核心要点，用层级分明的 Markdown 列表呈现，加粗核心词，并逐条附带时间戳 `(MM:SS)`。",
    "❓ Q&A": "请根据视频内容，整理出观众最可能关心的 3-5 个问题并解答。使用一问一答的格式，并在答案的依据处标注时间戳 `(MM:SS)`。",
    "生成知识卡片": KNOWLEDGE_CARD_PRESET,
    "分析": "请客观分析这个视频讲得好的地方和讲得不好的地方。分别从内容准确性、论证逻辑、信息密度、表达结构和证据充分性进行评价，引用具体内容作为依据并附带时间戳 `(MM:SS)` 或 `(HH:MM:SS)`。明确区分视频中的事实、作者观点和你的分析判断。",
    "自定义": None,
}

os.makedirs(SRT_VAULT_DIR, exist_ok=True)
# ================= 工具函数 =================
def build_qa_system_prompt(transcript, mode):
    if mode == QA_MODE_FREE_EXPLORE:
        mode_instructions = (
            "【回答模式：自由探索】\n"
            "1. 视频字幕是重要背景资料，但不是知识边界。优先理解用户的真实意图，可以结合模型已有知识进行解释、比较、质疑、推演和创意扩展。\n"
            "2. 清楚区分视频明确表达的内容、基于视频的推断和模型提供的扩展知识，不得把扩展知识说成视频原意。\n"
            "3. 当前未启用联网搜索。遇到最新、实时或可能变化的信息时，不要声称已经联网核实，并提醒用户相关信息可能不是最新。\n"
        )
    else:
        mode_instructions = (
            "【回答模式：视频优先】\n"
            "1. 以视频字幕为主要事实依据，准确回答用户的问题。可以使用模型已有知识补充背景、解释概念或帮助理解，但不得把补充内容说成视频原意。\n"
            "2. 如果字幕没有提供足够依据，应明确区分视频内容与扩展知识；无法可靠回答时，如实说明不确定之处。\n"
        )

    return (
        "你是一个专业的视频内容分析与思考助手。\n\n"
        f"{mode_instructions}"
        "【时间戳规则】\n"
        "1. 字幕中方括号内的内容（如 [01:30]）代表原视频时间戳。归纳视频要点、梳理视频大纲或引用视频原文时，必须在关键信息后附带对应时间戳。\n"
        "2. 时间戳必须使用圆括号格式 `(MM:SS)` 或 `(HH:MM:SS)`，例如：`屏幕与 UI 检查是否流畅 (01:30)`。\n"
        "3. 只有能够从字幕中定位的信息才能附带时间戳；模型扩展知识不得伪造时间戳。\n\n"
        "【视频字幕开始】\n"
        f"{transcript}\n"
        "【视频字幕结束】"
    )


def get_chat_segments(messages):
    """Return deletable chat segments as lists of message indexes."""
    segments = []
    index = 0
    while index < len(messages):
        role = messages[index].get("role")
        if role == "system":
            index += 1
            continue

        segment = [index]
        if (
            role == "user"
            and index + 1 < len(messages)
            and messages[index + 1].get("role") == "assistant"
        ):
            segment.append(index + 1)
            index += 1

        segments.append(segment)
        index += 1

    return segments


def get_chat_segment_fingerprint(messages, message_indexes):
    segment_payload = [
        {
            "role": messages[index].get("role", ""),
            "content": messages[index].get("content", ""),
        }
        for index in message_indexes
    ]
    serialized_payload = json.dumps(
        segment_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()[:16]


def get_segment_summary_profile(messages, message_indexes):
    for index in reversed(message_indexes):
        metadata = messages[index].get("summary_profile")
        if not isinstance(metadata, dict):
            continue
        profile = str(metadata.get("profile") or "").strip()
        if profile in SUMMARY_PROFILES:
            normalized = dict(metadata)
            normalized["profile"] = profile
            normalized["label"] = str(
                metadata.get("label") or SUMMARY_PROFILES[profile]
            ).strip()
            return normalized
    return None


def resolve_library_file_selection(
    selected_file,
    available_files,
    current_file,
    preserve_empty_selection=False,
):
    if selected_file in available_files:
        return selected_file
    if preserve_empty_selection and selected_file is None:
        return None
    if current_file in available_files:
        return current_file
    return available_files[0] if available_files else None


# [编程大师2.0 终极版]：物理目录深度扫描与自动迁移引擎
def get_library_data():
    library = {}
    os.makedirs(SRT_VAULT_DIR, exist_ok=True)
    
    # 【防御性数据迁移】：扫描根目录，发现带虚拟标签的老文件，立刻进行物理“搬家”
    for file in os.listdir(SRT_VAULT_DIR):
        if file.endswith('.srt'):
            old_path = os.path.join(SRT_VAULT_DIR, file)
            if os.path.isfile(old_path):
                try:
                    with open(old_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    parts = content.split(DELIMITER)
                    if len(parts) > 1:
                        data = json.loads(parts[1])
                        if isinstance(data, dict):
                            virtual_col = data.get("collection", "默认收藏夹")
                            # 净化文件夹名，防止操作系统报错
                            safe_col = re.sub(r'[\\/*?:"<>|]', "", virtual_col).strip()
                            if safe_col and safe_col != "默认收藏夹":
                                target_dir = os.path.join(SRT_VAULT_DIR, safe_col)
                                os.makedirs(target_dir, exist_ok=True)
                                new_path = os.path.join(target_dir, file)
                                os.rename(old_path, new_path) # 物理剪切
                except Exception:
                    pass

    # 保留空收藏夹，使文件移走后侧栏仍能停留在原目录。
    for entry in os.scandir(SRT_VAULT_DIR):
        if entry.is_dir():
            library.setdefault(entry.name, [])

    # 【核心逻辑】：物理穿透扫描 (OS Walk)，确立文件夹名为唯一的真理
    for root, dirs, files in os.walk(SRT_VAULT_DIR):
        for file in files:
            if file.endswith('.srt'):
                # 获取相对路径，例如: "AI前沿/test_123.srt" 或 "test_123.srt"
                rel_dir = os.path.relpath(root, SRT_VAULT_DIR)
                if rel_dir == '.':
                    collection = "默认收藏夹"
                    rel_path = file
                else:
                    collection = rel_dir.split(os.sep)[0]
                    rel_path = os.path.join(rel_dir, file)
                
                if collection not in library:
                    library[collection] = []
                
                # 记录物理文件的最后修改时间，用于排序
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                library[collection].append((rel_path, mtime))
                
    # 对每个分类进行时间倒序排序
    for col in library:
        library[col].sort(key=lambda x: x[1], reverse=True)
        library[col] = [x[0] for x in library[col]] # 仅保留相对路径
        
    return library


# [编程大师2.0 终极版]：智能资产伴生下载引擎
def download_high_res_video(url, save_dir, base_name):
    import yt_dlp 
    
    # 移除固定的 ext=mp4 限制，允许拉取最高画质的 Dash 流，然后交由 FFmpeg 强行无损封装为 mp4
    format_str = f'bestvideo[height<={DOWNLOAD_MAX_HEIGHT}]+bestaudio/best[height<={DOWNLOAD_MAX_HEIGHT}]/best'
    
    ydl_opts = {
        'format': format_str,
        'outtmpl': os.path.join(save_dir, f"{base_name}.%(ext)s"),
        'merge_output_format': 'mp4',
        'cookiefile': str(COOKIES_FILE), # 依赖最新 cookies 突破 B站清晰度限制
        'quiet': False,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            return True
    except Exception as e:
        raise Exception(f"下载引擎报错: {e}")
        
# [编程大师2.0 新增]：统一的数据固化保存函数 (遵循 DRY 原则，降低冗余)
def save_current_metadata():
    if not st.session_state.current_file:
        return False
    file_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
    temp_path = file_path + ".tmp"
    metadata_to_save = {
        "source_url": st.session_state.source_url,
        "collection": st.session_state.current_collection,
        "notes": st.session_state.current_notes,
        "history": [m for m in st.session_state.messages if m["role"] != "system"],
        "subtitle_correction": st.session_state.subtitle_correction,
        "video_metadata": st.session_state.video_metadata,
    }
    metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(st.session_state.base_srt_content + DELIMITER + metadata_json)
        os.replace(temp_path, file_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        st.error(f"保存元数据失败: {e}")
        return False


def format_source_count(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return ""
    if number >= 100_000_000:
        compact = f"{number / 100_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}亿"
    if number >= 10_000:
        compact = f"{number / 10_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}万"
    return f"{int(number):,}"


def format_source_date(value):
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        china_timezone = timezone(timedelta(hours=8))
        return parsed.astimezone(china_timezone).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def build_video_metadata_caption(metadata):
    if not isinstance(metadata, dict):
        return ""
    author = metadata.get("author") if isinstance(metadata.get("author"), dict) else {}
    statistics = (
        metadata.get("statistics")
        if isinstance(metadata.get("statistics"), dict)
        else {}
    )
    parts = []
    author_name = str(author.get("name") or "").strip()
    if author_name:
        parts.append(f"作者：@{author_name.lstrip('@')}")
    published_date = format_source_date(metadata.get("published_at"))
    if published_date:
        parts.append(f"发布：{published_date}")

    statistic_labels = (
        ("like_count", "点赞"),
        ("comment_count", "评论"),
        ("collect_count", "收藏"),
        ("share_count", "分享"),
        ("play_count", "播放"),
    )
    for key, label in statistic_labels:
        raw_value = statistics.get(key)
        if key == "play_count":
            try:
                if float(raw_value) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        count = format_source_count(raw_value)
        if count:
            parts.append(f"{label}：{count}")
    return " · ".join(parts)


def generate_alternative_summary(current_profile):
    title = derive_video_title(st.session_state.current_file)
    transcript = st.session_state.transcript
    active_llm_settings = load_llm_settings()
    llm_client = create_llm_client(active_llm_settings)

    progress_placeholder = st.empty()
    preview_placeholder = st.empty()
    if current_profile in SUMMARY_PROFILES:
        progress_placeholder.info("正在排除当前板块并选择最接近的视频结构...")
    else:
        progress_placeholder.info("原总结已删除，正在重新识别视频的主要内容结构...")
    baseline_route = resolve_summary_baseline_route(
        title,
        transcript,
        current_profile,
        llm_client,
        active_llm_settings,
    )
    if current_profile not in SUMMARY_PROFILES:
        progress_placeholder.info(
            f"已识别原结构为「{baseline_route.label}」，正在选择其他板块..."
        )
    route = classify_alternative_summary_profile(
        title,
        transcript,
        baseline_route.profile,
        llm_client,
        active_llm_settings,
    )
    progress_placeholder.info(f"已选择「{route.label}」，正在重新生成总结...")
    messages = build_summary_messages(title, transcript, route)

    full_response = ""
    try:
        response_stream = llm_client.chat.completions.create(
            model=active_llm_settings["model"],
            messages=messages,
            stream=True,
            temperature=active_llm_settings["temperature"],
        )
        for chunk in response_stream:
            choices = getattr(chunk, "choices", None)
            delta = getattr(choices[0], "delta", None) if choices else None
            content = getattr(delta, "content", None) if delta else None
            if content:
                full_response += content
                preview_placeholder.markdown(
                    full_response + "▌",
                    unsafe_allow_html=True,
                )
    except Exception:
        if full_response:
            raise
        response = llm_client.chat.completions.create(
            model=active_llm_settings["model"],
            messages=messages,
            temperature=active_llm_settings["temperature"],
        )
        full_response = response.choices[0].message.content or ""

    if not full_response.strip():
        raise RuntimeError("模型没有返回新的总结内容")
    preview_placeholder.markdown(full_response, unsafe_allow_html=True)
    progress_placeholder.success(f"已按「{route.label}」完成重新总结。")
    return baseline_route, route, full_response.strip()

def extract_pure_text_from_srt(srt_content):
    """
    [编程大师2.0 修复版]：预扫描防突变 SRT 解析器
    """
    # 1. 预扫描：检测全片是否跨越 1 小时
    has_hours = False
    for line in srt_content.splitlines():
        if '-->' in line:
            start_raw = line.split('-->')[0].strip()
            if not start_raw.startswith("00:"):
                has_hours = True
                break
                
    result_lines = []
    current_time = ""
    
    for line in srt_content.splitlines():
        line = line.strip()
        if not line: continue
        if line.isdigit() and not current_time: continue  
            
        if '-->' in line:
            start_raw = line.split('-->')[0].strip()
            base_time = start_raw.split(',')[0].split('.')[0]
            
            # 【核心修复】：只有全片都不足 1 小时，才允许抹去 "00:"。
            # 一旦超过1小时，全片强锁 HH:MM:SS 格式，防止大模型发生格式惯性幻觉！
            if not has_hours and base_time.startswith("00:"):
                base_time = base_time[3:]
                
            current_time = base_time
            continue
            
        if current_time:
            result_lines.append(f"[{current_time}] {line}")
            current_time = "" 
        else:
            if result_lines:
                result_lines[-1] += f" {line}"
            else:
                result_lines.append(line)
                
    return " ".join(result_lines)


def load_subtitle_correction_changes(current_file, corrected_srt):
    if not current_file:
        return [], "未选择字幕文件"
    srt_path = os.path.join(SRT_VAULT_DIR, current_file)
    raw_backup_path = srt_path + ".raw.bak"
    if not os.path.exists(raw_backup_path):
        return [], "未找到校对前的原始字幕备份"
    try:
        with open(raw_backup_path, "r", encoding="utf-8") as file:
            original_srt = file.read()
        return compare_srt_versions(original_srt, corrected_srt), ""
    except (OSError, SubtitleCorrectionError) as error:
        return [], f"字幕修改记录读取失败：{error}"


def subtitle_timeline_label(timeline):
    start_time = str(timeline or "").split("-->", 1)[0].strip()
    start_time = start_time.split(",", 1)[0].split(".", 1)[0]
    if start_time.startswith("00:"):
        start_time = start_time[3:]
    return start_time or "未知时间"


def render_subtitle_correction_changes(changes):
    rows = []
    for change in changes:
        rows.append(
            '<div class="subtitle-diff-row">'
            f'<div class="subtitle-diff-time">{html.escape(subtitle_timeline_label(change.timeline))}</div>'
            '<div class="subtitle-diff-grid">'
            '<div><div class="subtitle-diff-label before-label">修改前</div>'
            f'<div class="subtitle-diff-text before-text">{html.escape(change.before)}</div></div>'
            '<div><div class="subtitle-diff-label after-label">修改后</div>'
            f'<div class="subtitle-diff-text after-text">{html.escape(change.after)}</div></div>'
            '</div></div>'
        )
    st.markdown(
        '<div class="subtitle-diff-list">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )

def calculate_tokens(messages):
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars * 1.2)

def get_payload_messages(messages):
    payload = [
        {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
        }
        for message in copy.deepcopy(messages)
        if message.get("role") in {"system", "user", "assistant"}
    ]
    current_tokens = calculate_tokens(payload)
    while current_tokens > MAX_HISTORY_TOKENS and len(payload) > 3:
        payload.pop(1) 
        payload.pop(1) 
        current_tokens = calculate_tokens(payload)
    return payload

def get_tasks_from_db():
    if not os.path.exists(DB_PATH): return []
    try:
        with sqlite3.connect(DB_PATH, timeout=2) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM video_tasks ORDER BY created_at DESC LIMIT 10")
            return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []

def get_srt_files():
    files = [f for f in os.listdir(SRT_VAULT_DIR) if f.endswith('.srt')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SRT_VAULT_DIR, x)), reverse=True)
    return files

def init_llm_settings_state():
    if st.session_state.get("llm_settings_initialized"):
        return
    settings = load_llm_settings()
    st.session_state.llm_provider = settings["provider"]
    st.session_state.llm_base_url = settings["base_url"]
    st.session_state.llm_api_key = settings["api_key"]
    st.session_state.llm_model = settings["model"]
    st.session_state.llm_temperature = float(settings["temperature"])
    st.session_state.llm_summary_max_chars = int(settings["summary_max_chars"])
    st.session_state.llm_settings_initialized = True

def apply_llm_provider_defaults():
    defaults = get_provider_defaults(st.session_state.get("llm_provider", "local"))
    st.session_state.llm_base_url = defaults["base_url"]
    st.session_state.llm_api_key = defaults["api_key"]
    st.session_state.llm_model = defaults["model"]
    st.session_state.llm_temperature = float(defaults["temperature"])
    st.session_state.llm_summary_max_chars = int(defaults["summary_max_chars"])

def get_llm_settings_from_state():
    return {
        "provider": st.session_state.get("llm_provider", "local"),
        "base_url": st.session_state.get("llm_base_url", ""),
        "api_key": st.session_state.get("llm_api_key", ""),
        "model": st.session_state.get("llm_model", ""),
        "temperature": st.session_state.get("llm_temperature", 0.3),
        "summary_max_chars": st.session_state.get(
            "llm_summary_max_chars",
            MIN_SUMMARY_CHAR_LIMIT,
        ),
    }

def render_llm_settings_panel():
    init_llm_settings_state()
    provider_options = get_provider_options()
    provider_keys = list(provider_options.keys())
    saved_settings = load_llm_settings()

    with st.expander("模型 / API 设置", expanded=False):
        st.selectbox(
            "供应商",
            provider_keys,
            format_func=lambda key: provider_options.get(key, key),
            key="llm_provider",
            on_change=apply_llm_provider_defaults,
        )
        st.text_input("Base URL", key="llm_base_url", placeholder="例如：https://api.deepseek.com")
        st.text_input("API Key", key="llm_api_key", type="password")
        st.text_input("模型名称", key="llm_model", placeholder="例如：deepseek-v4-flash")
        st.slider("Temperature", 0.0, 2.0, step=0.1, key="llm_temperature")
        st.number_input(
            "总结字幕字符上限",
            min_value=MIN_SUMMARY_CHAR_LIMIT,
            max_value=MAX_SUMMARY_CHAR_LIMIT,
            step=SUMMARY_CHAR_LIMIT_STEP,
            key="llm_summary_max_chars",
            help="控制自动总结、AI提问和重新总结可使用的字幕范围。",
        )
        minimum_minutes, maximum_minutes = estimate_summary_video_minutes(
            st.session_state.get("llm_summary_max_chars")
        )
        st.caption(
            f"约对应视频 {minimum_minutes}-{maximum_minutes} 分钟内容，"
            "实际范围会随字幕密度变化。"
        )

        save_col, test_col = st.columns(2)
        with save_col:
            if st.button("保存设置", use_container_width=True):
                previous_limit = saved_settings["summary_max_chars"]
                saved = save_llm_settings(get_llm_settings_from_state())
                saved_settings = saved
                if (
                    saved["summary_max_chars"] != previous_limit
                    and st.session_state.get("current_file")
                ):
                    st.session_state.current_file = None
                st.toast("模型设置已保存", icon="✅")
        with test_col:
            if st.button("测试连接", use_container_width=True):
                try:
                    reply = test_llm_connection(get_llm_settings_from_state())
                    st.success(f"连接成功：{reply or 'OK'}")
                except Exception as e:
                    st.error(f"连接失败：{e}")

        saved_label = provider_options.get(saved_settings["provider"], "自定义")
        saved_model = saved_settings["model"] or "未设置"
        st.caption(
            f"当前已保存：{saved_label} / {saved_model} / "
            f"{saved_settings['summary_max_chars']:,} 字符"
        )


def init_processing_settings_state():
    if st.session_state.get("processing_settings_initialized"):
        return
    settings = load_processing_settings()
    speaker_count = settings["speaker_count"]
    st.session_state.audio_vocal_separation_enabled = settings[
        "use_vocal_separation"
    ]
    st.session_state.audio_diarization_enabled = settings[
        "use_speaker_diarization"
    ]
    st.session_state.audio_speaker_count_mode = (
        "自动判断" if speaker_count == AUTO_SPEAKER_COUNT else "指定人数"
    )
    st.session_state.audio_primary_speaker_count = (
        2 if speaker_count == AUTO_SPEAKER_COUNT else speaker_count
    )
    st.session_state.processing_settings_initialized = True


def get_processing_settings_from_state():
    speaker_count = (
        AUTO_SPEAKER_COUNT
        if st.session_state.get("audio_speaker_count_mode") == "自动判断"
        else st.session_state.get("audio_primary_speaker_count", 2)
    )
    return {
        "use_vocal_separation": st.session_state.get(
            "audio_vocal_separation_enabled", False
        ),
        "use_speaker_diarization": st.session_state.get(
            "audio_diarization_enabled", False
        ),
        "speaker_count": speaker_count,
    }


def render_processing_settings_panel():
    init_processing_settings_state()
    saved_settings = load_processing_settings()

    with st.expander("音频识别设置", expanded=False):
        st.toggle(
            "AI 深度降噪",
            key="audio_vocal_separation_enabled",
            help="先分离人声再进行识别，适合环境嘈杂或背景音乐明显的视频；处理更慢。",
        )
        st.toggle(
            "多人对话识别",
            key="audio_diarization_enabled",
            help="识别主要说话人并在字幕中标记；会增加一段本地处理时间。",
        )
        st.segmented_control(
            "主要说话人数",
            ("指定人数", "自动判断"),
            key="audio_speaker_count_mode",
            disabled=not st.session_state.audio_diarization_enabled,
        )
        if st.session_state.audio_speaker_count_mode == "指定人数":
            st.number_input(
                "人数",
                min_value=2,
                max_value=20,
                step=1,
                key="audio_primary_speaker_count",
                disabled=not st.session_state.audio_diarization_enabled,
            )
        else:
            st.caption("自动判断适合人数未知的视频；已知人数时指定更稳定。")

        if st.button(
            "保存音频设置",
            key="save_processing_settings",
            use_container_width=True,
        ):
            saved_settings = save_processing_settings(
                get_processing_settings_from_state()
            )
            st.toast("音频识别设置已保存", icon="✅")

        if saved_settings["use_speaker_diarization"]:
            speaker_label = (
                "自动判断"
                if saved_settings["speaker_count"] == AUTO_SPEAKER_COUNT
                else f"主要 {saved_settings['speaker_count']} 人"
            )
            diarization_label = f"多人识别已开启 / {speaker_label}"
        else:
            diarization_label = "普通单轨字幕识别"
        denoise_label = (
            "深度降噪已开启"
            if saved_settings["use_vocal_separation"]
            else "深度降噪已关闭"
        )
        st.caption(f"当前已保存：{denoise_label} / {diarization_label}")


def build_current_card_source():
    current_file = st.session_state.get("current_file") or ""
    return CardSource(
        source_id=derive_source_id(current_file),
        title=derive_video_title(current_file),
        url=st.session_state.get("source_url") or "",
        srt_file=current_file.replace("\\", "/"),
        collection=st.session_state.get("current_collection") or "默认收藏夹",
        transcript=extract_pure_text_from_srt(st.session_state.get("base_srt_content") or ""),
        notes=st.session_state.get("current_notes") or "",
    )


def store_card_bundle(bundle):
    st.session_state.knowledge_card_bundle = bundle
    st.session_state.knowledge_card_bundle_source_id = bundle.source.source_id
    st.session_state.knowledge_card_editor_revision = (
        int(st.session_state.get("knowledge_card_editor_revision", 0)) + 1
    )


def load_current_card_bundle(source_id):
    bundle = st.session_state.get("knowledge_card_bundle")
    if isinstance(bundle, CardBundle) and bundle.source.source_id == source_id:
        return bundle

    bundle = load_draft(source_id)
    if bundle is None:
        source = build_current_card_source()
        bundle = load_committed_bundle(source)
    st.session_state.knowledge_card_bundle = bundle
    st.session_state.knowledge_card_bundle_source_id = source_id
    st.session_state.knowledge_card_editor_revision = 0
    return bundle


def parse_editor_list(value):
    items = []
    for item in re.split(r"[,，\n]", str(value or "")):
        cleaned = item.strip()
        if cleaned and cleaned not in items:
            items.append(cleaned)
    return items


def parse_editor_evidence(value):
    evidence = []
    timestamp_pattern = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")
    for raw_line in str(value or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        timestamp_match = timestamp_pattern.search(line)
        timestamp = f"({timestamp_match.group(1)})" if timestamp_match else ""
        if "|" in line or "｜" in line:
            parts = re.split(r"[|｜]", line, maxsplit=1)
            point = parts[1].strip()
        else:
            point = timestamp_pattern.sub("", line).strip(" ()[]：:-")
        if point:
            evidence.append({"timestamp": timestamp, "point": point})
    return evidence


def format_editor_evidence(evidence):
    return "\n".join(
        f"{item.get('timestamp', '')} | {item.get('point', '')}".strip()
        for item in evidence
        if item.get("point")
    )


def create_card_progress_callback(progress_bar, status_slot):
    stage_base = {
        "generating": 10,
        "extracting": 5,
        "merging": 72,
        "validating": 88,
        "revising": 20,
        "draft_saved": 100,
    }

    def update(stage, current, total, message):
        if stage == "extracting":
            percent = 5 + int(62 * current / max(1, total))
        else:
            percent = stage_base.get(stage, 5)
        progress_bar.progress(max(0, min(100, percent)))
        status_slot.caption(f"{message}，API 正在工作...")

    return update


def submit_card_generation_job(source, forced_type=None):
    response = requests.post(
        KNOWLEDGE_CARD_JOB_API_URL,
        json={
            "source": {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "srt_file": source.srt_file,
                "collection": source.collection,
                "transcript": source.transcript,
                "notes": source.notes,
            },
            "forced_type": forced_type,
        },
        timeout=15,
    )
    if response.status_code >= 400:
        if response.status_code == 404:
            raise RuntimeError("后台任务接口尚未加载，请关闭并重新启动一次软件")
        try:
            detail = response.json().get("detail") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(detail or f"后台服务返回 HTTP {response.status_code}")
    payload = response.json()
    return payload.get("job") or {}, payload.get("status") or "accepted"


def get_card_generation_job(source_id):
    try:
        response = requests.get(
            f"{KNOWLEDGE_CARD_JOB_API_URL}/{source_id}",
            timeout=2,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("job") or None
    except (requests.RequestException, ValueError):
        return None


def get_active_card_generation_jobs():
    try:
        response = requests.get(
            KNOWLEDGE_CARD_JOB_API_URL,
            params={"active_only": "true", "limit": 10},
            timeout=2,
        )
        response.raise_for_status()
        jobs = response.json().get("jobs") or []
        return jobs if isinstance(jobs, list) else []
    except (requests.RequestException, ValueError):
        return []


def get_saved_card_entries(source_id):
    try:
        entries = [
            entry
            for entry in load_index().get("cards", [])
            if entry.get("source_id") == source_id
        ]
    except KnowledgeCardError:
        return []
    return sorted(entries, key=lambda item: item.get("updated_at") or "", reverse=True)


def get_knowledge_card_tab_label(source_id):
    job_status = get_card_generation_job(source_id)
    if job_status and job_status.get("status") in {"queued", "running"}:
        return "🔵 知识卡片 · 生成中"
    if draft_path(source_id).exists():
        return "🟠 知识卡片 · 待保存"
    if get_saved_card_entries(source_id):
        return "🟢 知识卡片 · 已入库"
    if job_status and job_status.get("status") in {"failed", "interrupted"}:
        return "🔴 知识卡片 · 需重试"
    return MAIN_VIEW_KNOWLEDGE


def format_main_view_option(value, knowledge_card_label):
    if value == MAIN_VIEW_KNOWLEDGE:
        return knowledge_card_label
    return value


def render_saved_card_status(source_id):
    entries = get_saved_card_entries(source_id)
    has_draft = draft_path(source_id).exists()
    receipt = st.session_state.get("knowledge_card_save_receipt") or {}

    if entries:
        latest = entries[0].get("updated_at") or "时间未知"
        if receipt.get("source_id") == source_id and not has_draft:
            st.success(
                f"刚刚保存成功：{receipt.get('count', len(entries))} 张卡片已正式入库。"
            )
        elif has_draft:
            st.warning(
                f"已正式入库 {len(entries)} 张；当前还有未正式保存的草稿修改。"
            )
        else:
            st.success(f"已正式入库 {len(entries)} 张知识卡片 · 最近保存 {latest}")

        with st.expander("查看已保存文件", expanded=False):
            for entry in entries:
                st.markdown(
                    f"- **{entry.get('title') or '未命名卡片'}**  "
                    f"`知识卡片/{entry.get('path') or ''}`"
                )
    elif has_draft:
        st.info("草稿已生成并自动落盘，尚未正式保存到知识库。")
    else:
        st.caption("当前视频还没有正式保存的知识卡片。")


def render_card_job_status_fragment(source_id):
    @st.fragment(run_every="2s")
    def status_fragment():
        status = get_card_generation_job(source_id)
        if not status:
            return

        job_status = status.get("status")
        progress = max(0, min(100, int(status.get("progress") or 0)))
        message = status.get("message") or "知识卡片后台任务处理中"
        if job_status in {"queued", "running"}:
            st.info(message)
            st.progress(progress / 100.0)
            st.caption("任务已脱离当前页面运行，可以切换到 AI 对话或刷新页面。")
        elif job_status == "completed":
            st.success(message)
            loaded_job_id = st.session_state.get(f"knowledge_card_loaded_job_{source_id}")
            if loaded_job_id != status.get("job_id"):
                st.session_state[f"knowledge_card_ready_job_{source_id}"] = status.get("job_id")
                st.rerun()
        elif job_status == "failed":
            st.error(f"{message}：{status.get('error') or '未知错误'}")
        elif job_status == "interrupted":
            st.warning(message)

    status_fragment()


def run_card_revision(bundle, card_id, instruction, revise_all=False):
    active_settings = load_llm_settings()
    llm_client = create_llm_client(active_settings)
    status_slot = st.empty()
    progress_bar = st.progress(0)
    callback = create_card_progress_callback(progress_bar, status_slot)
    if revise_all:
        revised = revise_bundle(
            llm_client,
            active_settings,
            bundle,
            instruction,
            progress=callback,
        )
    else:
        revised = revise_card(
            llm_client,
            active_settings,
            bundle,
            card_id,
            instruction,
            progress=callback,
        )
    progress_bar.progress(100)
    status_slot.success("修改完成，草稿已自动保存。")
    return revised


def render_card_preview(card):
    st.markdown(f"### {card.title}")
    if card.summary:
        st.markdown(f"> {card.summary}")
    st.markdown(card.body_markdown or "暂无正文")
    if card.evidence:
        st.markdown("#### 视频依据")
        for item in card.evidence:
            st.markdown(f"- {item.get('point', '')} {item.get('timestamp', '')}".rstrip())
    st.markdown("#### AI 补充（未联网核验）")
    st.markdown(card.ai_supplement_markdown or "无")
    if card.caveats:
        st.markdown("#### 注意事项与适用边界")
        for caveat in card.caveats:
            st.markdown(f"- {caveat}")


def render_knowledge_card_workbench():
    source = build_current_card_source()
    job_status = get_card_generation_job(source.source_id)
    completed_job_id = ""
    if job_status and job_status.get("status") == "completed":
        completed_job_id = str(job_status.get("job_id") or "")
    ready_job_id = str(
        st.session_state.pop(f"knowledge_card_ready_job_{source.source_id}", "") or ""
    )
    result_job_id = ready_job_id or completed_job_id
    loaded_job_key = f"knowledge_card_loaded_job_{source.source_id}"
    if result_job_id and st.session_state.get(loaded_job_key) != result_job_id:
        try:
            fresh_bundle = load_draft(source.source_id)
            if fresh_bundle:
                store_card_bundle(fresh_bundle)
            st.session_state[loaded_job_key] = result_job_id
        except KnowledgeCardError as error:
            st.error(f"后台任务已完成，但草稿加载失败：{error}")

    try:
        bundle = load_current_card_bundle(source.source_id)
    except KnowledgeCardError as error:
        st.error(str(error))
        bundle = None

    st.caption(
        f"来源：{source.title} · 完整字幕 {len(source.transcript):,} 字符 · "
        "普通 AI 对话不会写入卡片来源"
    )
    render_saved_card_status(source.source_id)

    active_job = bool(
        job_status and job_status.get("status") in {"queued", "running"}
    )

    preferred_type_key = f"knowledge_card_type_{source.source_id}"
    type_options = [KNOWLEDGE_CARD_AUTO_TYPE, *CARD_TYPES]
    if st.session_state.get(preferred_type_key) not in type_options:
        st.session_state[preferred_type_key] = (
            bundle.detected_type if bundle and bundle.detected_type in CARD_TYPES else KNOWLEDGE_CARD_AUTO_TYPE
        )

    type_col, generate_col = st.columns([0.62, 0.38], gap="small", vertical_alignment="bottom")
    with type_col:
        preferred_type = st.selectbox(
            "生成类型",
            type_options,
            key=preferred_type_key,
            format_func=lambda value: KNOWLEDGE_CARD_TYPE_OPTION_LABELS.get(
                value,
                CARD_TYPE_LABELS.get(value, value),
            ),
        )
    with generate_col:
        generate_clicked = st.button(
            "后台重新分析完整视频" if bundle else "后台生成知识卡片",
            key=f"generate_cards_{source.source_id}",
            type="primary",
            use_container_width=True,
            disabled=active_job,
        )
    st.caption(KNOWLEDGE_CARD_TYPE_DESCRIPTIONS.get(preferred_type, ""))

    shortcut_requested = bool(st.session_state.pop("_start_knowledge_card_generation", False))
    if generate_clicked or shortcut_requested:
        if active_job:
            st.toast("该视频已经在后台生成知识卡片。", icon="ℹ️")
        else:
            forced_type = None if preferred_type == KNOWLEDGE_CARD_AUTO_TYPE else preferred_type
            try:
                submitted_job, submit_status = submit_card_generation_job(source, forced_type)
                st.session_state[f"knowledge_card_pending_job_{source.source_id}"] = (
                    submitted_job.get("job_id") or ""
                )
                receipt = st.session_state.get("knowledge_card_save_receipt") or {}
                if receipt.get("source_id") == source.source_id:
                    st.session_state.pop("knowledge_card_save_receipt", None)
                if submit_status == "already_running":
                    st.toast("该视频已有后台任务，已恢复现有进度。", icon="ℹ️")
                else:
                    st.toast("已加入知识卡片后台队列，可以离开当前界面。", icon="✅")
                st.rerun()
            except Exception as error:
                st.error(f"后台任务提交失败：{error}")

    if job_status:
        render_card_job_status_fragment(source.source_id)

    if active_job:
        return

    if not bundle or not bundle.cards:
        st.info("请选择类型并生成草稿。生成结果会自动落盘，下次打开可继续编辑。")
        return

    bundle.source = source
    revision = int(st.session_state.get("knowledge_card_editor_revision", 0))
    st.caption(
        f"识别类型：{CARD_TYPE_LABELS.get(bundle.detected_type, bundle.detected_type)} · "
        f"共 {len(bundle.cards)} 张 · 草稿自动恢复"
    )

    active_key = f"knowledge_card_active_{source.source_id}"
    card_ids = [card.card_id for card in bundle.cards]
    if st.session_state.get(active_key) not in card_ids:
        st.session_state[active_key] = card_ids[0]

    selection_changed = False
    with st.container(height=180, border=True):
        for card in bundle.cards:
            keep_col, open_col = st.columns([0.12, 0.88], gap="small", vertical_alignment="center")
            selected_key = f"knowledge_card_keep_{source.source_id}_{card.card_id}_{revision}"
            with keep_col:
                selected = st.checkbox(
                    "保留",
                    value=card.selected,
                    key=selected_key,
                    label_visibility="collapsed",
                    help="勾选后会在正式保存时写入知识库",
                )
            if selected != card.selected:
                card.selected = selected
                selection_changed = True
            with open_col:
                if st.button(
                    card.title,
                    key=f"open_knowledge_card_{source.source_id}_{card.card_id}",
                    type="primary" if st.session_state[active_key] == card.card_id else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[active_key] = card.card_id
                    st.rerun()

    if selection_changed:
        save_draft(bundle)

    active_card = next(card for card in bundle.cards if card.card_id == st.session_state[active_key])
    action_col, relation_col = st.columns([0.3, 0.7], gap="small", vertical_alignment="center")
    with action_col:
        delete_card = st.button(
            "删除此草稿卡",
            key=f"delete_knowledge_card_{source.source_id}_{active_card.card_id}",
            use_container_width=True,
        )
    with relation_col:
        relationship = []
        if active_card.parent_card_id:
            relationship.append("子卡")
        if active_card.child_card_ids:
            relationship.append(f"含 {len(active_card.child_card_ids)} 张子卡")
        st.caption(" · ".join(relationship) or "独立卡片")

    if delete_card:
        bundle.cards = [card for card in bundle.cards if card.card_id != active_card.card_id]
        save_draft(bundle)
        store_card_bundle(bundle)
        st.toast("草稿卡已删除。", icon="🗑️")
        st.rerun()

    editor_key = f"knowledge_card_editor_{source.source_id}_{active_card.card_id}_{revision}"
    with st.form(editor_key, clear_on_submit=False):
        card_type = st.selectbox(
            "卡片类型",
            CARD_TYPES,
            index=(
                CARD_TYPES.index(active_card.card_type)
                if active_card.card_type in CARD_TYPES
                else CARD_TYPES.index("general_knowledge")
            ),
            format_func=lambda value: CARD_TYPE_LABELS.get(value, value),
        )
        title = st.text_input("标题", value=active_card.title)
        summary = st.text_area("摘要", value=active_card.summary, height=80)
        tags = st.text_input("标签（逗号或换行分隔）", value="，".join(active_card.tags))
        aliases = st.text_input("别名（逗号或换行分隔）", value="，".join(active_card.aliases))
        retrieval_questions = st.text_area(
            "自然语言检索问题（每行一个）",
            value="\n".join(active_card.retrieval_questions),
            height=100,
        )
        body_markdown = st.text_area("卡片正文（Markdown）", value=active_card.body_markdown, height=260)
        evidence_text = st.text_area(
            "时间戳证据（每行：时间戳 | 内容）",
            value=format_editor_evidence(active_card.evidence),
            height=120,
        )
        ai_supplement = st.text_area(
            "AI 补充（未联网核验）",
            value=active_card.ai_supplement_markdown,
            height=120,
        )
        caveats = st.text_area(
            "注意事项与适用边界（每行一个）",
            value="\n".join(active_card.caveats),
            height=100,
        )
        apply_manual_edit = st.form_submit_button(
            "应用到草稿",
            type="primary",
            use_container_width=True,
        )

    if apply_manual_edit:
        if not title.strip() or not body_markdown.strip():
            st.warning("标题和卡片正文不能为空。")
        else:
            active_card.card_type = card_type
            active_card.title = title.strip()
            active_card.summary = summary.strip()
            active_card.tags = parse_editor_list(tags)
            active_card.aliases = parse_editor_list(aliases)
            active_card.retrieval_questions = parse_editor_list(retrieval_questions)
            active_card.body_markdown = body_markdown.strip()
            active_card.evidence = parse_editor_evidence(evidence_text)
            active_card.ai_supplement_markdown = ai_supplement.strip()
            active_card.caveats = parse_editor_list(caveats)
            active_card.updated_at = now_china_iso()
            bundle.source = source
            save_draft(bundle)
            store_card_bundle(bundle)
            st.toast("人工修改已应用并保存到草稿。", icon="💾")
            st.rerun()

    with st.expander("Markdown 预览", expanded=False):
        render_card_preview(active_card)

    with st.form(
        f"knowledge_card_ai_revision_{source.source_id}_{active_card.card_id}_{revision}",
        clear_on_submit=True,
    ):
        revision_instruction = st.text_area(
            "AI 修改要求",
            placeholder="例如：把行动步骤改得更具体，并保留现有时间戳证据。",
            height=90,
        )
        current_col, all_col = st.columns(2, gap="small")
        with current_col:
            revise_current = st.form_submit_button("修改当前卡片", use_container_width=True)
        with all_col:
            revise_all = st.form_submit_button("修改整组卡片", use_container_width=True)

    if revise_current or revise_all:
        if not revision_instruction.strip():
            st.warning("请先填写 AI 修改要求。")
        else:
            try:
                bundle.source = source
                revised = run_card_revision(
                    bundle,
                    active_card.card_id,
                    revision_instruction.strip(),
                    revise_all=revise_all,
                )
                store_card_bundle(revised)
                st.toast("AI 修改已写入草稿。", icon="✅")
                st.rerun()
            except Exception as error:
                st.error(f"AI 修改失败，原草稿未被覆盖：{error}")

    selected_ids = [card.card_id for card in bundle.cards if card.selected]
    if st.button(
        f"正式保存选中的 {len(selected_ids)} 张卡片",
        key=f"commit_knowledge_cards_{source.source_id}",
        type="primary",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            bundle.source = source
            saved_paths = commit_bundle(bundle, selected_ids)
            st.session_state.knowledge_card_save_receipt = {
                "source_id": source.source_id,
                "count": len(saved_paths),
                "saved_at": now_china_iso(),
            }
            st.toast(f"已正式保存 {len(saved_paths)} 张知识卡片。", icon="✅")
            st.rerun()
        except KnowledgeCardError as error:
            st.error(str(error))


# ================= Streamlit 界面构建 =================
st.set_page_config(page_title="AI视频总结助手", page_icon="🎬", layout="wide")

# [编程大师2.0 新增]：注入全局 CSS 引擎，彻底杀死标题自动锚点，实现无痕纯净复制
st.markdown("""
    <style>
    html, body {
        height: 100% !important;
        overflow: hidden !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .stMain {
        height: 100dvh !important;
        overflow: hidden !important;
    }
    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 0.75rem !important;
        height: 100dvh !important;
        max-height: 100dvh !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    .app-mini-title {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
        position: fixed;
        top: 0.9rem;
        left: 1.75rem;
        z-index: 1000001;
        margin: 0;
        pointer-events: none;
    }
    /* 侧栏默认只露出触发边，鼠标移入或控件获得焦点时覆盖展开。 */
    section[data-testid="stSidebar"] {
        width: 20rem !important;
        min-width: 20rem !important;
        margin-right: calc(-20rem + 18px) !important;
        transform: translateX(calc(-100% + 18px));
        transition: transform 180ms ease, box-shadow 180ms ease;
        z-index: 1000002 !important;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: 20rem !important;
    }
    section[data-testid="stSidebar"]:hover,
    section[data-testid="stSidebar"]:focus-within {
        transform: translateX(0);
        box-shadow: 8px 0 24px rgba(0, 0, 0, 0.14);
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    /* 快捷指令使用紧凑的 3x3 网格，减少侧栏纵向占用。 */
    section[data-testid="stSidebar"] [class*="st-key-quick_preset_"] button {
        min-height: 2rem !important;
        height: 2rem !important;
        padding: 0.1rem 0.2rem !important;
        font-size: 0.76rem !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has([class*="st-key-quick_preset_"]) {
        gap: 0.35rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(> [class*="st-key-quick_preset_"]) {
        gap: 0.35rem !important;
    }
    /* 隐藏所有 Markdown 标题自动生成的锚点图标 */
    a.header-anchor {
        display: none !important;
    }
    /* 暴力防线：直接屏蔽所有标题里的 a 标签，防止它们被鼠标圈中复制 */
    .stMarkdown h1 a, 
    .stMarkdown h2 a, 
    .stMarkdown h3 a, 
    .stMarkdown h4 a, 
    .stMarkdown h5 a, 
    .stMarkdown h6 a {
        display: none !important;
    }
    div[data-testid="column"]:has(#ai-media-pane-marker),
    div[data-testid="stColumn"]:has(#ai-media-pane-marker),
    .stColumn:has(#ai-media-pane-marker) {
        position: sticky !important;
        top: 72px !important;
        align-self: flex-start !important;
        z-index: 5 !important;
        height: var(--ai-pane-height, calc(100dvh - 160px)) !important;
        max-height: var(--ai-pane-height, calc(100dvh - 160px)) !important;
        overflow: hidden !important;
        padding-right: 0.35rem !important;
    }
    div[data-testid="column"]:has(#ai-summary-pane-marker),
    div[data-testid="stColumn"]:has(#ai-summary-pane-marker),
    .stColumn:has(#ai-summary-pane-marker) {
        height: var(--ai-pane-height, calc(100dvh - 160px)) !important;
        max-height: var(--ai-pane-height, calc(100dvh - 160px)) !important;
        overflow-y: auto !important;
        overscroll-behavior: contain !important;
        padding-right: 0.75rem !important;
        scrollbar-gutter: stable !important;
    }
    .subtitle-diff-list {
        max-height: 19rem;
        overflow-y: auto;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 6px;
        margin: 0.35rem 0 0.75rem 0;
    }
    .subtitle-diff-row {
        padding: 0.65rem 0.75rem;
        border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    }
    .subtitle-diff-row:last-child {
        border-bottom: 0;
    }
    .subtitle-diff-time {
        color: #52606d;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .subtitle-diff-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 0.55rem;
    }
    .subtitle-diff-label {
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .before-label { color: #a63d40; }
    .after-label { color: #27734a; }
    .subtitle-diff-text {
        border-left: 3px solid;
        border-radius: 4px;
        padding: 0.45rem 0.55rem;
        font-size: 0.86rem;
        line-height: 1.5;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .before-text {
        border-left-color: #d96b70;
        background: rgba(217, 107, 112, 0.10);
    }
    .after-text {
        border-left-color: #43a66f;
        background: rgba(67, 166, 111, 0.11);
    }
    @media (max-width: 800px) {
        .subtitle-diff-grid {
            grid-template-columns: minmax(0, 1fr);
        }
    }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="app-mini-title">🎬 AI视频总结助手</div>', unsafe_allow_html=True)

if "current_file" not in st.session_state: st.session_state.current_file = None
if "messages" not in st.session_state: st.session_state.messages = []
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "base_srt_content" not in st.session_state: st.session_state.base_srt_content = ""
# 【核心修复】：必须初始化这个状态
if "source_url" not in st.session_state: st.session_state.source_url = ""
if "current_collection" not in st.session_state: st.session_state.current_collection = "默认收藏夹"
if "current_notes" not in st.session_state: st.session_state.current_notes = ""
if "subtitle_correction" not in st.session_state: st.session_state.subtitle_correction = {}
if "video_metadata" not in st.session_state: st.session_state.video_metadata = {}
if st.session_state.get("qa_mode") not in QA_MODE_OPTIONS:
    st.session_state.qa_mode = QA_MODE_VIDEO_FIRST
if st.session_state.get("main_view") not in MAIN_VIEW_OPTIONS:
    st.session_state.main_view = MAIN_VIEW_CHAT
if st.session_state.pop("_clear_qa_input", False):
    st.session_state.qa_input = ""
pending_card_path_warning = st.session_state.pop("_knowledge_card_path_warning", None)
if pending_card_path_warning:
    st.warning(pending_card_path_warning)

# 跨收藏夹移动发生在控件渲染之后，延迟到下一次重跑前清空侧栏文件选择。
pending_library_sidebar_hold = st.session_state.pop("_pending_library_sidebar_hold", None)
if pending_library_sidebar_hold:
    st.session_state.library_file_selection = None
    st.session_state["_held_library_collection"] = pending_library_sidebar_hold["collection"]

preset_clicked = None

# --- 左侧边栏 (极简无拷贝版) ---
with st.sidebar:
    st.markdown("**📡 任务看板**")
    @st.fragment(run_every="3s")
    def auto_refresh_task_board():
        tasks = get_tasks_from_db()
        card_jobs = get_active_card_generation_jobs()
        if not tasks and not card_jobs:
            st.caption("💤 当前无排队任务")
        if tasks:
            active_tasks = [t for t in tasks if t['status'] in ['downloading', 'transcribing', 'correcting', 'summarizing']]
            active_task = active_tasks[0] if active_tasks else None
            if active_task:
                status_display = {"downloading": "🌐 下载中", "transcribing": "🎙️ 转录中", "correcting": "📝 校对中", "summarizing": "🤖 总结中"}.get(active_task['status'], "⚡ 处理中")
                st.markdown(f"{status_display} `{active_task['title'][:12]}...` **{active_task['progress']}%**")
                st.progress(active_task['progress'] / 100.0)
                for extra_task in active_tasks[1:]:
                    progress = max(0, min(100, int(extra_task.get('progress') or 0)))
                    status_label = {"downloading": "下载", "transcribing": "转录", "correcting": "校对", "summarizing": "总结"}.get(extra_task['status'], "处理中")
                    st.markdown(f"**{status_label}** `{extra_task['title'][:12]}...` **{progress}%**")
                    st.progress(progress / 100.0)
            else:
                waiting_llm = [t for t in tasks if t['status'] == 'awaiting_llm']
                if waiting_llm:
                    st.caption(f"☕ {len(waiting_llm)} 个视频已识别，排队校对/总结中...")
                else:
                    st.caption("💤 GPU 待命中")

            with st.expander("📊 历史记录", expanded=False):
                for t in tasks:
                    emoji = {"pending": "⏳", "downloading": "🌐", "transcribing": "🎙️", "awaiting_llm": "☕", "correcting": "📝", "summarizing": "🤖", "completed": "✅", "error": "❌"}.get(t['status'], "❓")
                    st.caption(f"{emoji} {t['title'][:15]}... ({t['progress']}%)")

        for job in card_jobs:
            progress = max(0, min(100, int(job.get("progress") or 0)))
            title = str(job.get("source_title") or "未命名视频")
            status_label = "排队" if job.get("status") == "queued" else "生成卡片"
            st.markdown(f"**{status_label}** `{title[:12]}...` **{progress}%**")
            st.progress(progress / 100.0)
                
    auto_refresh_task_board()
    st.divider()

    st.markdown("**💡 快捷指令**")
    preset_columns = st.columns(3, gap="small")
    for index, (label, prompt_text) in enumerate(PRESETS.items()):
        with preset_columns[index % 3]:
            if st.button(
                label,
                key=f"quick_preset_{index}",
                use_container_width=True,
                disabled=(
                    prompt_text is None
                    or (
                        prompt_text == KNOWLEDGE_CARD_PRESET
                        and not st.session_state.current_file
                    )
                ),
            ):
                if prompt_text == KNOWLEDGE_CARD_PRESET:
                    st.session_state.main_view = MAIN_VIEW_KNOWLEDGE
                    st.session_state["_start_knowledge_card_generation"] = True
                else:
                    st.session_state.main_view = MAIN_VIEW_CHAT
                    preset_clicked = prompt_text
    st.divider()

    st.markdown("**📁 我的字幕库**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 刷新", use_container_width=True): pass
    with col2:
        if st.button("🗑️ 删除", use_container_width=True, type="primary"):
            if st.session_state.current_file:
                target_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
                if os.path.exists(target_path):
                    try: 
                        os.remove(target_path)
                        
                        # [编程大师2.0 新增]：资产伴生销毁，连同本地视频原片一起删除！
                        video_path = target_path.rsplit('.', 1)[0] + '.mp4'
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            
                        # GC 垃圾回收，如果删除后该文件夹空了，顺手销毁它
                        parent_dir = os.path.dirname(target_path)
                        if parent_dir != SRT_VAULT_DIR and os.path.isdir(parent_dir):
                            if not os.listdir(parent_dir): 
                                os.rmdir(parent_dir)
                    except: pass
                
                st.session_state.current_file = None
                st.session_state.messages = []
                st.session_state.transcript = ""
                st.session_state.source_url = ""
                st.session_state.current_collection = "默认收藏夹"
                st.session_state.current_notes = ""
                st.toast("✅ 资产已彻底销毁 (字幕+视频原片)！", icon="🗑️")
                st.rerun()

    # [编程大师2.0 升级版]：侧边栏收藏夹视图与路由
    library_data = get_library_data()
    all_collections = list(library_data.keys())
    
    # 防御性排序：确保“默认收藏夹”永远在下拉菜单的第一位
    if "默认收藏夹" in all_collections:
        all_collections.remove("默认收藏夹")
    all_collections.insert(0, "默认收藏夹") 

    # 使用稳定 key，任务完成导致选项增加时仍保持当前收藏夹。
    collection_state_key = "library_collection_view"
    selected_collection_state = st.session_state.get(collection_state_key)
    if selected_collection_state not in all_collections:
        preferred_collection = st.session_state.current_collection
        if preferred_collection not in all_collections:
            preferred_collection = "默认收藏夹"
        st.session_state[collection_state_key] = preferred_collection

    selected_collection = st.selectbox(
        "📂 选择收藏夹视图",
        all_collections,
        index=None,
        key=collection_state_key,
        label_visibility="collapsed",
    )

    held_library_collection = st.session_state.get("_held_library_collection")
    if held_library_collection and held_library_collection != selected_collection:
        st.session_state.pop("_held_library_collection", None)
        held_library_collection = None
    
    # 动态过滤出当前选中的收藏夹下的文件列表
    filtered_files = library_data.get(selected_collection, [])
    
    if not filtered_files:
        st.caption("📭 当前收藏夹空空如也！")
        selected_file = None
    else:
        file_state_key = "library_file_selection"
        selected_file_state = st.session_state.get(file_state_key)
        if selected_file_state not in filtered_files:
            preserve_empty_selection = (
                held_library_collection == selected_collection
                and selected_file_state is None
            )
            st.session_state[file_state_key] = resolve_library_file_selection(
                selected_file_state,
                filtered_files,
                st.session_state.current_file,
                preserve_empty_selection,
            )

        with st.container(height=400, border=True):
            # [视觉优化]：后台使用相对路径定位，前台只显示干净的文件名
            file_labels = build_library_file_labels(filtered_files)
            selected_file = st.radio(
                "选择视频",
                filtered_files,
                index=None,
                key=file_state_key,
                format_func=lambda file_path: file_labels.get(
                    file_path,
                    clean_library_file_label(file_path),
                ),
                label_visibility="collapsed",
            )
            if selected_file and held_library_collection == selected_collection:
                st.session_state.pop("_held_library_collection", None)
    st.divider()

    with st.popover("📂 导入本地音视频 (点击展开)", use_container_width=True):
        st.caption("无需复制！直接粘贴本地文件路径：")
        local_path = st.text_input(" ", placeholder="如: D:\\Videos\\test.mp4", label_visibility="collapsed")

        auto_download = st.checkbox("📥 添加任务时自动下载原片", value=AUTO_DOWNLOAD_VIDEO)

        if local_path and st.button("🚀 开始处理", use_container_width=True, type="primary"):
            clean_path = local_path.strip().strip('"').strip("'")
            if os.path.exists(clean_path):
                try:
                    res = requests.post(TASK_API_URL, json={
                        "source_type": "local_file",
                        "source_path": clean_path,
                        "title": os.path.splitext(os.path.basename(clean_path))[0],
                        "options": {"auto_download": auto_download}
                    })
                    if res.status_code == 200: st.toast("✅ 已加入队列！", icon="🚀")
                    else: st.error("加入队列失败")
                except Exception as e:
                    st.error(f"无法连接调度中心: {e}")
            else:
                st.error("❌ 找不到该文件，请检查路径。")
    st.divider()

    render_processing_settings_panel()

    render_llm_settings_panel()

# --- 上下文切换与兼容读取逻辑 ---
if selected_file and selected_file != st.session_state.current_file:
    st.session_state.current_file = selected_file
    st.session_state.source_url = "" # 先清空旧的链接
    st.session_state.current_collection = "默认收藏夹" # 清空旧收藏状态
    st.session_state.current_notes = "" # 清空旧笔记
    st.session_state.subtitle_correction = {}
    st.session_state.video_metadata = {}
    file_path = os.path.join(SRT_VAULT_DIR, selected_file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
            
        parts = full_content.split(DELIMITER)
        raw_srt = parts[0]
        st.session_state.base_srt_content = raw_srt 
        
        saved_history = []
        if len(parts) > 1:
            try:
                data = json.loads(parts[1])
                # [编程大师2.0 升级版]：核心向下兼容读取逻辑
                if isinstance(data, list):
                    saved_history = data
                    # 如果是远古版本的纯列表，赋予安全默认值
                    st.session_state.source_url = ""
                    st.session_state.current_collection = "默认收藏夹"
                    st.session_state.current_notes = ""
                    st.session_state.subtitle_correction = {}
                    st.session_state.video_metadata = {}
                elif isinstance(data, dict):
                    saved_history = data.get("history", [])
                    st.session_state.source_url = data.get("source_url", "")
                    st.session_state.current_notes = data.get("notes", "")
                    st.session_state.subtitle_correction = data.get("subtitle_correction") or {}
                    st.session_state.video_metadata = data.get("video_metadata") or {}
                    
                    # [编程大师2.0 重大转折]：以物理文件夹结构为单一真理 (Single Source of Truth)
                    rel_dir = os.path.dirname(selected_file)
                    if not rel_dir:
                        st.session_state.current_collection = "默认收藏夹"
                    else:
                        st.session_state.current_collection = rel_dir.split(os.sep)[0]
                st.toast("📚 已自动恢复数据档案与笔记", icon="🕰️")
            except Exception as e:
                pass
            
        pure_text = extract_pure_text_from_srt(raw_srt)
        transcript_limit = limit_transcript_to_complete_segments(
            pure_text,
            resolve_summary_char_limit(load_llm_settings()),
        )
        pure_text = transcript_limit.text
        is_truncated = transcript_limit.truncated
            
        st.session_state.transcript = pure_text
        sys_prompt = build_qa_system_prompt(pure_text, st.session_state.qa_mode)
        
        st.session_state.messages = [{"role": "system", "content": sys_prompt}] + saved_history
        if is_truncated:
            cutoff_label = transcript_limit.last_timestamp or "最近完整段落"
            st.toast(
                f"字幕已按 {transcript_limit.limit:,} 字符上限保留到 {cutoff_label}。",
                icon="✂️",
            )
    except Exception as e:
        st.error(f"加载失败: {e}")

# --- 主对话区 ---
if st.session_state.transcript and st.session_state.current_file:

# 保留模型输出的原始 Markdown 结构，仅转换可点击时间戳。
    def format_chat_content(content, url, is_local=False):
        if not url or not url.startswith("http"):
            return content
        
        pattern = r'[`\(\[\{]*\b((\d{1,2}:)?\d{1,2}:\d{2})(?:\s*[-~到至]\s*((?:\d{1,2}:)?\d{1,2}:\d{2}))?\b[`\)\]\}]*'
        
        def replacer(match):
            start_time = match.group(1) 
            end_time = match.group(3)
            parts = start_time.split(':')
            try:
                if len(parts) == 2: sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3: sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else: return match.group(0) 
                
                display_time = f"{start_time}-{end_time}" if end_time else start_time
                
                # 【核心黑科技】：如果本地视频存在，将链接变异为 JS 可以拦截的特定锚点！
                if is_local:
                    target_url = f"#local_seek={sec}"
                else:
                    joiner = "&" if "?" in url else "?"
                    target_url = f"{url}{joiner}t={sec}&ai_seek={sec}"
                
                return f" [({display_time})]({target_url}) "
            except Exception:
                return match.group(0)
                
        return re.sub(pattern, replacer, content)
        
# 【编程大师2.0 终极版】：智能资产状态感知与内嵌播放引擎
    current_srt_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
    current_video_path = current_srt_path.rsplit('.', 1)[0] + '.mp4'
    video_exists = os.path.exists(current_video_path)

    media_col, summary_col = st.columns([0.42, 0.58], gap="medium")

# [编程大师2.0 终极杀招]：内嵌播放器 + 底层 JS 跨维度遥控
    with media_col:
        st.subheader(f"💬 {st.session_state.current_file.rsplit('_', 1)[0]}")
        metadata_caption = build_video_metadata_caption(st.session_state.video_metadata)
        if metadata_caption:
            st.caption(metadata_caption)
        if video_exists:
            if "player_subtitles_enabled" not in st.session_state:
                st.session_state.player_subtitles_enabled = True
            if st.session_state.get("player_subtitle_position") not in {
                "顶部",
                "中部",
                "底部",
            }:
                st.session_state.player_subtitle_position = "底部"
            original_srt_content = ""
            raw_backup_path = current_srt_path + ".raw.bak"
            if os.path.exists(raw_backup_path):
                try:
                    with open(raw_backup_path, "r", encoding="utf-8") as file:
                        original_srt_content = file.read()
                except OSError:
                    original_srt_content = ""
            subtitle_tracks = build_video_subtitle_tracks(
                st.session_state.base_srt_content,
                original_srt_content,
                correction_applied=(
                    st.session_state.subtitle_correction.get("status") == "completed"
                ),
                position=st.session_state.player_subtitle_position,
            )
            visible_subtitle_tracks = (
                subtitle_tracks
                if st.session_state.player_subtitles_enabled
                else {}
            )
            player_fingerprint = build_video_player_fingerprint(
                current_video_path,
                visible_subtitle_tracks,
                st.session_state.player_subtitle_position,
            )
            st.markdown(
                '<span id="ai-media-pane-marker" '
                f'data-ai-media-fingerprint="{player_fingerprint}"></span>',
                unsafe_allow_html=True,
            )
            st.video(
                current_video_path,
                subtitles=visible_subtitle_tracks or None,
            )
            subtitle_toggle_col, subtitle_position_col = st.columns(
                [0.36, 0.64],
                gap="small",
                vertical_alignment="center",
            )
            with subtitle_toggle_col:
                st.toggle(
                    "显示字幕",
                    key="player_subtitles_enabled",
                )
            with subtitle_position_col:
                st.segmented_control(
                    "字幕位置",
                    ("顶部", "中部", "底部"),
                    key="player_subtitle_position",
                    disabled=not st.session_state.player_subtitles_enabled,
                    label_visibility="collapsed",
                )
        else:
            st.markdown(
                '<span id="ai-media-pane-marker"></span>',
                unsafe_allow_html=True,
            )
            st.info("当前还没有本地视频，可点击下方按钮下载原片后同屏观看。")

        if st.session_state.source_url or video_exists:
            action_columns = st.columns(2 if st.session_state.source_url else 1, gap="small")
            asset_action_index = 0

            if st.session_state.source_url:
                with action_columns[0]:
                    st.link_button("🔗 观看原网页", st.session_state.source_url, use_container_width=True)
                asset_action_index = 1

            with action_columns[asset_action_index]:
                if video_exists:
                    if st.button("🗑️ 删除本地原片", use_container_width=True, type="secondary"):
                        try:
                            os.remove(current_video_path)
                            st.toast("✅ 本地原片已删除，释放硬盘空间！", icon="🗑️")
                            st.rerun()
                        except Exception:
                            pass
                elif st.session_state.source_url:
                    if st.button(f"📥 下载 {DOWNLOAD_MAX_HEIGHT}P 原片", use_container_width=True, type="secondary"):
                        with st.spinner("🚀 正在拉取高清音视频轨并进行无损合并，请耐心稍候..."):
                            try:
                                save_dir = os.path.dirname(current_srt_path)
                                base_name = os.path.basename(current_srt_path).rsplit('.', 1)[0]
                                download_high_res_video(st.session_state.source_url, save_dir, base_name)
                                st.toast("✅ 资产已归档入库！", icon="🎉")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ 下载失败: {e}")

        # 按分栏的真实顶部位置计算可用高度，避免左右栏撑高整个页面。
        components.html("""
        <script>
        const parentDoc = window.parent.document;
        const parentWindow = window.parent;

        function findPane(markerId) {
            const marker = parentDoc.getElementById(markerId);
            return marker ? marker.closest(
                'div[data-testid="column"], div[data-testid="stColumn"], .stColumn'
            ) : null;
        }

        function getFullscreenElement() {
            return parentDoc.fullscreenElement
                || parentDoc.webkitFullscreenElement
                || null;
        }

        function fullscreenContainsVideo(fullscreenElement, vid) {
            if (!fullscreenElement || !vid) return false;
            return fullscreenElement === vid
                || (
                    typeof fullscreenElement.contains === 'function'
                    && fullscreenElement.contains(vid)
                )
                || (
                    typeof vid.contains === 'function'
                    && vid.contains(fullscreenElement)
                );
        }

        function setStyleIfChanged(element, property, value) {
            if (element.style.getPropertyValue(property) !== value) {
                element.style.setProperty(property, value);
            }
        }

        function removeStyleIfPresent(element, property) {
            if (element.style.getPropertyValue(property)) {
                element.style.removeProperty(property);
            }
        }

        function syncPaneHeights() {
            if (getFullscreenElement()) return;

            ['ai-media-pane-marker', 'ai-summary-pane-marker'].forEach(markerId => {
                const pane = findPane(markerId);
                if (!pane) return;

                const paneTop = Math.max(0, pane.getBoundingClientRect().top);
                const availableHeight = Math.max(
                    1,
                    Math.floor(parentWindow.innerHeight - paneTop - 12)
                );
                pane.style.setProperty('--ai-pane-height', `${availableHeight}px`);
            });

        }

        function optimizeVideos() {
            const fullscreenElement = getFullscreenElement();
            const mediaPane = findPane('ai-media-pane-marker');
            const paneHeight = mediaPane
                ? mediaPane.getBoundingClientRect().height
                : parentWindow.innerHeight;
            const maxVideoHeight = Math.max(
                140,
                Math.min(parentWindow.innerHeight * 0.46, paneHeight - 390)
            );

            parentDoc.querySelectorAll('video').forEach(vid => {
                if (fullscreenContainsVideo(fullscreenElement, vid)) {
                    removeStyleIfPresent(vid, 'max-height');
                    removeStyleIfPresent(vid, 'border-radius');
                    setStyleIfChanged(vid, 'object-fit', 'contain');
                    return;
                }
                setStyleIfChanged(vid, 'max-height', `${maxVideoHeight}px`);
                setStyleIfChanged(vid, 'object-fit', 'contain');
                setStyleIfChanged(vid, 'border-radius', '8px');
            });
        }

        function refreshVideoLayout() {
            if (getFullscreenElement()) return;
            syncVideoMedia();
            syncPaneHeights();
            optimizeVideos();
        }

        function handleVideoFullscreenChange() {
            const fullscreenElement = getFullscreenElement();
            if (fullscreenElement) {
                parentDoc.querySelectorAll('video').forEach(vid => {
                    if (!fullscreenContainsVideo(fullscreenElement, vid)) return;
                    removeStyleIfPresent(vid, 'max-height');
                    removeStyleIfPresent(vid, 'border-radius');
                    setStyleIfChanged(vid, 'object-fit', 'contain');
                });
                return;
            }
            refreshVideoLayout();
        }

        if (parentWindow.__aiVideoLayoutInterval) {
            parentWindow.clearInterval(parentWindow.__aiVideoLayoutInterval);
        }
        parentWindow.__aiVideoLayoutInterval = parentWindow.setInterval(
            refreshVideoLayout,
            500
        );

        if (parentWindow.__aiVideoFullscreenHandler) {
            parentDoc.removeEventListener(
                'fullscreenchange',
                parentWindow.__aiVideoFullscreenHandler
            );
            parentDoc.removeEventListener(
                'webkitfullscreenchange',
                parentWindow.__aiVideoFullscreenHandler
            );
        }
        parentWindow.__aiVideoFullscreenHandler = handleVideoFullscreenChange;
        parentDoc.addEventListener(
            'fullscreenchange',
            parentWindow.__aiVideoFullscreenHandler
        );
        parentDoc.addEventListener(
            'webkitfullscreenchange',
            parentWindow.__aiVideoFullscreenHandler
        );

        refreshVideoLayout();

        function findLocalVideo() {
            const mediaPane = findPane('ai-media-pane-marker');
            return mediaPane
                ? mediaPane.querySelector('video')
                : parentDoc.querySelector('video');
        }

        function getVideoFingerprint() {
            const marker = parentDoc.getElementById('ai-media-pane-marker');
            return marker ? marker.dataset.aiMediaFingerprint || '' : '';
        }

        function getVideoMediaSignature(vid) {
            if (!vid) return '';
            const sourceUrls = Array.from(vid.querySelectorAll('source'))
                .map(source => source.getAttribute('src') || '')
                .join('|');
            const trackUrls = Array.from(vid.querySelectorAll('track'))
                .map(track => `${track.label || ''}:${track.getAttribute('src') || ''}`)
                .join('|');
            return [vid.getAttribute('src') || '', sourceUrls, trackUrls].join('||');
        }

        function restoreSubtitleTrack(vid, preferredLabel) {
            if (!vid || !vid.textTracks) return;
            const tracks = Array.from(vid.textTracks);
            if (!tracks.length) return;
            const matchingTrack = tracks.find(track => track.label === preferredLabel);
            tracks.forEach((track, index) => {
                track.mode = matchingTrack
                    ? (track === matchingTrack ? 'showing' : 'disabled')
                    : (index === 0 ? 'showing' : 'disabled');
            });
        }

        function syncVideoMedia() {
            const vid = findLocalVideo();
            const fingerprint = getVideoFingerprint();
            if (!vid || !fingerprint) return;

            const mediaSignature = getVideoMediaSignature(vid);
            const sameFingerprint = vid.dataset.aiMediaFingerprint === fingerprint;
            const sameSignature = vid.dataset.aiMediaSignature === mediaSignature;
            if (sameFingerprint && sameSignature) {
                const activeTrack = Array.from(vid.textTracks || [])
                    .find(track => track.mode === 'showing');
                if (activeTrack) {
                    parentWindow.__aiSubtitleTrackLabel = activeTrack.label;
                }
                return;
            }

            const preferredLabel = parentWindow.__aiSubtitleTrackLabel || '';
            vid.dataset.aiMediaFingerprint = fingerprint;
            vid.dataset.aiMediaSignature = mediaSignature;
            try {
                vid.load();
            } catch (error) {
                return;
            }
            [0, 150, 500].forEach(delay => {
                parentWindow.setTimeout(
                    () => restoreSubtitleTrack(vid, preferredLabel),
                    delay
                );
            });
        }

        function seekLocalVideo(sec, attempt = 0) {
            const vid = findLocalVideo();
            if (!vid) {
                if (attempt < 20) {
                    parentWindow.setTimeout(() => seekLocalVideo(sec, attempt + 1), 100);
                }
                return;
            }

            const applySeek = () => {
                try {
                    vid.currentTime = sec;
                    const playResult = vid.play();
                    if (playResult && typeof playResult.catch === 'function') {
                        playResult.catch(() => {});
                    }
                } catch (error) {
                    if (attempt < 20) {
                        parentWindow.setTimeout(() => seekLocalVideo(sec, attempt + 1), 100);
                    }
                }
            };

            if (vid.readyState === 0) {
                vid.addEventListener('loadedmetadata', applySeek, { once: true });
            } else {
                applySeek();
            }
        }

        if (parentWindow.__aiVideoSeekHandler) {
            parentDoc.removeEventListener(
                'click',
                parentWindow.__aiVideoSeekHandler,
                true
            );
        }

        parentWindow.__aiVideoSeekHandler = function(e) {
            const target = e.target;
            if (!target || typeof target.closest !== 'function') return;

            const a = target.closest('a[href*="#local_seek="]');
            if (!a) return;

            const match = (a.hash || '').match(/#local_seek=([0-9.]+)/);
            if (!match) return;

            const sec = Number.parseFloat(match[1]);
            if (!Number.isFinite(sec)) return;

            e.preventDefault();
            e.stopPropagation();
            seekLocalVideo(sec);
        };
        parentDoc.addEventListener(
            'click',
            parentWindow.__aiVideoSeekHandler,
            true
        );
        delete parentDoc.documentElement.dataset.aiVideoSeekBound;
        </script>
        """, height=0, width=0)
    
    
    # 个人知识库管理面板：常驻显示，避免展开器额外占位和高度误算。
    with media_col.container(height=220, border=True):
        col_c, col_n = st.columns([1, 2], gap="small")
        
        with col_c:
            st.caption("📦 物理移动与归档")
            
            # [编程大师2.0 终极 UX 版]：动态获取全量收藏夹列表
            current_library = get_library_data()
            available_cols = list(current_library.keys())
            if "默认收藏夹" in available_cols:
                available_cols.remove("默认收藏夹")
            available_cols.insert(0, "默认收藏夹")
            
            # 增加新建选项作为触发器
            available_cols.append("➕ [新建收藏夹...]")
            
            # 智能定位：默认选中当前视频所在的分类
            default_idx = available_cols.index(st.session_state.current_collection) if st.session_state.current_collection in available_cols else 0
            
            # 渲染下拉选择框
            selected_option = st.selectbox("选择目标收藏夹", available_cols, index=default_idx, label_visibility="collapsed")
            
            # 动态 UI 切换逻辑：如果选择了“新建”，才渲染输入框
            if selected_option == "➕ [新建收藏夹...]":
                target_col_name = st.text_input("请输入新分类名称", placeholder="例如: AI前沿知识", label_visibility="collapsed")
            else:
                target_col_name = selected_option
                    
        with col_n:
            st.caption("✍️ 我的专属思考笔记")
            new_notes = st.text_area(
                "在这里记录你对这期视频的感悟或闪光点...",
                value=st.session_state.current_notes,
                height=78,
                label_visibility="collapsed",
            )

        if st.button("💾 保存", key="save_archive_and_notes", type="primary", use_container_width=True):
            raw_name = target_col_name.strip() if target_col_name else ""
            clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip()

            if not clean_name or clean_name == "➕ [新建收藏夹...]":
                st.warning("⚠️ 请输入有效的收藏夹名称。")
            else:
                previous_file = st.session_state.current_file
                previous_collection = st.session_state.current_collection
                previous_notes = st.session_state.current_notes
                old_full_path = os.path.join(SRT_VAULT_DIR, previous_file)
                old_video_path = old_full_path.rsplit('.', 1)[0] + '.mp4'
                old_raw_backup_path = old_full_path + '.raw.bak'
                new_full_path = old_full_path
                new_video_path = old_video_path
                new_raw_backup_path = old_raw_backup_path
                srt_moved = False
                video_moved = False
                raw_backup_moved = False

                try:
                    if clean_name != previous_collection:
                        if clean_name == "默认收藏夹":
                            target_dir = SRT_VAULT_DIR
                        else:
                            target_dir = os.path.join(SRT_VAULT_DIR, clean_name)
                            os.makedirs(target_dir, exist_ok=True)

                        filename = os.path.basename(previous_file)
                        new_full_path = os.path.join(target_dir, filename)
                        new_video_path = new_full_path.rsplit('.', 1)[0] + '.mp4'
                        new_raw_backup_path = new_full_path + '.raw.bak'

                        if os.path.exists(new_full_path):
                            raise FileExistsError(f"目标收藏夹中已存在同名字幕：{filename}")
                        if os.path.exists(old_video_path) and os.path.exists(new_video_path):
                            raise FileExistsError(f"目标收藏夹中已存在同名视频：{os.path.basename(new_video_path)}")
                        if os.path.exists(old_raw_backup_path) and os.path.exists(new_raw_backup_path):
                            raise FileExistsError("目标收藏夹中已存在同名原始字幕备份")

                        os.rename(old_full_path, new_full_path)
                        srt_moved = True
                        if os.path.exists(old_raw_backup_path):
                            os.rename(old_raw_backup_path, new_raw_backup_path)
                            raw_backup_moved = True
                        if os.path.exists(old_video_path):
                            os.rename(old_video_path, new_video_path)
                            video_moved = True

                        if clean_name == "默认收藏夹":
                            new_rel_path = filename
                        else:
                            new_rel_path = os.path.join(clean_name, filename).replace('\\', '/')

                        st.session_state.current_file = new_rel_path
                        st.session_state.current_collection = clean_name

                    st.session_state.current_notes = new_notes
                    if not save_current_metadata():
                        raise RuntimeError("笔记与元数据未能写入磁盘")

                    if srt_moved:
                        try:
                            update_source_path(
                                derive_source_id(previous_file),
                                new_rel_path,
                                clean_name,
                            )
                        except KnowledgeCardError as card_error:
                            st.session_state["_knowledge_card_path_warning"] = (
                                f"字幕已移动，但知识卡片来源路径同步失败：{card_error}"
                            )
                        st.session_state["_pending_library_sidebar_hold"] = {
                            "collection": selected_collection,
                        }

                        st.toast(f"已移动到“{clean_name}”并保存笔记。", icon="💾")
                        st.rerun()
                    else:
                        st.toast("笔记已保存。", icon="💾")
                except Exception as e:
                    rollback_error = None
                    try:
                        if video_moved and os.path.exists(new_video_path) and not os.path.exists(old_video_path):
                            os.rename(new_video_path, old_video_path)
                        if raw_backup_moved and os.path.exists(new_raw_backup_path) and not os.path.exists(old_raw_backup_path):
                            os.rename(new_raw_backup_path, old_raw_backup_path)
                        if srt_moved and os.path.exists(new_full_path) and not os.path.exists(old_full_path):
                            os.rename(new_full_path, old_full_path)
                    except Exception as rollback_exception:
                        rollback_error = rollback_exception

                    st.session_state.current_file = previous_file
                    st.session_state.current_collection = previous_collection
                    st.session_state.current_notes = previous_notes
                    st.session_state.pop("_pending_library_sidebar_hold", None)

                    if rollback_error:
                        st.error(f"保存失败，且文件回滚失败：{rollback_error}")
                    else:
                        st.error(f"保存失败，已恢复原状态：{e}")
            
    with summary_col:
        st.markdown('<span id="ai-summary-pane-marker"></span>', unsafe_allow_html=True)
        knowledge_card_tab_label = get_knowledge_card_tab_label(
            derive_source_id(st.session_state.current_file)
        )
        if hasattr(st, "segmented_control"):
            active_main_view = st.segmented_control(
                "右侧工作区",
                MAIN_VIEW_OPTIONS,
                key="main_view",
                format_func=lambda value: format_main_view_option(
                    value,
                    knowledge_card_tab_label,
                ),
                label_visibility="collapsed",
            )
        else:
            active_main_view = st.radio(
                "右侧工作区",
                MAIN_VIEW_OPTIONS,
                key="main_view",
                format_func=lambda value: format_main_view_option(
                    value,
                    knowledge_card_tab_label,
                ),
                horizontal=True,
                label_visibility="collapsed",
            )

        if active_main_view == MAIN_VIEW_KNOWLEDGE:
            render_knowledge_card_workbench()
            st.stop()

        correction_status = st.session_state.subtitle_correction.get("status")
        correction_changes = []
        correction_diff_error = ""
        if correction_status == "completed":
            correction_changes, correction_diff_error = load_subtitle_correction_changes(
                st.session_state.current_file,
                st.session_state.base_srt_content,
            )
            visible_change_count = (
                len(correction_changes)
                if not correction_diff_error
                else int(st.session_state.subtitle_correction.get("changed_count") or 0)
            )
            transcript_expander_label = f"📝 字幕 · AI 已校对 {visible_change_count} 处"
        elif correction_status == "fallback":
            transcript_expander_label = "📝 字幕 · 使用原始文本"
        else:
            transcript_expander_label = "📝 点击查看纯净版底层文本"

        with st.expander(transcript_expander_label, expanded=False):
            if correction_status == "completed":
                if correction_diff_error:
                    st.warning(correction_diff_error)
                elif correction_changes:
                    st.markdown(f"**校对修改明细 · {len(correction_changes)} 处**")
                    render_subtitle_correction_changes(correction_changes)
                else:
                    st.caption("✅ AI 已完成上下文校对，程序对比未发现文本变化。")
                st.caption("完整校正版字幕")
            elif correction_status == "fallback":
                st.caption("⚠️ AI 校对未完成，本视频使用原始字幕。")
            st.write(st.session_state.transcript)

        chat_segments = get_chat_segments(st.session_state.messages)
        segment_profiles = {
            tuple(message_indexes): get_segment_summary_profile(
                st.session_state.messages,
                message_indexes,
            )
            for message_indexes in chat_segments
        }
        has_summary_segment = any(segment_profiles.values())
        segment_key_counts = {}
        reroll_request = None
        if not has_summary_segment:
            _, empty_reroll_col = st.columns([0.82, 0.18], gap="small")
            with empty_reroll_col:
                if st.button(
                    "🔄 重新抽卡",
                    key="reroll_summary_without_existing_segment",
                    help="先识别原视频最匹配的板块，再排除它并使用其他板块重新总结",
                    use_container_width=True,
                ):
                    reroll_request = {"profile": "", "label": ""}

        for message_indexes in chat_segments:
            segment_fingerprint = get_chat_segment_fingerprint(
                st.session_state.messages,
                message_indexes,
            )
            duplicate_number = segment_key_counts.get(segment_fingerprint, 0)
            segment_key_counts[segment_fingerprint] = duplicate_number + 1
            segment_key = f"{segment_fingerprint}_{duplicate_number}"

            first_content = st.session_state.messages[message_indexes[0]].get("content", "")
            segment_preview = re.sub(r"\s+", " ", first_content).strip()
            if len(segment_preview) > 48:
                segment_preview = segment_preview[:48] + "..."

            segment_summary_profile = segment_profiles[tuple(message_indexes)]
            if segment_summary_profile:
                _, reroll_col, delete_col = st.columns(
                    [0.84, 0.08, 0.08],
                    gap="small",
                )
                with reroll_col:
                    if st.button(
                        "🔄",
                        key=f"reroll_summary_segment_{segment_key}",
                        help=(
                            f"排除「{segment_summary_profile['label']}」，"
                            "选择最接近的其他板块重新总结"
                        ),
                        use_container_width=True,
                    ):
                        reroll_request = dict(segment_summary_profile)
            else:
                _, delete_col = st.columns([0.92, 0.08], gap="small")
            with delete_col:
                with st.popover("🗑️", help="删除下方这段对话"):
                    st.caption(f"即将删除：{segment_preview or '空白对话'}")
                    if st.button(
                        "确认删除",
                        key=f"confirm_delete_chat_segment_{segment_key}",
                        type="primary",
                        use_container_width=True,
                    ):
                        previous_messages = st.session_state.messages
                        indexes_to_delete = set(message_indexes)
                        st.session_state.messages = [
                            message
                            for index, message in enumerate(previous_messages)
                            if index not in indexes_to_delete
                        ]
                        if save_current_metadata():
                            st.toast("本段对话已删除。", icon="🗑️")
                            st.rerun()
                        else:
                            st.session_state.messages = previous_messages

            for message_index in message_indexes:
                msg = st.session_state.messages[message_index]
                with st.chat_message(msg["role"]):
                    display_content = format_chat_content(msg["content"], st.session_state.source_url, is_local=video_exists)

                    summary_profile = msg.get("summary_profile")
                    if msg["role"] == "assistant" and isinstance(summary_profile, dict):
                        profile = str(summary_profile.get("profile") or "")
                        if profile in SUMMARY_PROFILES:
                            profile_label = str(
                                summary_profile.get("label") or SUMMARY_PROFILES[profile]
                            )
                            confidence = summary_profile.get("confidence")
                            confidence_text = ""
                            if isinstance(confidence, (int, float)) and confidence > 0:
                                confidence_text = f" · 匹配度 {confidence:.0%}"
                            st.caption(f"总结结构：{profile_label}{confidence_text}")

                    # [编程大师2.0 UI 优化]：自动折叠冗长的 Prompt 预设，保持界面清爽
                    if msg.get("message_kind") == "summary_reroll_request":
                        st.caption(display_content)
                    elif msg["role"] == "user" and ("# 角色设定" in display_content or "请仔细阅读文本" in display_content or len(display_content) > 200):
                        with st.expander("⚙️ 展开查看系统任务指令 (Prompt)"):
                            st.markdown(display_content, unsafe_allow_html=True)
                    else:
                        st.markdown(display_content, unsafe_allow_html=True)

        if reroll_request:
            previous_messages = copy.deepcopy(st.session_state.messages)
            try:
                baseline_route, new_route, new_summary = generate_alternative_summary(
                    reroll_request["profile"]
                )
                reroll_metadata = {
                    "from_profile": baseline_route.profile,
                    "from_label": baseline_route.label,
                    "to_profile": new_route.profile,
                    "to_label": new_route.label,
                    "confidence": new_route.confidence,
                    "reason": new_route.reason,
                    "created_at": now_china_iso(),
                }
                st.session_state.messages.extend(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"🔄 换板块重总结：{baseline_route.label} → "
                                f"{new_route.label}"
                            ),
                            "message_kind": "summary_reroll_request",
                            "summary_reroll": reroll_metadata,
                        },
                        {
                            "role": "assistant",
                            "content": new_summary,
                            "summary_profile": new_route.to_metadata(),
                            "summary_reroll": reroll_metadata,
                        },
                    ]
                )
                if not save_current_metadata():
                    raise RuntimeError("新总结未能保存到字幕档案")
                st.toast(
                    f"已改用「{new_route.label}」重新总结并保存。",
                    icon="🔄",
                )
                st.rerun()
            except Exception as error:
                st.session_state.messages = previous_messages
                st.error(f"重新总结失败，原总结未受影响：{error}")

        with st.form("qa_composer_form", clear_on_submit=False, border=False):
            input_col, video_send_col, free_send_col = st.columns(
                [0.64, 0.18, 0.18],
                gap="small",
                vertical_alignment="bottom",
            )
            with input_col:
                user_input = st.text_input(
                    "向 AI 提问",
                    key="qa_input",
                    placeholder="向 AI 提问，也可以点击左侧快捷指令...",
                    label_visibility="collapsed",
                )
            with video_send_col:
                send_video_first = st.form_submit_button(
                    QA_MODE_VIDEO_FIRST,
                    type="primary",
                    use_container_width=True,
                )
            with free_send_col:
                send_free_explore = st.form_submit_button(
                    QA_MODE_FREE_EXPLORE,
                    type="primary",
                    use_container_width=True,
                )

        prompt = preset_clicked
        clear_qa_input_after_success = False
        submitted_mode = None
        if send_video_first:
            submitted_mode = QA_MODE_VIDEO_FIRST
        elif send_free_explore:
            submitted_mode = QA_MODE_FREE_EXPLORE

        if submitted_mode:
            cleaned_user_input = user_input.strip()
            if cleaned_user_input:
                st.session_state.qa_mode = submitted_mode
                prompt = cleaned_user_input
                clear_qa_input_after_success = True
            else:
                st.warning("请输入问题后再选择回答方式。")
        
        if prompt:
            current_system_prompt = build_qa_system_prompt(
                st.session_state.transcript,
                st.session_state.qa_mode,
            )
            if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
                st.session_state.messages[0] = {"role": "system", "content": current_system_prompt}
            else:
                st.session_state.messages.insert(0, {"role": "system", "content": current_system_prompt})

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                # 即时发送时也同样触发折叠检测
                if "# 角色设定" in prompt or "请仔细阅读文本" in prompt or len(prompt) > 200:
                    with st.expander("⚙️ 展开查看系统任务指令 (Prompt)"):
                        st.markdown(prompt, unsafe_allow_html=True)
                else:
                    st.markdown(prompt, unsafe_allow_html=True)

        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
            payload_messages = get_payload_messages(st.session_state.messages)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    active_llm_settings = load_llm_settings()
                    llm_client = create_llm_client(active_llm_settings)
                    response_stream = llm_client.chat.completions.create(
                        model=active_llm_settings["model"],
                        messages=payload_messages, 
                        stream=True,
                        temperature=active_llm_settings["temperature"] 
                    )
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            # 【修复】：允许解析 HTML 标签
                            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    final_display = format_chat_content(full_response, st.session_state.source_url, is_local=video_exists)
                    # 【修复】：允许解析 HTML 标签
                    message_placeholder.markdown(final_display, unsafe_allow_html=True)
                    
                    # 保存进内存的依然是原生纯文本（带括号的时间戳），确保一致性
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    save_current_metadata()
                    if clear_qa_input_after_success:
                        st.session_state["_clear_qa_input"] = True
                    
                    st.rerun() 
                except Exception as e:
                    st.error(f"调用本地模型失败: {e}")
                    st.session_state.messages.pop() 
else:
    st.info("👈 请在左侧选中一个视频开始对话，或等待后台自动处理完成。")
