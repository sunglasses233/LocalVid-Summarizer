import os
import time
import json
import re
import tempfile
import subprocess
import sys
import threading
from urllib.parse import urlparse
import db
import requests
from llm_settings import create_llm_client, load_llm_settings
from subtitle_correction import (
    correct_srt_with_llm,
    load_persisted_correction,
    persist_corrected_srt,
)
from summary_generation import (
    build_summary_messages,
    classify_summary_profile,
    fallback_summary_route,
    limit_transcript_to_complete_segments,
    resolve_summary_char_limit,
)

# ================= 统一配置 =================
from config import (
    AUTO_DOWNLOAD_VIDEO,
    BASE_DIR,
    COOKIES_FILE,
    DEFAULT_USE_VOCAL_SEPARATION,
    DOWNLOAD_MAX_HEIGHT,
    LOCAL_UPLOADS_DIR,
    SRT_VAULT_DIR,
    WHISPER_MODEL_NAME,
)

WORKER_IDLE_SECONDS = 3
LLM_PROGRESS_INTERVAL_SECONDS = 1.5
CORRECTION_MIN_PROGRESS = 5
CORRECTION_MAX_PROGRESS = 25
LLM_MIN_PROGRESS = 38
LLM_MAX_PROGRESS = 95
YTDLP_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BILIBILI_REFERER = "https://www.bilibili.com/"

VIDEO_METADATA_TEXT_LIMIT = 2000
VIDEO_STAT_FIELDS = (
    "like_count",
    "comment_count",
    "collect_count",
    "share_count",
    "forward_count",
    "download_count",
    "play_count",
)
SPEAKER_LABEL_PATTERN = re.compile(r"^(?:说话人\d+|其他声音|说话人未知)$")

# ================= 工具函数 =================
def generate_srt_string(subtitles):
    if not subtitles: return ""
    srt_lines = []
    for sub in subtitles:
        srt_lines.append(str(sub.get('id', '')))
        srt_lines.append(f"{sub.get('start_time', '00:00:00,000')} --> {sub.get('end_time', '00:00:00,000')}")
        text = str(sub.get('text', ''))
        speaker = str(sub.get('speaker') or '').strip()
        if speaker and SPEAKER_LABEL_PATTERN.fullmatch(speaker):
            text = f"[{speaker}] {text}".rstrip()
        srt_lines.append(text)
        srt_lines.append("") 
    return "\n".join(srt_lines)


def normalize_primary_speaker_count(value):
    normalized = str(value if value is not None else "2").strip().lower()
    if normalized in {"auto", "自动", "-1"}:
        return -1
    try:
        count = int(normalized)
    except (TypeError, ValueError):
        return 2
    return count if 2 <= count <= 20 else 2


def build_whisper_command(
    media_path,
    output_path,
    use_denoise=False,
    use_speaker_diarization=False,
    primary_speakers=2,
):
    command = [
        sys.executable,
        str(BASE_DIR / "whisper_worker.py"),
        "--input",
        str(media_path),
        "--output",
        str(output_path),
        "--model",
        WHISPER_MODEL_NAME,
    ]
    if use_denoise:
        command.append("--denoise")
    if use_speaker_diarization:
        command.extend(
            ["--diarize", "--speakers", str(normalize_primary_speaker_count(primary_speakers))]
        )
    return command

def sanitize_filename(title):
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    return safe_title.strip()[:100]


def parse_task_options(task):
    value = task.get("options") if isinstance(task, dict) else None
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def is_bilibili_url(url):
    try:
        hostname = (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return False
    return hostname == "b23.tv" or hostname == "bilibili.com" or hostname.endswith(
        ".bilibili.com"
    )


def build_yt_dlp_network_options(url):
    headers = {"User-Agent": YTDLP_BROWSER_USER_AGENT}
    if is_bilibili_url(url):
        headers["Referer"] = BILIBILI_REFERER

    options = {"http_headers": headers}
    if COOKIES_FILE.exists():
        options["cookiefile"] = str(COOKIES_FILE)
    return options


def format_yt_dlp_download_error(url, error):
    message = str(error)
    if is_bilibili_url(url) and "HTTP Error 412" in message:
        return (
            "B站拒绝了下载请求（HTTP 412）。请升级 yt-dlp 后重新启动软件；"
            "若仍失败，请重新导出已登录 B站 的 cookies.txt，或稍后重试。"
            f"原始错误：{message}"
        )
    return message


def _metadata_text(value, limit=VIDEO_METADATA_TEXT_LIMIT):
    return str(value or "").strip()[:limit]


def _metadata_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number != number or number in (float("inf"), float("-inf")):
        return None
    return int(number) if number.is_integer() else number


def _without_empty_values(data):
    return {
        key: value
        for key, value in data.items()
        if value not in (None, "", {}, [])
    }


def normalize_video_metadata(value):
    if not isinstance(value, dict):
        return {}

    author = value.get("author") if isinstance(value.get("author"), dict) else {}
    statistics = (
        value.get("statistics") if isinstance(value.get("statistics"), dict) else {}
    )
    media = value.get("media") if isinstance(value.get("media"), dict) else {}

    normalized_stats = {}
    for field in VIDEO_STAT_FIELDS:
        number = _metadata_number(statistics.get(field))
        if number is None or (field == "play_count" and number <= 0):
            continue
        normalized_stats[field] = number

    normalized = {
        "schema_version": 1,
        "platform": _metadata_text(value.get("platform"), 32),
        "video_id": _metadata_text(value.get("video_id"), 128),
        "title": _metadata_text(value.get("title")),
        "source_url": _metadata_text(value.get("source_url"), 4000),
        "author": _without_empty_values(
            {
                "name": _metadata_text(author.get("name"), 500),
                "user_id": _metadata_text(author.get("user_id"), 256),
                "sec_uid": _metadata_text(author.get("sec_uid"), 512),
                "follower_count": _metadata_number(author.get("follower_count")),
                "total_favorited": _metadata_number(author.get("total_favorited")),
            }
        ),
        "published_at": _metadata_text(value.get("published_at"), 64),
        "published_at_unix": _metadata_number(value.get("published_at_unix")),
        "captured_at": _metadata_text(value.get("captured_at"), 64),
        "statistics": normalized_stats,
        "media": _without_empty_values(
            {
                "duration_seconds": _metadata_number(media.get("duration_seconds")),
                "width": _metadata_number(media.get("width")),
                "height": _metadata_number(media.get("height")),
                "quality": _metadata_text(media.get("quality"), 64),
                "cover_url": _metadata_text(media.get("cover_url"), 4000),
            }
        ),
    }
    return _without_empty_values(normalized)


def download_high_res_video(url, save_dir, base_name):
    import yt_dlp 
    
    # 移除固定的 ext=mp4 限制，允许拉取最高画质的 Dash 流，然后交由 FFmpeg 强行无损封装为 mp4
    format_str = f'bestvideo[height<={DOWNLOAD_MAX_HEIGHT}]+bestaudio/best[height<={DOWNLOAD_MAX_HEIGHT}]/best'
    
    ydl_opts = {
        'format': format_str,
        'outtmpl': os.path.join(save_dir, f"{base_name}.%(ext)s"),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        **build_yt_dlp_network_options(url),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            return True
    except Exception as error:
        raise Exception(
            f"下载引擎报错: {format_yt_dlp_download_error(url, error)}"
        ) from error


def extract_pure_text_from_srt(srt_content):
    """
    [编程大师2.0 升级版]：保留时间戳的 SRT 解析器
    将 SRT 转换为带有时间戳标记的连续文本，供 LLM 消费并建立时间维度认知。
    """
    result_lines = []
    current_time = ""
    
    for line in srt_content.splitlines():
        line = line.strip()
        if not line: 
            continue
            
        # 过滤独立的纯数字序号
        if line.isdigit() and not current_time: 
            continue  
            
        # 捕获时间轴格式 (例如: 00:01:30,000 --> 00:01:35,000)
        if '-->' in line:
            start_raw = line.split('-->')[0].strip()
            
            # 提取核心时间，砍掉毫秒位以节省 Token ("00:01:30,000" -> "00:01:30")
            base_time = start_raw.split(',')[0].split('.')[0]
            
            # 智能压缩格式：如果视频不足1小时，抹去前置的 "00:" (如 "00:01:30" -> "01:30")
            if base_time.startswith("00:"):
                base_time = base_time[3:]
                
            current_time = base_time
            continue
            
        # 组装文本
        if current_time:
            # 拼接格式：[01:30] 视频文本
            result_lines.append(f"[{current_time}] {line}")
            # 置空 current_time，防止 SRT 格式异常时同一段落打多次时间戳
            current_time = "" 
        else:
            # 防御性逻辑：如果当前行是纯文本但没有时间戳（如多行字幕），则追加到上一句话中
            if result_lines:
                result_lines[-1] += f" {line}"
            else:
                result_lines.append(line)
                
    # 用空格拼接，保证最后送给大模型的是一段带锚点的自然流文本
    return " ".join(result_lines)

def download_audio_from_url(url, task_id):
    import yt_dlp
    output_template = os.path.join(tempfile.gettempdir(), f"worker_{task_id[:8]}_%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        **build_yt_dlp_network_options(url),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            expected_filename = ydl.prepare_filename(info)
            if not os.path.exists(expected_filename):
                for ext in ['webm', 'm4a', 'mp3', 'mp4']:
                    alt_path = expected_filename.rsplit('.', 1)[0] + f".{ext}"
                    if os.path.exists(alt_path): return alt_path
            return expected_filename
    except Exception as error:
        raise Exception(format_yt_dlp_download_error(url, error)) from error

# [编程大师2.0 新增]：极速直链下载器（前端提供解密后的 CDN 裸链接，彻底绕过反爬）
def download_direct_media(url, task_id):
    output_path = os.path.join(tempfile.gettempdir(), f"worker_{task_id[:8]}_direct.mp4")
    # 【防御性编程：防盗链伪装】部分 CDN 会拒绝没有 Referer 的裸请求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://www.douyin.com/'
    }
    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return output_path
    except Exception as e:
        print(f"极速下载失败: {e}")
        return None
        
def get_target_srt_path(title, task_id):
    safe_name = sanitize_filename(title)
    return os.path.join(SRT_VAULT_DIR, f"{safe_name}_{task_id[:8]}.srt")

# ================= 阶段 1：苦力干活 (Whisper) =================
def process_whisper_phase(task):
    task_id, source_type, source_path, title = task['id'], task['source_type'], task['source_path'], task['title']
    temp_media_path, temp_json_path = None, None
    
    try:
        if source_type == "url":
            db.update_task_status(task_id, "downloading", 0)
            print(f"[{title}] 🌐 [常规模式] 下载网页音频...")
            temp_media_path = download_audio_from_url(source_path, task_id)
            if not temp_media_path: raise Exception("音频下载失败")
            
        elif source_type == "direct_url":
            db.update_task_status(task_id, "downloading", 0)
            print(f"[{title}] ⚡ [极速直链模式] 正在拉取无水印源文件...")
            
            # [编程大师2.0 新增]：拆包逻辑，提取 ||| 后面的 MP4 直链用来干活
            direct_url = source_path.split("|||")[1] if "|||" in source_path else source_path
            
            temp_media_path = download_direct_media(direct_url, task_id)
            if not temp_media_path: raise Exception("直链下载失败")
            
        else:
            temp_media_path = source_path
            if not os.path.exists(temp_media_path): raise Exception(f"找不到本地文件: {source_path}")

        db.update_task_status(task_id, "transcribing", 10)
        print(f"[{title}] 🎙️ 唤醒 Whisper 进行识别...")
        _, temp_json_path = tempfile.mkstemp(suffix=".json")
        
        # [编程大师2.0 新增]：解析动态配置
        task_options = parse_task_options(task)
        use_denoise = task_options.get('use_vocal_separation', DEFAULT_USE_VOCAL_SEPARATION)
        use_speaker_diarization = bool(
            task_options.get("use_speaker_diarization", False)
        )
        primary_speakers = normalize_primary_speaker_count(
            task_options.get("speaker_count", 2)
        )
        
        command = build_whisper_command(
            temp_media_path,
            temp_json_path,
            use_denoise=use_denoise,
            use_speaker_diarization=use_speaker_diarization,
            primary_speakers=primary_speakers,
        )
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"], custom_env["PYTHONUNBUFFERED"] = "utf-8", "1"
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=custom_env, bufsize=1)
        
        last_update = time.time()
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue
            
            # [编程大师2.0 修复]：将底层的非进度条日志，透传打印到主控制台！
            if not line.startswith(("[PROGRESS]", "[DIARIZATION_PROGRESS]")):
                print(line)
                
            if line.startswith("[PROGRESS]"):
                try:
                    pct = float(line.split()[1])
                    if time.time() - last_update > 2.0:
                        if use_speaker_diarization:
                            mapped_progress = 99 if pct >= 100 else int(10 + pct * 0.69)
                        else:
                            mapped_progress = int(10 + pct * 0.89)
                        db.update_task_status(task_id, "transcribing", mapped_progress)
                        last_update = time.time()
                except: pass
            elif line.startswith("[DIARIZATION_PROGRESS]"):
                try:
                    pct = float(line.split()[1])
                    if time.time() - last_update > 2.0:
                        db.update_task_status(
                            task_id,
                            "transcribing",
                            int(80 + pct * 0.19),
                        )
                        last_update = time.time()
                except: pass

        process.wait()
        if process.returncode != 0: raise Exception("Whisper 崩溃")
            
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        srt_content = generate_srt_string(data.get("subtitles", []))
        final_srt_path = get_target_srt_path(title, task_id)
        
        with open(final_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
            
        db.update_task_status(task_id, "awaiting_llm", 100)
        print(f"[{title}] ⏸️ 识别结束，已进入大模型排队序列。")

    except Exception as e:
        print(f"[{title}] ❌ 识别失败: {e}")
        db.update_task_status(task_id, "error", 0)
    finally:
        if source_type == "url" and temp_media_path and os.path.exists(temp_media_path):
            try: os.unlink(temp_media_path)
            except: pass
        if temp_json_path and os.path.exists(temp_json_path):
            try: os.unlink(temp_json_path)
            except: pass

# ================= 阶段 2：字幕校对与总结 (大模型) =================
def get_stream_delta_content(chunk):
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "content", None) or ""

def call_llm_summary(task_id, title, client, settings, messages):
    chunks = []
    progress = LLM_MIN_PROGRESS
    last_update = time.time()

    try:
        response_stream = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=settings["temperature"],
            stream=True,
        )
        db.update_task_status(task_id, "summarizing", progress)
        for chunk in response_stream:
            content = get_stream_delta_content(chunk)
            if content:
                chunks.append(content)

            now = time.time()
            if now - last_update >= LLM_PROGRESS_INTERVAL_SECONDS:
                progress = min(LLM_MAX_PROGRESS, progress + 2)
                db.update_task_status(task_id, "summarizing", progress)
                last_update = now

        db.update_task_status(task_id, "summarizing", LLM_MAX_PROGRESS)
        return "".join(chunks)
    except Exception:
        if chunks:
            raise
        print(f"[{title}] 流式总结失败，尝试普通 API 调用。")

    db.update_task_status(task_id, "summarizing", 45)
    response = client.chat.completions.create(
        model=settings["model"],
        messages=messages,
        temperature=settings["temperature"],
    )
    db.update_task_status(task_id, "summarizing", 90)
    return response.choices[0].message.content or ""


def run_subtitle_correction(
    task_id,
    title,
    client,
    settings,
    srt_content,
    final_srt_path,
):
    stop_event = threading.Event()
    progress_lock = threading.Lock()
    progress_state = {"value": CORRECTION_MIN_PROGRESS}

    def publish_progress(value):
        bounded = max(CORRECTION_MIN_PROGRESS, min(CORRECTION_MAX_PROGRESS, int(value)))
        with progress_lock:
            progress_state["value"] = max(progress_state["value"], bounded)
            current_value = progress_state["value"]
        db.update_task_status(task_id, "correcting", current_value)

    def correction_progress(stage, current, total, message):
        if stage == "completed":
            publish_progress(CORRECTION_MAX_PROGRESS)
            return
        fraction = (current / total) if total else 0
        span = CORRECTION_MAX_PROGRESS - CORRECTION_MIN_PROGRESS
        publish_progress(CORRECTION_MIN_PROGRESS + int(span * fraction))

    def heartbeat():
        while not stop_event.wait(LLM_PROGRESS_INTERVAL_SECONDS):
            with progress_lock:
                next_value = min(
                    CORRECTION_MAX_PROGRESS - 1,
                    progress_state["value"] + 1,
                )
            publish_progress(next_value)

    publish_progress(CORRECTION_MIN_PROGRESS)
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name=f"SubtitleCorrectionProgress-{task_id[:8]}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        result = correct_srt_with_llm(
            srt_content=srt_content,
            title=title,
            client=client,
            settings=settings,
            progress=correction_progress,
        )
        raw_backup_path = persist_corrected_srt(
            final_srt_path,
            srt_content,
            result.corrected_srt,
        )
        publish_progress(CORRECTION_MAX_PROGRESS)
        return result, raw_backup_path
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)


def process_llm_phase(task):
    task_id, title = task['id'], task['title']
    source_type, source_path = task['source_type'], task['source_path']
    final_srt_path = get_target_srt_path(title, task_id)
    
    try:
        db.update_task_status(task_id, "correcting", 0)
        print(f"[{title}] 📝 开始结合上下文校对字幕，随后自动总结...")
        
        if not os.path.exists(final_srt_path):
            raise Exception("未找到对应的字幕文件进行总结")
            
        db.update_task_status(task_id, "correcting", 2)
        with open(final_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        db.update_task_status(task_id, "correcting", 4)
            
        # 🛡️ 防御性隔离 1：在执行任何不可控的 API 请求前，先将前端刚需的 UI 基础数据解析完毕
        display_url = ""
        if source_type == "url":
            display_url = source_path
        elif source_type == "direct_url":
            display_url = source_path.split("|||")[0] if "|||" in source_path else source_path

        task_options = parse_task_options(task)
        video_metadata = normalize_video_metadata(task_options.get("video_metadata"))

        # 🛡️ 防御性隔离 2：防止重试机制引发的重复追加脏数据
        if "====================== AI_CHAT_HISTORY ======================" in srt_content:
            print(f"[{title}] ⚠️ 检测到该字幕已包含元数据，跳过写入，直接完结任务。")
            db.update_task_status(task_id, "completed", 100)
            return

        active_llm_settings = load_llm_settings()
        client = None
        correction_metadata = {
            "status": "fallback",
            "model": active_llm_settings.get("model", ""),
            "changed_count": 0,
            "total_segments": 0,
            "raw_backup": "",
        }
        try:
            correction_result = load_persisted_correction(
                final_srt_path,
                active_llm_settings.get("model", ""),
            )
            recovered = correction_result is not None
            if recovered:
                raw_backup_path = final_srt_path + ".raw.bak"
                db.update_task_status(task_id, "correcting", CORRECTION_MAX_PROGRESS)
                print(f"[{title}] ♻️ 检测到已落盘的校对结果，跳过重复 API 调用。")
            else:
                client = create_llm_client(active_llm_settings)
                correction_result, raw_backup_path = run_subtitle_correction(
                    task_id,
                    title,
                    client,
                    active_llm_settings,
                    srt_content,
                    final_srt_path,
                )

            srt_content = correction_result.corrected_srt
            correction_metadata = {
                "status": "completed",
                "model": correction_result.model,
                "changed_count": correction_result.changed_count,
                "total_segments": correction_result.total_segments,
                "raw_backup": os.path.basename(str(raw_backup_path)),
                "recovered_after_restart": recovered,
            }
            print(
                f"[{title}] ✅ 字幕语义校对完成，共修正 "
                f"{correction_result.changed_count} 处。"
            )
        except Exception as correction_error:
            correction_metadata["error"] = str(correction_error)[:500]
            print(f"[{title}] ⚠️ 字幕校对失败，使用原始字幕继续总结: {correction_error}")

        pure_text = extract_pure_text_from_srt(srt_content)
        transcript_limit = limit_transcript_to_complete_segments(
            pure_text,
            resolve_summary_char_limit(active_llm_settings),
        )
        pure_text = transcript_limit.text
        if transcript_limit.truncated:
            print(
                f"[{title}] ⚠️ 字幕共 {transcript_limit.original_chars:,} 字符，"
                f"已按 {transcript_limit.limit:,} 字符上限保留到 "
                f"{transcript_limit.last_timestamp or '最近完整段落'}。"
            )
            
        # 先识别视频的内容组织形式，再使用对应的总结结构。
        db.update_task_status(task_id, "summarizing", 28)
        try:
            if client is None:
                client = create_llm_client(active_llm_settings)
            summary_route = classify_summary_profile(
                title,
                pure_text,
                client,
                active_llm_settings,
            )
        except Exception as route_error:
            summary_route = fallback_summary_route(route_error)
        if summary_route.source == "fallback":
            print(
                f"[{title}] ⚠️ 总结类型识别失败，使用通用模板: "
                f"{summary_route.error}"
            )
        else:
            print(
                f"[{title}] 🧭 已识别为{summary_route.label} "
                f"(置信度 {summary_route.confidence:.0%})"
            )
        db.update_task_status(task_id, "summarizing", 36)
        messages = build_summary_messages(title, pure_text, summary_route)
        llm_success = False
        
        # 🛡️ 防御性隔离 3：将网络调用完全包裹在内部 Try 块中
        try:
            if client is None:
                client = create_llm_client(active_llm_settings)
            db.update_task_status(task_id, "summarizing", 37)
            ai_reply = call_llm_summary(task_id, title, client, active_llm_settings, messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": ai_reply,
                    "summary_profile": summary_route.to_metadata(),
                }
            )
            llm_success = True
            print(f"[{title}] ✨ 大模型总结完毕！")
        except Exception as llm_error:
            print(f"[{title}] ⚠️ 大模型服务异常 (或未启动): {llm_error}")
            print(f"[{title}] 🔄 启动降级模式：仅保留视频源链接与字幕，清空历史记录。")
            # [关键]：如果不成功，必须清空 messages 列表。这样用户在网页端依然有一块干净的白板去手动提问。
            messages = []

        # 🛡️ 终极兜底：无论大模型是死是活，这一步都绝对会被执行，前端的命脉保住了！
        metadata_to_save = {
            "source_url": display_url, 
            "collection": "默认收藏夹", 
            "notes": "",              
            "history": [m for m in messages if m["role"] != "system"],
            "subtitle_correction": correction_metadata,
            "video_metadata": video_metadata,
        }
        metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
        
        with open(final_srt_path, 'a', encoding='utf-8') as f:
            f.write("\n\n====================== AI_CHAT_HISTORY ======================\n")
            f.write(metadata_json)
            
# ================= [编程大师2.0 新增]：后台静默物理归档引擎 =================
        # 逻辑：前端传来的配置优先级最高，如果没有传（比如通过油猴发来的），则听从 worker.py 的全局设定
        should_download = task_options.get('auto_download', AUTO_DOWNLOAD_VIDEO)
        
        if should_download and display_url:
            print(f"[{title}] 📥 触发自动归档设定，正在后台静默拉取 {DOWNLOAD_MAX_HEIGHT}P 原片...")
            try:
                base_name = os.path.basename(final_srt_path).rsplit('.', 1)[0]
                download_high_res_video(display_url, SRT_VAULT_DIR, base_name)
                print(f"[{title}] ✅ {DOWNLOAD_MAX_HEIGHT}P 原片后台物理归档成功！")
            except Exception as e:
                print(f"[{title}] ⚠️ 自动下载原片失败: {e}")
        
        db.update_task_status(task_id, "completed", 100)
        
        if not llm_success:
            print(f"[{title}] 🛡️ 兜底机制生效：已安全写入源链接元数据，前端 UI 功能不受任何影响！")
        
    except Exception as e:
        print(f"[{title}] ❌ 大模型校对/总结阶段发生严重外层异常: {e}")
        db.update_task_status(task_id, "completed", 100)
        
# ================= 调度引擎 =================
def fetch_next_task(status):
    with db.get_conn() as conn:
        cursor = conn.execute(
            "SELECT * FROM video_tasks WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (status,),
        )
        task = cursor.fetchone()
        return dict(task) if task else None

def run_whisper_loop():
    print("[Whisper Worker] 转录线程已启动，等待 pending 任务...")
    while True:
        try:
            task = fetch_next_task("pending")
            if task:
                process_whisper_phase(task)
                continue
            time.sleep(WORKER_IDLE_SECONDS)
        except Exception as e:
            print(f"[Whisper Worker] 调度错误: {e}")
            time.sleep(5)

def run_llm_loop():
    print("[LLM Worker] API 校对/总结线程已启动，等待 awaiting_llm 任务...")
    while True:
        try:
            task = fetch_next_task("awaiting_llm")
            if task:
                process_llm_phase(task)
                continue
            time.sleep(WORKER_IDLE_SECONDS)
        except Exception as e:
            print(f"[LLM Worker] 调度错误: {e}")
            time.sleep(5)

def run_worker_loop():
    print("[Worker] 并行调度已启动：转录线程 1 个，API 校对/总结线程 1 个。")
    workers = [
        threading.Thread(target=run_whisper_loop, name="WhisperWorker"),
        threading.Thread(target=run_llm_loop, name="LLMWorker"),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    run_worker_loop()
