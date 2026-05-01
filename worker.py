import os
import time
import json
import re
import tempfile
import subprocess
import db
from openai import OpenAI

# ================= 配置区域 =================
SRT_VAULT_DIR = "srt_vault"
LOCAL_UPLOADS_DIR = "local_uploads"
os.makedirs(SRT_VAULT_DIR, exist_ok=True)
os.makedirs(LOCAL_UPLOADS_DIR, exist_ok=True)

LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
MODEL_NAME = "local_model"
MAX_CHARS_LIMIT = 26000

# ================= 工具函数 =================
def generate_srt_string(subtitles):
    if not subtitles: return ""
    srt_lines = []
    for sub in subtitles:
        srt_lines.append(str(sub.get('id', '')))
        srt_lines.append(f"{sub.get('start_time', '00:00:00,000')} --> {sub.get('end_time', '00:00:00,000')}")
        srt_lines.append(str(sub.get('text', '')))
        srt_lines.append("") 
    return "\n".join(srt_lines)

def sanitize_filename(title):
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    return safe_title.strip()[:100]

def extract_pure_text_from_srt(srt_content):
    pure_lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        if not line: continue
        if line.isdigit(): continue  
        if '-->' in line: continue   
        pure_lines.append(line)
    return " ".join(pure_lines)

def download_audio_from_url(url, task_id):
    import yt_dlp
    output_template = os.path.join(tempfile.gettempdir(), f"worker_{task_id[:8]}_%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        # 🚀【工业级防风控】：读取静态的 cookies.txt 文件，绕过文件占用和加密限制
        'cookiefile': 'cookies.txt',
        # 伪装常见的 User-Agent
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        }
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        expected_filename = ydl.prepare_filename(info)
        if not os.path.exists(expected_filename):
            for ext in ['webm', 'm4a', 'mp3', 'mp4']:
                alt_path = expected_filename.rsplit('.', 1)[0] + f".{ext}"
                if os.path.exists(alt_path): return alt_path
        return expected_filename

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
            print(f"[{title}] 🌐 下载网页音频...")
            temp_media_path = download_audio_from_url(source_path, task_id)
            if not temp_media_path: raise Exception("音频下载失败")
        else:
            temp_media_path = source_path
            if not os.path.exists(temp_media_path): raise Exception(f"找不到本地文件: {source_path}")

        db.update_task_status(task_id, "transcribing", 10)
        print(f"[{title}] 🎙️ 唤醒 Whisper 进行识别...")
        _, temp_json_path = tempfile.mkstemp(suffix=".json")
        
        command = ["python", "whisper_worker.py", "--input", temp_media_path, "--output", temp_json_path]
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"], custom_env["PYTHONUNBUFFERED"] = "utf-8", "1"
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=custom_env, bufsize=1)
        
        last_update = time.time()
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue
            if line.startswith("[PROGRESS]"):
                try:
                    pct = float(line.split()[1])
                    if time.time() - last_update > 2.0:
                        db.update_task_status(task_id, "transcribing", int(10 + pct * 0.89))
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

# ================= 阶段 2：脑力总结 (大模型) =================
def process_llm_phase(task):
    task_id, title = task['id'], task['title']
    source_type, source_path = task['source_type'], task['source_path']
    final_srt_path = get_target_srt_path(title, task_id)
    
    try:
        db.update_task_status(task_id, "summarizing", 0)
        print(f"[{title}] 🤖 开始大模型自动总结并记录源链接...")
        
        if not os.path.exists(final_srt_path):
            raise Exception("未找到对应的字幕文件进行总结")
            
        with open(final_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            
        pure_text = extract_pure_text_from_srt(srt_content)
        if len(pure_text) > MAX_CHARS_LIMIT:
            pure_text = pure_text[:MAX_CHARS_LIMIT] + "...(为防止显存溢出，后续内容已截断)"
            
        sys_prompt = f"你是一个专业的视频内容分析助手。以下是视频的完整字幕文本（已净化）：\n\n{pure_text}\n\n请根据上述文本回答用户的问题。如果超出文本范围，请如实告知。"
        user_prompt = "请简要给出以下内容：1.全视频的核心内容；2.这个视频的信息密度，是否值得学习；3.是否可能是暗广"
        
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.3)
        
        ai_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": ai_reply})
        
        # 【核心修复】：升级为带 source_url 的字典格式落盘
        metadata_to_save = {
            "source_url": source_path if source_type == "url" else "",
            "history": [m for m in messages if m["role"] != "system"]
        }
        metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
        
        with open(final_srt_path, 'a', encoding='utf-8') as f:
            f.write("\n\n====================== AI_CHAT_HISTORY ======================\n")
            f.write(metadata_json)
            
        db.update_task_status(task_id, "completed", 100)
        print(f"[{title}] ✨ 总结与源链接已封入字幕文件！")
        
    except Exception as e:
        print(f"[{title}] ⚠️ 总结失败或跳过: {e}")
        db.update_task_status(task_id, "completed", 100) 

# ================= 调度引擎 =================
def run_worker_loop():
    print("🚀 [Worker] 智能调度进程已启动，等待派单...")
    while True:
        try:
            with db.get_conn() as conn:
                cursor = conn.execute("SELECT * FROM video_tasks WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1")
                pending_task = cursor.fetchone()
                if pending_task:
                    process_whisper_phase(dict(pending_task))
                    continue 
                
                cursor = conn.execute("SELECT * FROM video_tasks WHERE status = 'awaiting_llm' ORDER BY created_at ASC LIMIT 1")
                llm_task = cursor.fetchone()
                if llm_task:
                    process_llm_phase(dict(llm_task))
                    continue
                    
            time.sleep(3) 
        except Exception as e:
            print(f"[Worker] 致命错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_worker_loop()