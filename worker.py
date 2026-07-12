import os
import time
import json
import re
import tempfile
import subprocess
import sys
import db
import requests
from openai import OpenAI

# ================= 统一配置 =================
from config import (
    AUTO_DOWNLOAD_VIDEO,
    BASE_DIR,
    COOKIES_FILE,
    DEFAULT_USE_VOCAL_SEPARATION,
    DOWNLOAD_MAX_HEIGHT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    LOCAL_UPLOADS_DIR,
    MAX_CHARS_LIMIT,
    MODEL_NAME,
    SRT_VAULT_DIR,
    WHISPER_MODEL_NAME,
)

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


def download_high_res_video(url, save_dir, base_name):
    import yt_dlp 
    
    # 移除固定的 ext=mp4 限制，允许拉取最高画质的 Dash 流，然后交由 FFmpeg 强行无损封装为 mp4
    format_str = f'bestvideo[height<={DOWNLOAD_MAX_HEIGHT}]+bestaudio/best[height<={DOWNLOAD_MAX_HEIGHT}]/best'
    
    ydl_opts = {
        'format': format_str,
        'outtmpl': os.path.join(save_dir, f"{base_name}.%(ext)s"),
        'merge_output_format': 'mp4',
        'cookiefile': str(COOKIES_FILE), # 依赖最新 cookies 突破 B站 1080P 限制
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
        'cookiefile': str(COOKIES_FILE),
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
        options_str = task.get('options')
        task_options = json.loads(options_str) if options_str else {}
        use_denoise = task_options.get('use_vocal_separation', DEFAULT_USE_VOCAL_SEPARATION)
        
        command = [sys.executable, str(BASE_DIR / "whisper_worker.py"), "--input", temp_media_path, "--output", temp_json_path, "--model", WHISPER_MODEL_NAME]
        if use_denoise:
            command.append("--denoise")
        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"], custom_env["PYTHONUNBUFFERED"] = "utf-8", "1"
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=custom_env, bufsize=1)
        
        last_update = time.time()
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line: continue
            
            # [编程大师2.0 修复]：将底层的非进度条日志，透传打印到主控制台！
            if not line.startswith("[PROGRESS]"):
                print(line)
                
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
# ================= 阶段 2：脑力总结 (大模型) =================
def process_llm_phase(task):
    task_id, title = task['id'], task['title']
    source_type, source_path = task['source_type'], task['source_path']
    final_srt_path = get_target_srt_path(title, task_id)
    
    try:
        db.update_task_status(task_id, "summarizing", 0)
        print(f"[{title}] 🤖 开始大模型自动总结并准备持久化元数据...")
        
        if not os.path.exists(final_srt_path):
            raise Exception("未找到对应的字幕文件进行总结")
            
        with open(final_srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            
        # 🛡️ 防御性隔离 1：在执行任何不可控的 API 请求前，先将前端刚需的 UI 基础数据解析完毕
        display_url = ""
        if source_type == "url":
            display_url = source_path
        elif source_type == "direct_url":
            display_url = source_path.split("|||")[0] if "|||" in source_path else source_path

        # 🛡️ 防御性隔离 2：防止重试机制引发的重复追加脏数据
        if "====================== AI_CHAT_HISTORY ======================" in srt_content:
            print(f"[{title}] ⚠️ 检测到该字幕已包含元数据，跳过写入，直接完结任务。")
            db.update_task_status(task_id, "completed", 100)
            return

        pure_text = extract_pure_text_from_srt(srt_content)
        if len(pure_text) > MAX_CHARS_LIMIT:
            pure_text = pure_text[:MAX_CHARS_LIMIT] + "...(为防止显存溢出，后续内容已截断)"
            
        # [编程大师2.0 终极版]：系统强认知约束
        sys_prompt = (
            f"你是一个专业的视频内容分析助手。以下是带有时间戳标记的视频完整字幕文本：\n\n"
            f"{pure_text}\n\n"
            f"【核心约束指令】：\n"
            f"1. 请根据上述文本准确回答用户的问题。\n"
            f"2. 文本中方括号内的内容（如 [01:30]）代表时间戳。在引用信息时，**必须**附带对应的时间戳。\n"
            f"3. 输出的时间戳格式必须严格包裹在圆括号内，并【绝对忠实于原文的时间层级】！如果原文带有小时（如 [01:06:48]），就必须输出 (01:06:48)，绝对不允许擅自删减成 (06:48)！\n"
            f"4. 如果问题超出文本范围，请如实告知。"
        )

        # [编程大师2.0 终极版]：动态领域路由 + 强时间锚点
        user_prompt = """# 角色设定
你是一个全能型、深度的视频内容分析引擎。你的任务是将未经人工校对、带有明显口语化的 ASR 转录文本，结构化提炼为高信息密度、逻辑严密、细节丰满的深度笔记。

# 核心处理原则 (Strict Rules)
1. 自动降噪：静默修复同音错字，过滤情绪化发泄和广告推销。
2. 提升信息密度：必须保留核心论证过程、数据指标和生动案例。宁可详尽，不可遗漏关键逻辑链（建议 800-1500 字）。
3. 逻辑重构：绝不按时间轴流水账总结。必须打破原文顺序，按内在逻辑重新排列。

# 执行步骤 (Workflow)
严格按照以下 Markdown 结构输出，必须正确使用回车换行！

### 🎯 核心结论 (TL;DR)
- 用 2-3 句话一针见血地概括视频的最核心价值或最终结论。

### 🧠 深度逻辑解构 (必须详尽展开)
（请在后台静默判断视频领域，并**只选择以下最匹配的一个框架**进行输出。绝对不要把未选中的框架提示词输出出来！）

- **如果是【数码测评/硬件】**：使用 Markdown 表格横向对比核心参数与优缺点；分点展开实测触发场景与购买建议。
- **如果是【前沿科学/科普】**：深入浅出解释底层工作原理；详述突破了什么历史局限；列出现阶段的缺陷或未解之谜。
- **如果是【心理/情感/人际】**：深度剖析矛盾本质；拆解背后的心理学动因；给出切实可行的应对策略或话术示范。
- **如果是【金融商业/硬核分析】**：提取所有关键数据、金额、比例作为论据；推演利弊影响及行业传导链条。
- **如果是【通用干货/方法论】**：明确痛点；详述解决问题的具体步骤（Step-by-Step），包含动作和避坑指南。

### 📂 案例与论据库
- **核心案例复盘**：详细提取 1-2 个核心案例（背景、行动、结果），绝对不能一笔带过。
- **关键数据/公式**：列出文本中提到的重要数据模型。

### 💡 关键细节与金句
- 提取 3 个最引人深思的金句或反共识观点。
- 汇总需要避坑的“易错点”或“盲区”。

# 🔴 【最高排版纪律】（违规将导致系统崩溃，请绝对遵守）
1. **时间锚点覆盖**：在上述每一个列表要点、表格行、案例和金句的末尾，【必须】附带原文本中的具体时间戳！
2. **格式红线**：时间戳必须严格包在一个单圆括号内，并忠实于原文长度！有小时必须带小时（如 (01:06:48)），没小时带分钟（如 (01:30)），时间段请用 (01:06:48-01:08:17)。绝对不允许擅自删减时间位数！
3. **表格规范**：如果输出表格，必须严格正确换行，不要挤在同一行。"""

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        llm_success = False
        
        # 🛡️ 防御性隔离 3：将网络调用完全包裹在内部 Try 块中
        try:
            client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=LLM_TEMPERATURE)
            ai_reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": ai_reply})
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
            "history": [m for m in messages if m["role"] != "system"]
        }
        metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
        
        with open(final_srt_path, 'a', encoding='utf-8') as f:
            f.write("\n\n====================== AI_CHAT_HISTORY ======================\n")
            f.write(metadata_json)
            
# ================= [编程大师2.0 新增]：后台静默物理归档引擎 =================
        options_str = task.get('options')
        task_options = json.loads(options_str) if options_str else {}
        # 逻辑：前端传来的配置优先级最高，如果没有传（比如通过油猴发来的），则听从 worker.py 的全局设定
        should_download = task_options.get('auto_download', AUTO_DOWNLOAD_VIDEO)
        
        if should_download and display_url:
            print(f"[{title}] 📥 触发自动归档设定，正在后台静默拉取 1080P 原片...")
            try:
                base_name = os.path.basename(final_srt_path).rsplit('.', 1)[0]
                download_high_res_video(display_url, SRT_VAULT_DIR, base_name)
                print(f"[{title}] ✅ 1080P 原片后台物理归档成功！")
            except Exception as e:
                print(f"[{title}] ⚠️ 自动下载原片失败: {e}")
        
        db.update_task_status(task_id, "completed", 100)
        
        if not llm_success:
            print(f"[{title}] 🛡️ 兜底机制生效：已安全写入源链接元数据，前端 UI 功能不受任何影响！")
        
    except Exception as e:
        print(f"[{title}] ❌ 总结阶段发生严重外层异常: {e}")
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