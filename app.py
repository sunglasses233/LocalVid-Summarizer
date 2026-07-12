import os
import sqlite3
import json
import copy
import re
import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ================= 统一配置 =================
from config import (
    AUTO_DOWNLOAD_VIDEO,
    COOKIES_FILE,
    DB_PATH,
    DOWNLOAD_MAX_HEIGHT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    MAX_CHARS_LIMIT,
    MAX_HISTORY_TOKENS,
    MODEL_NAME,
    SRT_VAULT_DIR,
    TASK_API_URL,
    VIDEO_VAULT_DIR,
    DEFAULT_USE_VOCAL_SEPARATION,
)

DELIMITER = "\n\n====================== AI_CHAT_HISTORY ======================\n"

PRESETS = {
    "📝 总结": "请仔细阅读文本，按照以下框架输出：\n1. 🎯 **核心摘要**\n2. 🗺️ **逻辑脉络**\n3. ⚖️ **对比/亮点**（如有对比，【必须】使用标准的换行 Markdown 表格，如：\n| 维度 | A | B |\n|---|---|---|\n| 1 | 2 | 3 |）\n\n【排版纪律】（不要输出规则本身）：\n- 必须使用 Emoji 和**加粗**。\n- 所有要点必须带时间戳，严格使用单圆括号包裹，如 `(01:30)` 或 `(01:30-02:15)`，严禁分开写。",
    "🗺️ 大纲": "请梳理视频的逻辑脉络，输出一份清晰的、带有 Emoji 装饰的思维导图大纲，**必须**在每个节点后标注对应的时间戳 `(MM:SS)`。",
    "💬 大白话": "请用通俗易懂的大白话、类似朋友聊天的口吻总结这个视频，提炼出 3-4 个最接地气的要点，并在关键句后附带时间戳 `(MM:SS)`。",
    "🤫 是暗广吗？": "请像一位资深的内容审核员一样，分析这期视频是否可能是隐性广告。请列出**怀疑点**和**澄清点**的对比表格，并在引用证据时标注时间戳 `(MM:SS)`。",
    "📌 要点": "请提取视频中的核心要点，用层级分明的 Markdown 列表呈现，加粗核心词，并逐条附带时间戳 `(MM:SS)`。",
    "❓ Q&A": "请根据视频内容，整理出观众最可能关心的 3-5 个问题并解答。使用一问一答的格式，并在答案的依据处标注时间戳 `(MM:SS)`。"
}

os.makedirs(SRT_VAULT_DIR, exist_ok=True)
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# ================= 工具函数 =================
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
        
# [编程大师2.0 新增]：统一的数据固化保存函数 (遵循 DRY 原则，降低冗余)
def save_current_metadata():
    if not st.session_state.current_file: return
    file_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
    metadata_to_save = {
        "source_url": st.session_state.source_url,
        "collection": st.session_state.current_collection,
        "notes": st.session_state.current_notes,
        "history": [m for m in st.session_state.messages if m["role"] != "system"]
    }
    metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(st.session_state.base_srt_content + DELIMITER + metadata_json)
    except Exception as e:
        st.error(f"保存元数据失败: {e}")

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

def calculate_tokens(messages):
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars * 1.2)

def get_payload_messages(messages):
    payload = copy.deepcopy(messages)
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

# ================= Streamlit 界面构建 =================
st.set_page_config(page_title="AI视频总结助手", page_icon="🎬", layout="wide")
st.title("🎬 AI视频总结助手")

# [编程大师2.0 新增]：注入全局 CSS 引擎，彻底杀死标题自动锚点，实现无痕纯净复制
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

if "current_file" not in st.session_state: st.session_state.current_file = None
if "messages" not in st.session_state: st.session_state.messages = []
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "base_srt_content" not in st.session_state: st.session_state.base_srt_content = ""
# 【核心修复】：必须初始化这个状态
if "source_url" not in st.session_state: st.session_state.source_url = ""
if "current_collection" not in st.session_state: st.session_state.current_collection = "默认收藏夹"
if "current_notes" not in st.session_state: st.session_state.current_notes = ""

preset_clicked = None

# --- 左侧边栏 (极简无拷贝版) ---
with st.sidebar:
    with st.popover("📂 导入本地音视频 (点击展开)", use_container_width=True):
        st.caption("无需复制！直接粘贴本地文件路径：")
        local_path = st.text_input(" ", placeholder="如: D:\\Videos\\test.mp4", label_visibility="collapsed")
        
        # 新增可选配置
        # 新增可选配置
        use_denoise = st.checkbox("🎛️ 开启 AI 深度降噪 (适用于嘈杂视频，较慢)", value=DEFAULT_USE_VOCAL_SEPARATION)
        auto_download = st.checkbox("📥 添加任务时自动下载原片", value=AUTO_DOWNLOAD_VIDEO)
        
        if local_path and st.button("🚀 开始处理", use_container_width=True, type="primary"):
            clean_path = local_path.strip().strip('"').strip("'")
            if os.path.exists(clean_path):
                try:
                    res = requests.post(TASK_API_URL, json={
                        "source_type": "local_file",
                        "source_path": clean_path,
                        "title": os.path.splitext(os.path.basename(clean_path))[0],
                        # 注入配置
                        "options": {"use_vocal_separation": use_denoise, "auto_download": auto_download}
                    })
                    if res.status_code == 200: st.toast("✅ 已加入队列！", icon="🚀")
                    else: st.error("加入队列失败")
                except Exception as e:
                    st.error(f"无法连接调度中心: {e}")
            else:
                st.error("❌ 找不到该文件，请检查路径。")

    st.markdown("**📡 任务看板**")
    @st.fragment(run_every="3s")
    def auto_refresh_task_board():
        tasks = get_tasks_from_db()
        if not tasks:
            st.caption("💤 当前无排队任务")
        else:
            active_task = next((t for t in tasks if t['status'] in ['downloading', 'transcribing', 'summarizing']), None)
            if active_task:
                status_icon = {"downloading": "🌐", "transcribing": "🎙️", "summarizing": "🤖"}.get(active_task['status'], "⚡")
                st.markdown(f"{status_icon} `{active_task['title'][:12]}...` **{active_task['progress']}%**")
                st.progress(active_task['progress'] / 100.0)
            else:
                waiting_llm = [t for t in tasks if t['status'] == 'awaiting_llm']
                if waiting_llm:
                    st.caption(f"☕ {len(waiting_llm)} 个视频已识别，排队总结中...")
                else:
                    st.caption("💤 GPU 待命中")

            with st.expander("📊 历史记录", expanded=False):
                for t in tasks:
                    emoji = {"pending": "⏳", "downloading": "🌐", "transcribing": "🎙️", "awaiting_llm": "☕", "summarizing": "🤖", "completed": "✅", "error": "❌"}.get(t['status'], "❓")
                    st.caption(f"{emoji} {t['title'][:15]}... ({t['progress']}%)")
                
    auto_refresh_task_board()
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

    # 渲染下拉框供用户选择试图
    selected_collection = st.selectbox("📂 选择收藏夹视图", all_collections, label_visibility="collapsed")
    
    # 动态过滤出当前选中的收藏夹下的文件列表
    filtered_files = library_data.get(selected_collection, [])
    
    if not filtered_files:
        st.caption("📭 当前收藏夹空空如也！")
        selected_file = None
    else:
        with st.container(height=400, border=True):
            # [视觉优化]：后台使用相对路径定位，前台只显示干净的文件名
            selected_file = st.radio(
                "选择视频", 
                filtered_files, 
                format_func=lambda x: os.path.basename(x), 
                label_visibility="collapsed"
            )
    st.divider()

    st.markdown("**💡 快捷指令**")
    for label, prompt_text in PRESETS.items():
        if st.button(label, use_container_width=True):
            preset_clicked = prompt_text

# --- 上下文切换与兼容读取逻辑 ---
if selected_file and selected_file != st.session_state.current_file:
    st.session_state.current_file = selected_file
    st.session_state.source_url = "" # 先清空旧的链接
    st.session_state.current_collection = "默认收藏夹" # 清空旧收藏状态
    st.session_state.current_notes = "" # 清空旧笔记
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
                elif isinstance(data, dict):
                    saved_history = data.get("history", [])
                    st.session_state.source_url = data.get("source_url", "")
                    st.session_state.current_notes = data.get("notes", "")
                    
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
        is_truncated = False
        if len(pure_text) > MAX_CHARS_LIMIT:
            pure_text = pure_text[:MAX_CHARS_LIMIT] + "...(截断)"
            is_truncated = True
            
        st.session_state.transcript = pure_text
        # [编程大师2.0 升级版]：强化时间感知与输出约束的 Prompt
        sys_prompt = (
            f"你是一个专业的视频内容分析助手。以下是带有时间戳标记的视频完整字幕文本：\n\n"
            f"{pure_text}\n\n"
            f"【核心约束指令】：\n"
            f"1. 请根据上述文本准确回答用户的问题。\n"
            f"2. 文本中方括号内的内容（如 [01:30]）代表时间戳。在归纳要点、梳理大纲或引用原文时，**必须**在关键信息后附带该段落对应的原视频时间戳。\n"
            f"3. 输出的时间戳格式必须严格包裹在圆括号内，即 `(MM:SS)` 或 `(HH:MM:SS)`。例如：'2. 屏幕与UI检查是否流畅 (01:30)'。\n"
            f"4. 如果问题超出文本范围，请如实告知。"
        )
        
        st.session_state.messages = [{"role": "system", "content": sys_prompt}] + saved_history
        if is_truncated: st.toast("⚠️ 视频过长，已智能截断以保护内存。", icon="✂️")
    except Exception as e:
        st.error(f"加载失败: {e}")

# --- 主对话区 ---
if st.session_state.transcript and st.session_state.current_file:

# [编程大师2.0 终极容错版]：大模型排版急救与清洗引擎
    def format_chat_content(content, url, is_local=False):
        if not url or not url.startswith("http"):
            return content
            
        content = re.sub(r'([：:】\*])\s*\|', r'\1\n\n|', content)
        content = re.sub(r'\|\s*\|', '|\n|', content)
        
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
    title_col, link_col, dl_col = st.columns([0.6, 0.2, 0.2])
    
    current_srt_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
    current_video_path = current_srt_path.rsplit('.', 1)[0] + '.mp4'
    video_exists = os.path.exists(current_video_path)
    
    with title_col:
        st.subheader(f"💬 {st.session_state.current_file.rsplit('_', 1)[0]}")
        
    with link_col:
        if st.session_state.source_url:
            st.link_button("🔗 观看原网页", st.session_state.source_url, use_container_width=True)
            
    with dl_col:
        if video_exists:
            # UI 联动优化 1：感知到本地视频存在，提供独立的【仅删除视频】功能！
            if st.button("🗑️ 删除本地原片", use_container_width=True, type="secondary"):
                try:
                    os.remove(current_video_path)
                    st.toast("✅ 本地原片已删除，释放硬盘空间！", icon="🗑️")
                    st.rerun()
                except Exception as e: pass
        elif st.session_state.source_url:
            if st.button("📥 下载 1080P 原片", use_container_width=True, type="secondary"):
                with st.spinner("🚀 正在拉取高清音视频轨并进行无损合并，请耐心稍候..."):
                    try:
                        save_dir = os.path.dirname(current_srt_path)
                        base_name = os.path.basename(current_srt_path).rsplit('.', 1)[0]
                        download_high_res_video(st.session_state.source_url, save_dir, base_name)
                        st.toast("✅ 资产已归档入库！", icon="🎉")
                        st.rerun()
                    except Exception as e: st.error(f"❌ 下载失败: {e}")

# [编程大师2.0 终极杀招]：内嵌播放器 + 底层 JS 跨维度遥控
    if video_exists:
        with st.expander("🎬 本地内嵌播放器 (超清/无广告)", expanded=True):
            st.video(current_video_path)
            
            # 注入隐形黑科技脚本 (高度设为0，静默执行)
            components.html("""
            <script>
            const parentDoc = window.parent.document;
            
            // 1. [视觉优化]：动态感知并限制竖屏视频的逆天高度
            setInterval(() => {
                const vids = parentDoc.querySelectorAll('video');
                vids.forEach(vid => {
                    if (!vid.dataset.optimized) {
                        vid.style.maxHeight = '60vh'; 
                        vid.style.objectFit = 'contain';
                        vid.style.borderRadius = '8px';
                        vid.dataset.optimized = 'true';
                    }
                });
            }, 500);

            // 2. [交互优化]：拦截所有携带 #local_seek= 的超链接，直接操纵上方播放器进度条
            parentDoc.addEventListener('click', function(e) {
                const a = e.target.closest('a');
                if (a && a.href.includes('#local_seek=')) {
                    e.preventDefault(); 
                    const sec = parseFloat(a.href.split('#local_seek=')[1]);
                    const vid = parentDoc.querySelector('video');
                    if (vid) {
                        vid.currentTime = sec;
                        vid.play();
                        // 让屏幕平滑滚动到视频播放器位置
                        vid.scrollIntoView({behavior: "smooth", block: "center"});
                    }
                }
            });
            </script>
            """, height=0, width=0)
    
    
    # [编程大师2.0 新增]：个人知识库管理面板 (包含收藏夹移动与笔记)
    with st.expander("📝 专属笔记与收藏夹管理 (点击展开)", expanded=False):
        col_c, col_n = st.columns([1, 2])
        
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

            if st.button("跨物理目录移动", use_container_width=True):
                raw_name = target_col_name.strip() if target_col_name else ""
                # 1. 路径净化防御：剔除操作系统禁止的特殊字符
                clean_name = re.sub(r'[\\/*?:"<>|]', "", raw_name).strip()
                
                # 防御性拦截：用户选了新建但没填名字
                if not clean_name or clean_name == "➕ [新建收藏夹...]":
                    st.warning("⚠️ 请输入有效的分类名称！")
                elif clean_name != st.session_state.current_collection:
                    old_rel_path = st.session_state.current_file
                    old_full_path = os.path.join(SRT_VAULT_DIR, old_rel_path)
                    
                    # 2. 决定目标物理路径
                    if clean_name == "默认收藏夹":
                        target_dir = SRT_VAULT_DIR
                    else:
                        target_dir = os.path.join(SRT_VAULT_DIR, clean_name)
                        os.makedirs(target_dir, exist_ok=True)
                        
                    filename = os.path.basename(old_rel_path)
                    new_full_path = os.path.join(target_dir, filename)
                    
                    try:
                        # 3. 核心：执行操作系统的真实文件剪切
                        os.rename(old_full_path, new_full_path)
                        
                        # [编程大师2.0 新增]：连带移动本地视频！
                        old_video_path = old_full_path.rsplit('.', 1)[0] + '.mp4'
                        new_video_path = new_full_path.rsplit('.', 1)[0] + '.mp4'
                        if os.path.exists(old_video_path):
                            os.rename(old_video_path, new_video_path)
                            
                        # 4. 指针重定向：更新前端状态中的当前文件路径
                        if clean_name == "默认收藏夹":
                            st.session_state.current_file = filename
                        else:
                            st.session_state.current_file = os.path.join(clean_name, filename).replace('\\', '/')
                            
                        st.session_state.current_collection = clean_name
                        
                        # 5. 指针更新后，调用原有的保存函数，将新元数据写回文件尾部
                        save_current_metadata()
                        
                        # 6. GC 垃圾回收：如果老文件夹被搬空了，立即销毁
                        old_dir = os.path.dirname(old_full_path)
                        if old_dir != SRT_VAULT_DIR and os.path.isdir(old_dir):
                            if not os.listdir(old_dir):
                                try: os.rmdir(old_dir)
                                except: pass
                                
                        st.toast(f"✅ 已物理移至文件夹: {clean_name}", icon="📦")
                        st.rerun()
                    except Exception as e:
                        st.error(f"跨目录物理移动失败: {e}")
                    
        with col_n:
            st.caption("✍️ 我的专属思考笔记")
            new_notes = st.text_area("在这里记录你对这期视频的感悟或闪光点...", value=st.session_state.current_notes, height=100, label_visibility="collapsed")
            if st.button("💾 保存笔记入库", use_container_width=True):
                st.session_state.current_notes = new_notes
                save_current_metadata()
                st.toast("✅ 笔记已永久保存入库", icon="💾")
            
    with st.expander("📝 点击查看纯净版底层文本", expanded=False):
        st.write(st.session_state.transcript)
        
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                display_content = format_chat_content(msg["content"], st.session_state.source_url, is_local=video_exists)
                
                # [编程大师2.0 UI 优化]：自动折叠冗长的 Prompt 预设，保持界面清爽
                if msg["role"] == "user" and ("# 角色设定" in display_content or "请仔细阅读文本" in display_content or len(display_content) > 200):
                    with st.expander("⚙️ 展开查看系统任务指令 (Prompt)"):
                        st.markdown(display_content, unsafe_allow_html=True)
                else:
                    st.markdown(display_content, unsafe_allow_html=True)
                
    user_input = st.chat_input("向 AI 提问 (也可以点击左侧快捷按钮)...")
    prompt = preset_clicked or user_input
    
    if prompt:
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
                response_stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload_messages, 
                    stream=True,
                    temperature=LLM_TEMPERATURE 
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
                
                st.rerun() 
            except Exception as e:
                st.error(f"调用本地模型失败: {e}")
                st.session_state.messages.pop() 
else:
    st.info("👈 请在左侧选中一个视频开始对话，或等待后台自动处理完成。")