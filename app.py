import os
import sqlite3
import json
import copy
import re
import requests
import streamlit as st
from openai import OpenAI

# ================= 配置区域 =================
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
MODEL_NAME = "local_model" 
MAX_CHARS_LIMIT = 26000       
MAX_HISTORY_TOKENS = 28000   
SRT_VAULT_DIR = "srt_vault"
DB_PATH = "tasks.db"
DELIMITER = "\n\n====================== AI_CHAT_HISTORY ======================\n"

PRESETS = {
    "📝 总结": "请简要给出以下内容：1.全视频的核心内容；2.这个视频的信息密度，是否值得学习；3.是否可能是暗广",
    "🗺️ 大纲": "请梳理视频的逻辑脉络，输出一份清晰的思维导图大纲。",
    "💬 大白话": "请用大白话总结一下",
    "🤫 是暗广吗？": "请评价这期视频是否可能是隐性广告？",
    "📌 要点": "请提取视频中的核心要点，用列表形式呈现。",
    "❓ Q&A": "请根据视频内容，整理出观众最可能关心的几个问题并解答。"
}

os.makedirs(SRT_VAULT_DIR, exist_ok=True)
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# ================= 工具函数 =================
def extract_pure_text_from_srt(srt_content):
    pure_lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        if not line: continue
        if line.isdigit(): continue  
        if '-->' in line: continue   
        pure_lines.append(line)
    return " ".join(pure_lines)

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
st.set_page_config(page_title="视听内容 AI 助手", page_icon="🎬", layout="wide")
st.title("🎬 视听内容 AI 助手")

if "current_file" not in st.session_state: st.session_state.current_file = None
if "messages" not in st.session_state: st.session_state.messages = []
if "transcript" not in st.session_state: st.session_state.transcript = ""
if "base_srt_content" not in st.session_state: st.session_state.base_srt_content = ""
# 【核心修复】：必须初始化这个状态
if "source_url" not in st.session_state: st.session_state.source_url = ""

preset_clicked = None

# --- 左侧边栏 (极简无拷贝版) ---
with st.sidebar:
    with st.popover("📂 导入本地音视频 (点击展开)", use_container_width=True):
        st.caption("无需复制！直接粘贴本地文件路径：")
        local_path = st.text_input(" ", placeholder="如: D:\\Videos\\test.mp4", label_visibility="collapsed")
        
        if local_path and st.button("🚀 开始处理", use_container_width=True, type="primary"):
            clean_path = local_path.strip().strip('"').strip("'")
            if os.path.exists(clean_path):
                try:
                    res = requests.post("http://127.0.0.1:8000/api/tasks", json={
                        "source_type": "local_file",
                        "source_path": clean_path,
                        "title": os.path.splitext(os.path.basename(clean_path))[0]
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
                    try: os.remove(target_path)
                    except: pass
                st.session_state.current_file = None
                st.session_state.messages = []
                st.session_state.transcript = ""
                st.session_state.source_url = ""
                st.toast("✅ 字幕及绑定的历史记录已彻底删除！", icon="🗑️")
                st.rerun()

    srt_files = get_srt_files()
    if not srt_files:
        st.caption("空空如也，快去抓取视频吧！")
        selected_file = None
    else:
        with st.container(height=400, border=True):
            selected_file = st.radio("选择视频", srt_files, label_visibility="collapsed")
    st.divider()

    st.markdown("**💡 快捷指令**")
    for label, prompt_text in PRESETS.items():
        if st.button(label, use_container_width=True):
            preset_clicked = prompt_text

# --- 上下文切换与兼容读取逻辑 ---
if selected_file and selected_file != st.session_state.current_file:
    st.session_state.current_file = selected_file
    st.session_state.source_url = "" # 先清空旧的链接
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
                # 【核心兼容读取】：如果是旧版列表则只读对话，如果是新版字典则拆分链接和对话
                if isinstance(data, list):
                    saved_history = data
                elif isinstance(data, dict):
                    saved_history = data.get("history", [])
                    st.session_state.source_url = data.get("source_url", "")
                st.toast("📚 已自动恢复 AI 讨论记录", icon="🕰️")
            except Exception as e:
                pass
            
        pure_text = extract_pure_text_from_srt(raw_srt)
        is_truncated = False
        if len(pure_text) > MAX_CHARS_LIMIT:
            pure_text = pure_text[:MAX_CHARS_LIMIT] + "...(截断)"
            is_truncated = True
            
        st.session_state.transcript = pure_text
        sys_prompt = f"你是一个专业的视频内容分析助手。以下是视频的完整字幕文本（已净化）：\n\n{pure_text}\n\n请根据上述文本回答用户的问题。如果超出文本范围，请如实告知。"
        
        st.session_state.messages = [{"role": "system", "content": sys_prompt}] + saved_history
        if is_truncated: st.toast("⚠️ 视频过长，已智能截断以保护内存。", icon="✂️")
    except Exception as e:
        st.error(f"加载失败: {e}")

# --- 主对话区 ---
if st.session_state.transcript and st.session_state.current_file:
    
    # 【核心修复】：渲染原视频跳转按钮
    title_col, link_col = st.columns([0.7, 0.3])
    with title_col:
        st.subheader(f"💬 {st.session_state.current_file.rsplit('_', 1)[0]}")
    with link_col:
        if st.session_state.source_url:
            st.link_button("🔗 观看原视频", st.session_state.source_url, use_container_width=True)
            
    with st.expander("📝 点击查看纯净版底层文本", expanded=False):
        st.write(st.session_state.transcript)
        
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    user_input = st.chat_input("向 AI 提问 (也可以点击左侧快捷按钮)...")
    prompt = preset_clicked or user_input
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

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
                    temperature=0.3 
                )
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # 【核心修复】：重新落盘时，也是按新格式（含 source_url 的字典）保存
                file_path = os.path.join(SRT_VAULT_DIR, st.session_state.current_file)
                metadata_to_save = {
                    "source_url": st.session_state.source_url,
                    "history": [m for m in st.session_state.messages if m["role"] != "system"]
                }
                metadata_json = json.dumps(metadata_to_save, ensure_ascii=False, indent=2)
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(st.session_state.base_srt_content + DELIMITER + metadata_json)
                except Exception as e:
                    st.error(f"保存历史记录失败: {e}")
                
                st.rerun() 
            except Exception as e:
                st.error(f"调用本地模型失败: {e}")
                st.session_state.messages.pop() 
else:
    st.info("👈 请在左侧选中一个视频开始对话，或等待后台自动处理完成。")