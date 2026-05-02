以下基本都是AI生成的，我也在B站做了个视频讲这玩意怎么用（BV1VxR5BPEjt、BV1xJoNBPEtD）

---

# 🎬 视听内容 AI 助手 - 总控司令部

这是一个强大的、完全支持本地化部署的 AI 视听内容处理系统。它能够自动提取视频/音频内容，利用 Whisper 进行高精度语音转写，并调用本地大模型（如通过 LM Studio）生成思维导图、内容大纲、核心要点等总结信息。

不仅支持本地文件的物理拖拽，还附带了专属的 Tampermonkey 油猴脚本，支持 B 站等平台的一键发送排队，实现端到端的无缝观影+总结体验。

## ✨ 核心特性

*   **⚡ 异步调度中心**：基于 FastAPI 构建的底层 API，任务进入 SQLite 数据库排队，不阻塞前端。
*   **🎙️ 智能双引擎转录**：针对中日文与表音文字（英语等）采取不同的断句路由策略，内建 FFmpeg 音轨净化。
*   **🤖 纯本地大模型接入**：兼容 LM Studio 等 OpenAI 格式 API，断网也能做视频分析，隐私 100% 安全。
*   **🖥️ 全能控制台**：基于 Streamlit 构建的可视化仪表盘，实时查看 GPU 任务进度、浏览历史字幕与 AI 对话记录。
*   **🪄 隐藏交互彩蛋**：支持主控台直接物理拖拽本地音视频文件，瞬间推送任务入列。

## 🛠️ 硬件与环境要求

*   **操作系统**：Windows / Linux
*   **Python**：>= 3.9
*   **GPU**：推荐配备充足显存的 NVIDIA 显卡（如 RTX 5070 Ti 16GB 或以上，以获取极速的 Whisper Large v3 转录及大模型推理体验）。
*   **外部依赖**：系统需安装并配置好 `FFmpeg` 环境变量。

## 🚀 快速启动

### 1. 克隆与安装依赖

```bash
git clone https://github.com/sunglasses233/LocalVid-Summarizer.git
cd LocalVid-Summarizer
pip install -r requirements.txt
```

*(注：请确保你已经安装了 `faster-whisper`, `fastapi`, `streamlit`, `openai`, `yt_dlp` 等依赖)*

### 2. 配置本地大模型

本系统默认连接本地运行的兼容 OpenAI API 格式的大模型服务。
请打开你的 [LM Studio](https://lmstudio.ai/)，加载任意你喜欢的模型，并启动 Local Server（默认端口 `[http://127.0.0.1:1234/v1](http://127.0.0.1:1234/v1)`）。

### 3. 配置视频下载 Cookie (可选)

如果你需要抓取需要登录的视频平台资源，请在项目根目录创建一个 `cookies.txt` 文件，并放入你的浏览器导出 Cookies。

### 4. 一键启动系统

在项目根目录下运行总控脚本（请根据你实际的入口文件名修改，如 `launcher.py`）：

```bash
python launcher.py
```

系统将自动依次启动：
1. `FastAPI` 后台调度中心 (端口 8000)
2. `Whisper` 消费进程
3. `Streamlit` 交互界面

启动成功后，浏览器会自动打开 `http://localhost:8501`。

## 🔌 浏览器扩展 (Tampermonkey)

项目中包含了一个专属的油猴脚本：`🎬 一键发送至本地视听 AI 助手.user.js`（原名 `油猴.txt`）。
*   将其安装到你的浏览器 Tampermonkey 插件中。
*   当你在 B 站（支持播放页、首页推荐卡片、稍后再看等）浏览时，会出现 🤖 **AI 按钮**。
*   点击即可将视频一键推送到你本地运行的 FastAPI 后台进行排队处理！

## 📁 目录结构简介

*   `launcher.py` - 一键启动所有微服务的总控入口。
*   `server.py` - FastAPI 任务接收与分发中心。
*   `worker.py` - 核心业务逻辑（含视频下载、调用 Whisper 进程、请求 LLM）。
*   `whisper_worker.py` - 独立的 Whisper 进程，负责高强度的语音识别，避免阻塞主进程。
*   `app.py` - Streamlit 编写的现代化 UI 交互端。
*   `db.py` - SQLite 数据库封装，记录任务状态与进度。

## 🤝 贡献与支持

欢迎提交 Pull Request 或 Issue 探讨更多功能，比如支持更多视频平台、引入不同的大模型视觉能力等。如果觉得好用，请给个 ⭐ Star！

## 📄 开源协议

本项目采用 MIT License 开源，详情请参见 LICENSE 文件。


---

需要注意的小问题（懒得改了）：
whisper_worker.py代码第 57 行写死了模型下载路径：download_root=r"G:\WhisperModels"。在别人的电脑上大概率会报错。  
行动建议：建议将其修改为相对路径（如 ./models）或使用环境变量，例如：download_root=os.environ.get("WHISPER_MODEL_DIR", "./models")。

另外如果本地没有whisper模型的话，可以将“whisper_worker.py”的这一串代码修改一下，就会自动下载对应的模型：
'''
model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="float16", 
            download_root=r"G:\WhisperModels", 
            local_files_only=False  # 改为 False 允许自动下载
        )
'''

第一次运行可能会多花一点时间下载，之后就会直接读取本地缓存，非常方便！

---
后续升级计划：
- [ ] 更好的whisper模型
- [ ] 支持导出更多格式
- [ ] 支持文件夹
- [ ] 支持更多平台
