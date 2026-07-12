以下基本都是AI生成的，我也在B站做了个视频讲这玩意怎么用（BV1VxR5BPEjt、BV1xJoNBPEtD）

---
# 🎬 AI视频总结助手

一个本地运行的视频字幕识别、AI 总结与视频知识管理工具。

它可以接收 B 站、抖音或本地音视频，自动完成音频提取、人声分离、Whisper 语音识别和大模型总结，并在总结内容后附带时间戳。点击时间戳，可以跳转到本地播放器或原视频对应位置。

> [!IMPORTANT]
> 当前版本主要面向 **Windows + NVIDIA 显卡** 用户，需要自行安装 Python、FFmpeg、CUDA/cuDNN，并准备本地 Whisper 模型和大模型服务。
>
> 本项目不是双击即用的软件，请按照本说明完成环境配置。

---

## ✨ 主要功能

- B 站、抖音视频一键发送到本地 AI 助手
- 本地视频、音频文件导入与黑框拖拽提交
- Faster-Whisper 本地语音识别
- 可选 AI 人声分离与背景音剥离
- 油猴插件提交时自动开启人声分离
- 大模型自动生成结构化总结
- 总结、大纲、要点、Q&A、大白话、暗广分析等快捷指令
- 所有核心内容自动附带视频时间戳
- 点击时间戳跳转到本地播放器或原网页
- 视频、字幕、AI 对话和个人笔记统一保存
- 收藏夹分类与本地文件物理移动
- 后台任务队列、进度显示与失败降级
- 可选自动归档 1080P 原视频

---

## 🧩 项目结构

```text
项目目录/
├─ config.py             # 统一配置文件，普通用户主要修改这里
├─ app.py                # Streamlit 网页界面
├─ server.py             # FastAPI 任务接收服务
├─ db.py                 # SQLite 任务数据库
├─ worker.py             # 下载、任务调度、首次总结与视频归档
├─ whisper_worker.py     # 音频处理、人声分离与 Whisper 转录
├─ launcher.py           # 一键启动全部服务
├─ requirements.txt      # Python 依赖清单
├─ 油猴V6.3.txt          # B站、抖音网页端油猴脚本
├─ cookies.txt           # 用户自行准备，不要上传到 GitHub
└─ README.md
```

首次运行后会自动创建：

```text
srt_vault/               # 字幕、总结、笔记、聊天记录和归档视频
tasks.db                 # 任务数据库
local_uploads/           # 本地文件相关目录
video_downloads/         # 视频相关目录
```

---

# 🚀 最短安装流程

已经熟悉 Python、CUDA 和 LM Studio 的用户，可以按下面顺序快速安装：

1. 安装 Python 3.10 或 3.11。
2. 安装 NVIDIA 驱动、CUDA 12、cuDNN 9。
3. 安装 FFmpeg，并加入系统 `PATH`。
4. 安装本项目 Python 依赖。
5. 下载 Faster-Whisper `large-v3` 模型。
6. 修改 `config.py` 中的模型目录和临时目录。
7. 在项目根目录放入有效的 `cookies.txt`。
8. 启动 LM Studio 的本地 API 服务。
9. 运行 `python launcher.py`。
10. 安装并启用 `油猴V6.3.txt`。

下面是每一步的详细说明。

---

# 一、硬件与系统要求

## 推荐环境

- Windows 10 / Windows 11 64 位
- Python 3.10 或 Python 3.11
- NVIDIA 显卡
- 建议显存 8GB 或以上
- 建议预留 20GB 以上磁盘空间
- CUDA 12
- cuDNN 9
- FFmpeg
- LM Studio，或其他兼容 OpenAI Chat Completions API 的大模型服务

## 为什么默认需要 NVIDIA 显卡？

`config.py` 默认使用：

```python
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
```

当前版本没有自动切换 CPU。没有 NVIDIA 显卡时，需要自行把配置改成：

```python
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
```

但 CPU 转录速度会明显降低，人声分离的 GPU 环境也可能需要另外调整。当前版本主要按照 NVIDIA 显卡环境测试。

---

# 二、安装 Python

建议安装 Python 3.10 或 Python 3.11。

安装时务必勾选：

```text
Add Python to PATH
```

安装后打开 PowerShell 或 CMD，检查：

```powershell
python --version
python -m pip --version
```

建议先升级 pip：

```powershell
python -m pip install --upgrade pip
```

---

# 三、安装 Python 依赖

项目根目录已经提供 `requirements.txt`。打开 PowerShell 或 CMD，进入项目目录后执行：

```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```

`requirements.txt` 会安装：

- Streamlit 网页界面
- FastAPI、Uvicorn 和 Pydantic
- OpenAI 兼容接口客户端
- yt-dlp 与 curl-cffi 浏览器指纹模拟组件
- Faster-Whisper
- Hugging Face 模型下载工具
- GPU 版 Audio Separator 人声分离组件

> [!IMPORTANT]
> 油猴 V6.3 提交任务时会强制发送 `use_vocal_separation: true`，因此准备使用油猴插件的用户必须安装 `audio-separator[gpu]`。它已经包含在本项目的 `requirements.txt` 中。
>
> `requirements.txt` 只负责 Python 包。FFmpeg、NVIDIA 驱动、CUDA、cuDNN、Whisper 模型、`Kim_Vocal_2.onnx`、LM Studio 和 `cookies.txt` 仍需按照后续章节单独准备。

如果 B 站或抖音后来因网站接口更新而无法解析，可以单独把 yt-dlp 更新到预发布版本：

```powershell
python -m pip install --upgrade --pre "yt-dlp[default,curl-cffi]"
```

安装完成后可检查：

```powershell
python -c "import fastapi, streamlit, requests, openai, yt_dlp, faster_whisper; print('核心依赖正常')"
```

检查人声分离：

```powershell
python -c "from audio_separator.separator import Separator; print('人声分离依赖正常')"
```

也可以运行：

```powershell
audio-separator --env_info
```

---

# 四、安装 FFmpeg

本项目会直接调用系统中的 `ffmpeg.exe`，用于：

- 从视频中抽取音轨
- 将音频转换成 Whisper 需要的格式
- 人声分离前后的音频转换
- 合并网页视频的音频轨和视频轨

请安装完整的 FFmpeg，并将 FFmpeg 的 `bin` 目录加入 Windows 环境变量 `PATH`。

检查是否安装成功：

```powershell
ffmpeg -version
```

能显示版本信息即表示配置成功。

> [!WARNING]
> 不要只执行 `pip install ffmpeg`。本项目需要的是 FFmpeg 可执行程序，而不是同名 Python 包。

---

# 五、配置 NVIDIA、CUDA 和 cuDNN

先确认 NVIDIA 驱动正常：

```powershell
nvidia-smi
```

当前新版 Faster-Whisper / CTranslate2 的 GPU 环境通常需要：

- CUDA 12
- cuDNN 9

如果运行时出现下列错误，一般是 CUDA、cuDNN 或环境变量问题：

```text
Could not locate cublas64_12.dll
Could not load cudnn_ops64_9.dll
Library cublas is not found
Library cudnn is not found
Requested float16 compute type, but the target device does not support it
```

请检查：

1. NVIDIA 驱动是否正常。
2. CUDA 12 是否正确安装。
3. cuDNN 9 是否正确安装。
4. CUDA 和 cuDNN 的 DLL 所在目录是否已加入系统 `PATH`。
5. 修改环境变量后是否重新打开了 PowerShell，必要时重启电脑。

Faster-Whisper 官方说明：

- https://github.com/SYSTRAN/faster-whisper

---

# 六、准备 Whisper 模型和人声分离模型

## 1. 建议的模型目录结构

例如统一放在：

```text
G:\WhisperModels\
├─ models--Systran--faster-whisper-large-v3\
│  └─ snapshots\...
└─ audio-separator-models\
   └─ Kim_Vocal_2.onnx
```

实际盘符可以自行修改。

## 2. 下载 Faster-Whisper `large-v3`

当前代码默认：

```python
WHISPER_MODEL_NAME = "large-v3"
WHISPER_LOCAL_FILES_ONLY = True
```

这表示程序只读取本地模型，不会在转录时自动联网下载 Whisper 模型。

可以执行：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download Systran/faster-whisper-large-v3 --cache-dir G:\WhisperModels
```

如果没有 `hf` 命令，先确认已经安装：

```powershell
python -m pip install --upgrade huggingface-hub
```

然后重新打开 PowerShell再试。

下载完成后，确保 `config.py` 中：

```python
WHISPER_MODEL_DIR = Path(r"G:\WhisperModels")
```

和实际下载目录完全一致。

## 3. 人声分离模型 `Kim_Vocal_2.onnx`

默认配置：

```python
VOCAL_MODEL_DIR = WHISPER_MODEL_DIR / "audio-separator-models"
VOCAL_SEPARATOR_MODEL_NAME = "Kim_Vocal_2.onnx"
```

最终模型路径应类似：

```text
G:\WhisperModels\audio-separator-models\Kim_Vocal_2.onnx
```

如果文件不存在，`audio-separator` 在第一次真正启用人声分离时通常会自动下载模型。

也可以把已有的模型手动复制到上述目录。

> 人声分离完成后产生的临时 WAV 会被程序自动清理，`Kim_Vocal_2.onnx` 模型文件不会被删除。

---

# 七、修改 `config.py`

普通用户主要只需要修改 `config.py`，不需要分别修改 `app.py`、`worker.py` 和 `whisper_worker.py`。

## 1. 大模型接口

默认使用 LM Studio：

```python
LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
MODEL_NAME = "local_model"
LLM_TEMPERATURE = 0.3
```

说明：

- `LLM_BASE_URL`：兼容 OpenAI API 的服务地址。
- `LLM_API_KEY`：LM Studio 未启用认证时可以保留当前值。
- `MODEL_NAME`：建议改成大模型服务实际使用的模型 ID。
- `LLM_TEMPERATURE`：数值越低，输出通常越稳定。

如果使用其他 OpenAI 兼容服务，改成对应的地址、密钥和模型名称即可。

## 2. 上下文与字幕长度

```python
MAX_CHARS_LIMIT = 26000
MAX_HISTORY_TOKENS = 28000
```

- `MAX_CHARS_LIMIT`：首次自动总结和后续打开视频时，最多读取多少字幕字符。
- `MAX_HISTORY_TOKENS`：网页连续对话保留的历史上限，是代码中的近似估算，并非模型的精确 Token 数。

如果大模型上下文较小，应适当降低这两个值。

参考设置：

```text
模型上下文约 16K：MAX_CHARS_LIMIT 可先尝试 10000～12000
模型上下文约 32K：MAX_CHARS_LIMIT 可先尝试 20000～26000
模型上下文约 64K：可保持默认值或适当增加
```

必须为系统提示词、历史对话和模型输出预留空间，不要把字幕上限直接设置成模型上下文的最大值。

## 3. 自动下载原片

```python
AUTO_DOWNLOAD_VIDEO = True
DOWNLOAD_MAX_HEIGHT = 1080
```

- `True`：任务完成后尝试自动归档原视频。
- `False`：只保存字幕和总结，需要时再手动下载。
- `DOWNLOAD_MAX_HEIGHT`：下载视频的最高分辨率。

## 4. Cookie 路径

```python
COOKIES_FILE = BASE_DIR / "cookies.txt"
```

默认要求把 `cookies.txt` 放在项目根目录。

## 5. B 站下载兼容设置

```python
YTDLP_IMPERSONATE = "chrome"
YTDLP_COOKIES_FROM_BROWSER = None
```

默认使用项目根目录的 `cookies.txt`。

如果希望直接读取浏览器登录状态，可以改为：

```python
YTDLP_COOKIES_FROM_BROWSER = "edge"
```

或：

```python
YTDLP_COOKIES_FROM_BROWSER = "chrome"
```

启用浏览器 Cookie 后，会优先使用浏览器登录状态。

如果读取 Chromium Cookie 失败，请完全关闭对应浏览器及其后台进程后重试。

## 6. 模型与临时目录

```python
WHISPER_MODEL_DIR = Path(r"G:\WhisperModels")
VOCAL_MODEL_DIR = WHISPER_MODEL_DIR / "audio-separator-models"
AUDIO_WORKSPACE_DIR = Path(r"D:\AI_Workspace")
```

请改成自己电脑真实存在的路径。

`AUDIO_WORKSPACE_DIR` 用于存放：

- 音频抽取产生的临时 WAV
- 人声分离的中间文件
- 送入 Whisper 前的标准化音频

长视频会短时间占用较多磁盘空间，建议选择空间充足的非系统盘。

## 7. 人声分离默认值

```python
DEFAULT_USE_VOCAL_SEPARATION = False
```

不同任务入口的行为如下：

| 提交方式 | 人声分离行为 |
|---|---|
| 油猴 V6.3 | 强制开启 |
| 网页导入本地文件 | 按网页复选框决定 |
| 黑框拖入本地文件 | 使用 `DEFAULT_USE_VOCAL_SEPARATION` |
| 其他未传入选项的任务 | 使用 `DEFAULT_USE_VOCAL_SEPARATION` |

因此即使这里是 `False`，通过油猴提交的 B 站和抖音视频仍然会自动做人声分离。

---

# 八、准备 `cookies.txt`

B 站下载通常需要有效的登录 Cookie。缺少或失效时，可能出现：

```text
HTTP Error 412: Precondition Failed
Unable to download JSON metadata
```

## 正确做法

1. 在浏览器中登录 B 站。
2. 确认目标视频可以正常播放。
3. 使用浏览器扩展导出 **Netscape 格式** 的 Cookie。
4. 将文件命名为：

```text
cookies.txt
```

5. 放在项目根目录，与 `config.py`、`worker.py` 同级。

目录应类似：

```text
项目目录/
├─ config.py
├─ worker.py
├─ launcher.py
└─ cookies.txt
```

> [!CAUTION]
> `cookies.txt` 相当于登录凭据，可能包含账号会话信息。
>
> 不要发送给别人，不要截图公开，不要上传到 GitHub。

Cookie 失效后需要重新导出。

---

# 九、配置 LM Studio

本项目默认连接：

```text
http://127.0.0.1:1234/v1
```

操作步骤：

1. 安装并打开 LM Studio。
2. 下载并加载一个支持较长上下文的对话模型。
3. 根据显存设置合理的 Context Length。
4. 进入 `Developer` 页面。
5. 启动本地 API Server。
6. 默认端口保持为 `1234`。

LM Studio 官方说明：

- https://lmstudio.ai/docs/developer/core/server
- https://lmstudio.ai/docs/developer/openai-compat

建议至少使用支持 32K 上下文的模型。显存不足时，降低上下文长度，同时降低 `config.py` 中的 `MAX_CHARS_LIMIT`。

如果 LM Studio 没有启动，程序仍会保存已经识别完成的字幕和原视频信息，但首次 AI 总结会进入降级模式。

---

# 十、安装油猴插件

## 1. 安装 Tampermonkey

在 Chrome、Edge 或其他 Chromium 浏览器中安装 Tampermonkey。

## 2. 导入脚本

1. 打开 Tampermonkey 管理面板。
2. 新建脚本。
3. 删除默认内容。
4. 将 `油猴V6.3.txt` 中的完整代码复制进去。
5. 保存并启用脚本。

## 3. 使用方式

启动本地项目后，打开 B 站或抖音网页。

油猴会在：

- 视频列表卡片
- 视频详情页

注入“发送给 AI 总结”按钮。

点击后，任务会发送到：

```text
http://127.0.0.1:8000/api/tasks
```

油猴 V6.3 会自动传入：

```javascript
options: {
    use_vocal_separation: true
}
```

因此油猴任务默认自动启用 AI 人声分离。

> [!NOTE]
> 油猴脚本无法读取 Python 的 `config.py`。
>
> 如果修改了 `API_PORT`，还需要手动修改油猴脚本中的：
>
> ```javascript
> const API_URL = "http://127.0.0.1:8000/api/tasks";
> ```

网页结构改变后，油猴按钮可能暂时失效，需要更新选择器或脚本版本。

---

# 十一、启动项目

启动项目前，建议确认：

- LM Studio 已加载模型并启动 API Server
- `cookies.txt` 已放入项目根目录
- `config.py` 中的模型和临时目录已经修改
- FFmpeg、CUDA、cuDNN 和 Python 依赖已经安装

在项目根目录打开 PowerShell 或 CMD：

```powershell
python launcher.py
```

`launcher.py` 会同时启动：

1. FastAPI 调度中心
2. Worker 后台任务进程
3. Streamlit 网页界面

浏览器通常会自动打开 Streamlit 页面。默认地址一般为：

```text
http://localhost:8501
```

请保持启动窗口开启。

退出时可以：

```text
输入 q 后回车
```

或按：

```text
Ctrl + C
```

## 手动分开启动

排查问题时，可以打开三个终端窗口，分别执行：

```powershell
python server.py
```

```powershell
python worker.py
```

```powershell
python -m streamlit run app.py
```

这样更容易判断具体是哪一个服务报错。

---

# 十二、使用方法

## 方法一：油猴提交网页视频

1. 先运行 `python launcher.py`。
2. 打开 B 站或抖音网页。
3. 点击视频卡片或详情页上的 AI 按钮。
4. 等待任务进入队列。
5. 在网页左侧查看下载、转录和总结进度。

油猴任务会自动开启人声分离。

## 方法二：网页导入本地音视频

在 Streamlit 左侧打开：

```text
导入本地音视频
```

填写本地文件完整路径，例如：

```text
D:\Videos\test.mp4
```

可以自行选择是否开启：

```text
AI 深度降噪 / 人声分离
```

## 方法三：拖入启动黑框

把本地视频或音频文件直接拖进 `launcher.py` 的黑色窗口，按回车即可加入任务队列。

这种方式没有单独的复选框，人声分离是否启用由：

```python
DEFAULT_USE_VOCAL_SEPARATION
```

决定。

---

# 十三、数据保存位置

## 字幕、总结和视频档案

默认保存在：

```text
srt_vault/
```

每个 `.srt` 文件除了字幕，还会在尾部保存：

- 原视频地址
- 收藏夹信息
- 个人笔记
- AI 对话历史

如果开启原片归档，同名 MP4 通常会保存在字幕文件旁边。

## 任务数据库

```text
tasks.db
```

用于记录任务状态、进度和选项。

## 临时音频目录

由下面的配置决定：

```python
AUDIO_WORKSPACE_DIR = Path(r"D:\AI_Workspace")
```

程序正常完成后会删除大部分临时音频。程序异常中断时，可能残留 WAV 或媒体文件，可以在确认没有任务运行后手动清理。

---

# 十四、常见问题

## 1. B 站报 `HTTP Error 412`

错误示例：

```text
Unable to download JSON metadata: HTTP Error 412: Precondition Failed
```

优先检查：

1. 项目根目录是否存在 `cookies.txt`。
2. Cookie 是否为登录 B 站后新导出的 Netscape 格式。
3. Cookie 是否已经过期。
4. 是否安装了带 `curl-cffi` 的新版 `yt-dlp`。

重新安装或更新：

```powershell
python -m pip install --upgrade --pre "yt-dlp[default,curl-cffi]"
```

仍然失败时，可以在 `config.py` 中尝试：

```python
YTDLP_COOKIES_FROM_BROWSER = "edge"
```

或：

```python
YTDLP_COOKIES_FROM_BROWSER = "chrome"
```

## 2. 报 `Impersonate target "chrome" is not available`

说明缺少 `curl-cffi`：

```powershell
python -m pip install --upgrade --pre "yt-dlp[default,curl-cffi]"
```

## 3. 提示找不到 FFmpeg

```text
未找到 ffmpeg
```

执行：

```powershell
ffmpeg -version
```

如果命令不存在，说明 FFmpeg 没有安装或没有加入 `PATH`。

## 4. Whisper 模型加载失败

检查：

```python
WHISPER_MODEL_DIR
WHISPER_MODEL_NAME
WHISPER_LOCAL_FILES_ONLY
```

确认模型已经下载，并且缓存目录和 `config.py` 完全一致。

默认模型应为：

```text
Systran/faster-whisper-large-v3
```

## 5. 报 CUDA、cuBLAS 或 cuDNN 错误

检查：

```powershell
nvidia-smi
```

并重新核对 CUDA 12、cuDNN 9 和系统环境变量。

## 6. 人声分离失败后仍然继续识别

这是正常的容错行为。

当 `audio-separator`、CUDA、ONNX Runtime 或模型加载失败时，程序会输出类似：

```text
人声分离过程遭遇异常
降级退回基础 FFmpeg 音轨提取模式
```

之后会直接使用普通音轨继续 Whisper 转录。

## 7. 找不到 `Kim_Vocal_2.onnx`

确认：

```text
G:\WhisperModels\audio-separator-models\Kim_Vocal_2.onnx
```

或者确认 `config.py` 中 `VOCAL_MODEL_DIR` 指向实际目录。

首次启用人声分离时，需要能够联网下载模型；也可以手动复制模型到指定目录。

## 8. 大模型无法连接

错误通常类似：

```text
Connection refused
大模型服务异常
```

检查：

1. LM Studio 是否已经启动。
2. 是否已经加载模型。
3. Developer 页面中的 API Server 是否已启动。
4. 端口是否为 `1234`。
5. `LLM_BASE_URL` 是否为 `http://127.0.0.1:1234/v1`。

大模型失败不会删除已经生成的字幕。

## 9. 油猴按钮点击后提示失败

检查本地 API：

```text
http://127.0.0.1:8000
```

确认：

- `launcher.py` 正在运行
- `server.py` 没有报错
- 防火墙没有拦截本地 Python
- 油猴中的 `API_URL` 与 `config.py` 的端口一致

## 10. Streamlit 页面能打开，但任务不处理

检查启动窗口中是否出现：

```text
🚀 [Worker] 智能调度进程已启动，等待派单...
```

如果没有，单独运行：

```powershell
python worker.py
```

查看具体报错。

## 11. 下载到了字幕，但没有 AI 总结

说明 Whisper 已经成功，但大模型服务没有正常工作。

启动 LM Studio API 后，可以在网页中打开对应字幕，再使用快捷指令或手动提问。

## 12. 视频过长，后半段没有进入总结

代码会根据：

```python
MAX_CHARS_LIMIT
```

截断过长字幕，以避免大模型上下文溢出。

可以提高该值，但必须同时保证大模型支持足够长的上下文，并预留输出空间。

---

# 十五、更新 yt-dlp

视频网站接口经常变化。遇到网页视频突然无法下载时，建议先更新：

```powershell
python -m pip install --upgrade --pre "yt-dlp[default,curl-cffi]"
```

查看版本：

```powershell
python -c "import yt_dlp; print(yt_dlp.version.__version__)"
```

yt-dlp 官方项目：

- https://github.com/yt-dlp/yt-dlp


---

# 十六、隐私与版权说明

- 本项目默认在本地处理音频、字幕和视频文件。
- 如果使用在线大模型 API，字幕内容会发送给对应的 API 服务商。
- 请勿处理、下载或传播无权使用的内容。
- 网页视频下载能力仅用于个人学习、研究、备份和内容整理。
- 请遵守视频平台服务条款、著作权法律和当地法规。
- Cookie 属于敏感登录凭据，请妥善保管。

---

# ✅ 启动前检查清单

- [ ] Python 3.10 或 3.11 可以正常运行
- [ ] `python -m pip` 可以正常使用
- [ ] 已执行 `python -m pip install --upgrade -r requirements.txt`
- [ ] `yt-dlp[default,curl-cffi]` 已安装
- [ ] FFmpeg 已加入 `PATH`
- [ ] `nvidia-smi` 正常
- [ ] CUDA 12 和 cuDNN 9 已安装
- [ ] Faster-Whisper `large-v3` 已下载
- [ ] `WHISPER_MODEL_DIR` 已改为真实路径
- [ ] `AUDIO_WORKSPACE_DIR` 已改为真实路径
- [ ] 使用油猴时已安装 `audio-separator[gpu]`
- [ ] `Kim_Vocal_2.onnx` 已存在或首次运行时可以联网下载
- [ ] B 站用户已在项目根目录放入有效的 `cookies.txt`
- [ ] LM Studio 已加载模型
- [ ] LM Studio API Server 已在 1234 端口启动
- [ ] 油猴脚本已经安装并启用
- [ ] 已运行 `python launcher.py`

---

## 相关项目

- Faster-Whisper：<https://github.com/SYSTRAN/faster-whisper>
- Audio Separator：<https://github.com/nomadkaraoke/python-audio-separator>
- yt-dlp：<https://github.com/yt-dlp/yt-dlp>
- LM Studio 文档：<https://lmstudio.ai/docs/developer>
- Streamlit：<https://streamlit.io/>
- FastAPI：<https://fastapi.tiangolo.com/>

---

如果程序运行异常，请优先复制启动黑框中的完整错误信息，而不是只截图网页提示。后台日志通常会明确显示问题发生在下载、音频处理、Whisper 转录还是大模型总结阶段。可以将错误日志及程序源码发给AI进行故障排查。


---
后续升级计划：
- [ ] 更好的whisper模型
- [ ] 支持导出更多格式
- [ ] 支持文件夹
- [ ] 支持更多平台
- [ ] 本地模型思考模式切换
- [ ] 支持网页链接导入
- [ ] 整合包
