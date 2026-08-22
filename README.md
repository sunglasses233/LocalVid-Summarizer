# AI 视频总结助手

一个面向 Windows + NVIDIA 显卡的本地视频处理工具，提供视频下载、Whisper GPU 转录、说话人识别、字幕语义校对、在线大语言模型总结和知识卡片管理。人声分离可作为可选组件安装。

当前为公开预览版。部署过程采用透明的半自动方式：脚本负责项目内 Python 环境和依赖，用户按清单自行准备模型、CUDA DLL 与 FFmpeg。

## 支持范围

- Windows 10/11 64 位；
- NVIDIA 显卡及可用的显卡驱动；
- CPython 3.10 64 位，推荐 3.10.11；
- 视频下载组件要求 `yt-dlp >= 2026.7.4`，当前依赖锁固定为 `2026.7.4`；
- 标准版包含 Whisper GPU 转录和说话人识别；
- 人声分离为可选组件；
- 总结功能使用用户自行配置的在线或 OpenAI 兼容 API，暂不包含本地大语言模型。

CPU 通用模式和其他操作系统尚未纳入当前版本的验证范围。

## 硬件要求

当前版本固定使用 `faster-whisper-large-v3 + CUDA + float16`。因此必须使用支持 CUDA float16 的 NVIDIA 显卡；AMD 显卡、Intel Arc 和纯 CPU 电脑不在当前支持范围内。CPU 通用模式属于后续方案。

| 项目 | 当前要求或最低线 | 建议配置 |
|---|---|---|
| 显卡 | NVIDIA 显卡、可用的官方驱动，并通过 CUDA float16 检查 | RTX 20 系列或更新型号 |
| 显存 | 至少 6000 MiB；低于该值时 `检查环境.bat` 会判定失败 | 标准版建议 8GB；经常使用人声分离建议 12GB 以上 |
| 系统内存 | 程序没有设置硬性下限 | 标准版建议 16GB；人声分离或长视频建议 32GB |
| CPU | Windows x86-64 处理器 | 建议 4 核 8 线程以上 |
| 可用磁盘空间 | 需要容纳 Python 环境、模型、CUDA DLL、FFmpeg 和运行文件 | 标准版至少预留 10GB；人声分离版至少预留 20GB |

6GB 显存属于最低线，运行时应关闭占用显存较多的游戏、剪辑软件或其他 AI 程序。GTX 10 系列及更早显卡不建议使用。显卡型号、驱动和 CUDA 能否实际配合，以 `检查环境.bat` 的 NVIDIA 显卡、CUDA float16 和 Whisper 模型检查结果为准。

## 安装

1. 下载并解压本仓库，建议放在空间充足的非系统盘。
2. 自行安装 Python 3.10 64 位，并确保 `py -3.10` 或 `python` 命令可用。
3. 双击 `安装.bat`，选择标准版或“标准版 + 人声分离”。
4. 按 [`docs/RESOURCE_DOWNLOADS.md`](docs/RESOURCE_DOWNLOADS.md) 下载并放置 FFmpeg、CUDA DLL、Whisper 和说话人识别模型；人声分离模型按需下载。
5. 双击 `检查环境.bat`，根据提示补齐缺少的资源。
6. 检查通过后双击 `启动.bat`，在界面中填写自己的大模型 API 配置。

`安装.bat` 只在 `runtime/envs` 下创建项目专属虚拟环境并安装锁定依赖，不会安装或修复系统 Python，也不会自动下载模型、CUDA 和 FFmpeg。

如果是覆盖更新旧版本，请在替换项目文件后重新运行一次 `安装.bat`。仅覆盖源码不会自动更新已经创建的虚拟环境；重新安装后可运行 `检查环境.bat` 确认 `yt-dlp` 版本和其余资源状态。

## 主要入口

- `安装.bat`：创建标准版/人声分离版 Python 环境；
- `检查环境.bat`：检查依赖、工具、模型和 NVIDIA 运行条件；
- `启动.bat`：统一启动 FastAPI、Worker 和 Streamlit；
- `油猴V7.txt`：浏览器端任务提交脚本；
- `requirements/*.lock`：Python 3.10 的完整锁定依赖；
- `manifests/artifacts.json`：外部工具和模型的版本、下载地址、大小及 SHA-256；
- `docs/RESOURCE_DOWNLOADS.md`：外部资源下载和目录说明。

## 模型整理下载链接
百度网盘链接: https://pan.baidu.com/s/1oBaB4r3Y6KaGjX9i7m9COw?pwd=pjhy 提取码: pjhy

## 许可证

本项目自有源代码使用 [MIT License](LICENSE)。Python 依赖、模型、FFmpeg 和 CUDA/cuDNN 等第三方组件适用各自许可证；其版本、来源和许可信息记录在 `manifests/artifacts.json`。MIT License 不会替代第三方组件的许可证。

## 已知限制

- 首次部署需要下载数 GB 的 Whisper 模型及 CUDA 运行文件；
- 显卡驱动版本和本机安全软件可能影响 DLL 加载；
- 当前发布版尚需在更多非开发机环境持续验证；
- 哔哩哔哩可能因访问频率、登录状态或风控返回 HTTP 412。程序会自动附加站点 Referer；如果仍然失败，请先重新运行 `安装.bat`，再尝试更新 `data/cookies.txt` 中由已登录浏览器导出的 Cookie，或稍后重试；
- 人声分离环境体积明显大于标准版，安装失败时可继续使用标准版功能。
