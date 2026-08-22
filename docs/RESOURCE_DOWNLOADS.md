# 外部资源下载与放置清单

本文档面向首次部署公开发布版的用户。仓库不包含大模型、说话人模型、CUDA 运行库和 FFmpeg，也不会自动下载这些资源；所有版本、文件大小和 SHA-256 的唯一权威来源是 `manifests/artifacts.json`，本文只说明下载顺序和目录结构，避免两处维护哈希值。

## 1. 部署前提

- Windows 10/11 64 位；
- NVIDIA 显卡及可用的较新驱动；
- CPython 3.10 64 位，推荐 3.10.11；安装 Python 时建议勾选“Add Python to PATH”；
- 标准版建议至少预留 10 GB，可选人声分离版建议至少预留 20 GB；
- 项目最好放在空间充足的非系统盘，路径可以包含中文，但不要放在需要管理员权限的目录。

## 2. 总体顺序

1. 解压分享包。
2. 双击 `安装.bat`，选择标准版或人声分离版。该步骤只创建项目内虚拟环境并安装 Python 依赖。
3. 按下文下载并放置 FFmpeg、CUDA DLL 和 Whisper 模型。
4. 下载标准版必需的两个说话人识别模型。
5. 如果安装了人声分离版，再下载对应的 4 个可选文件。
6. 双击 `检查环境.bat`；通过后双击 `启动.bat`。

## 3. 标准版必需资源

下文中的目录如果尚不存在，请自行创建；程序首次导入配置时也会自动建立基础目录。

### 3.1 FFmpeg

- 版本：`9.0.1-essentials_build`
- 下载地址：<https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.1-essentials_build.zip>
- 建议先保存为：`runtime/downloads/ffmpeg-9.0.1-essentials_build.zip`
- 解压后整理到：`tools/ffmpeg`
- 最终必须存在：`tools/ffmpeg/bin/ffmpeg.exe`

压缩包本身带有一层版本目录。复制的是该版本目录里面的内容，不要让最终路径多出一层版本号。

### 3.2 CUDA 12 运行 DLL

- 固定资源版本：`CUDA12_v3-cuBLAS12.8.4.1-cuDNN9.8.0.87`
- 官方下载地址：<https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA12_win_v3.7z>
- 网盘备用地址：`【请分享者填写】`
- 提取码：`【请分享者填写；没有则写“无”】`
- 压缩包文件名：`cuBLAS.and.cuDNN_CUDA12_win_v3.7z`

是否提供网盘备用地址由分享者自行确认相关许可；不能确认时请留空，只保留官方地址。

压缩包可以下载到任意位置。请使用电脑上已有且支持 7z/BCJ2 的解压软件自行解压；项目不提供、不安装也不检查解压工具。

从解压结果中找到并复制以下 10 个文件到 `tools/cuda/bin`：

- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudnn64_9.dll`
- `cudnn_adv64_9.dll`
- `cudnn_cnn64_9.dll`
- `cudnn_engines_precompiled64_9.dll`
- `cudnn_engines_runtime_compiled64_9.dll`
- `cudnn_graph64_9.dll`
- `cudnn_heuristic64_9.dll`
- `cudnn_ops64_9.dll`

只放入清单中的 DLL，不要混入电脑上其他 CUDA Toolkit 的文件。程序启动 Whisper 时会按绝对路径预加载这些 DLL，避免不同 CUDA 安装互相干扰。

### 3.3 Whisper large-v3

固定模型提交：`edaa852ec7e145841d8ffdb056a99866b5f0a478`

目标目录：

    models/whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478/

将以下 5 个文件下载到目标目录。国内网络可先尝试 `hf-mirror.com`，失败后改用同一行的 Hugging Face 官方地址。

| 文件 | 国内镜像 | 官方地址 |
|---|---|---|
| `config.json` | <https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/config.json> | <https://huggingface.co/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/config.json> |
| `model.bin` | <https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/model.bin> | <https://huggingface.co/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/model.bin> |
| `preprocessor_config.json` | <https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/preprocessor_config.json> | <https://huggingface.co/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/preprocessor_config.json> |
| `tokenizer.json` | <https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/tokenizer.json> | <https://huggingface.co/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/tokenizer.json> |
| `vocabulary.json` | <https://hf-mirror.com/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/vocabulary.json> | <https://huggingface.co/Systran/faster-whisper-large-v3/resolve/edaa852ec7e145841d8ffdb056a99866b5f0a478/vocabulary.json> |

另外创建文本文件：

    models/whisper/models--Systran--faster-whisper-large-v3/refs/main

文件内容只有一行：

    edaa852ec7e145841d8ffdb056a99866b5f0a478

请确认 Windows 没有把它实际保存成 `main.txt`。

## 4. 说话人识别标准资源

说话人识别属于标准版，但模型不进入 GitHub 仓库。请下载并保存为以下两个目标文件：

| 目标文件 | 国内镜像 | 官方地址 |
|---|---|---|
| `models/speaker_diarization/segmentation.onnx` | <https://hf-mirror.com/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx> | <https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx> |
| `models/speaker_diarization/speaker_embedding.onnx` | <https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx> | <https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx> |

下载源文件名通常都是 `model.onnx`，保存时必须按表格改成两个不同的目标文件名，不能互相覆盖。

## 5. 人声分离可选资源

只有在 `安装.bat` 中选择人声分离版时才需要以下文件。全部保存到 `models/audio_separator`；目录不存在时先创建。

| 目标文件 | 下载地址 |
|---|---|
| `download_checks.json` | <https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/filelists/download_checks.json> |
| `vr_model_data.json` | <https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/vr_model_data/model_data_new.json> |
| `mdx_model_data.json` | <https://raw.githubusercontent.com/TRvlvr/application_data/3826b05b570dbd4fbedbc807758803b35348ba1b/mdx_model_data/model_data_new.json> |
| `Kim_Vocal_2.onnx` | <https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx> |

前三个元数据文件也要下载，它们用于固定模型参数版本，避免程序运行时访问上游 `main` 分支。

## 6. 检查与启动

先双击 `检查环境.bat`。它检查当前安装档位所需的环境、模型和工具，不会自动修改系统。

如需逐文件计算 SHA-256，在项目根目录运行下列命令之一：

标准版：

    .\runtime\envs\core\Scripts\python.exe .\artifact_manifest.py --scope installed --full

人声分离版：

    .\runtime\envs\vocal\Scripts\python.exe .\artifact_manifest.py --scope installed --include-optional --full

全部通过后，双击 `启动.bat`。首次使用还需要在界面中填写自己的大模型 API 配置；公开仓库不包含开发者的 API Key、Cookie、数据库或个人任务数据。

已安装文件校验通过后，下载的压缩包和临时解压目录可以手动删除，以释放磁盘空间。
