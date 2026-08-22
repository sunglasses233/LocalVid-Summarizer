"""分享版运行环境自检。

第一版只支持 Windows + NVIDIA GPU。该模块只做只读诊断和临时目录
写入测试，不负责安装依赖或下载模型；安装行为由后续安装器实现。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import artifact_manifest
from cuda_runtime import preload_bundled_cuda
from config import (
    API_HOST,
    API_PORT,
    AUDIO_WORKSPACE_DIR,
    FFMPEG_EXECUTABLE,
    SPEAKER_MODEL_DIR,
    STREAMLIT_HOST,
    STREAMLIT_PORT,
    TOOLS_DIR,
    VOCAL_MODEL_DIR,
    WHISPER_MODEL_DIR,
    WHISPER_MODEL_REPO_ID,
)


MIN_NVIDIA_MEMORY_MB = 6000
MIN_RUNTIME_FREE_BYTES = 2 * 1024**3
VERIFIED_PYTHON_VERSION = (3, 10, 11)
MIN_YT_DLP_VERSION = (2026, 7, 4)
CUDA_BIN_DIR = TOOLS_DIR / "cuda" / "bin"
CUDA_REQUIRED_FILES = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
)
WHISPER_REQUIRED_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
)
SPEAKER_REQUIRED_FILES = (
    "segmentation.onnx",
    "speaker_embedding.onnx",
)
CORE_MODULES = {
    "fastapi": "FastAPI",
    "pydantic": "Pydantic",
    "uvicorn": "Uvicorn",
    "streamlit": "Streamlit",
    "requests": "Requests",
    "openai": "OpenAI SDK",
    "faster_whisper": "faster-whisper",
    "yt_dlp": "yt-dlp",
    "sherpa_onnx": "sherpa-onnx",
}
VOCAL_MODULES = {
    "audio_separator": "audio-separator",
    "onnxruntime": "ONNX Runtime",
    "torch": "PyTorch",
}
VALID_PROFILES = frozenset({"core", "vocal"})


@dataclass(frozen=True)
class CheckResult:
    key: str
    label: str
    ok: bool
    message: str
    required: bool = True


def results_pass(results: Iterable[CheckResult]) -> bool:
    return all(result.ok or not result.required for result in results)


def check_platform() -> CheckResult:
    ok = sys.platform == "win32"
    return CheckResult(
        "platform",
        "Windows 系统",
        ok,
        "已检测到 Windows" if ok else f"当前系统为 {sys.platform}，第一版仅支持 Windows",
    )


def check_python_runtime(
    version_info: Sequence[int] | None = None,
    maxsize: int | None = None,
) -> CheckResult:
    version_info = sys.version_info if version_info is None else version_info
    maxsize = sys.maxsize if maxsize is None else maxsize
    version = tuple(version_info[:3])
    is_python_310 = version[:2] == VERIFIED_PYTHON_VERSION[:2]
    is_64bit = maxsize > 2**32
    ok = is_python_310 and is_64bit

    detected = ".".join(str(part) for part in version)
    if not ok:
        problems = []
        if not is_python_310:
            problems.append("需要 Python 3.10")
        if not is_64bit:
            problems.append("需要 64 位 Python")
        message = f"检测到 Python {detected}；" + "、".join(problems)
    elif version != VERIFIED_PYTHON_VERSION:
        verified = ".".join(str(part) for part in VERIFIED_PYTHON_VERSION)
        message = f"Python {detected} 64 位可用；完整验证版本为 {verified}"
    else:
        message = f"Python {detected} 64 位，与锁定环境一致"
    return CheckResult("python_runtime", "Python 3.10（64 位）", ok, message)


def check_yt_dlp_version(version: str | None = None) -> CheckResult:
    if version is None:
        try:
            from yt_dlp.version import __version__ as version
        except Exception as error:
            return CheckResult(
                "yt_dlp_version",
                "yt-dlp 版本",
                False,
                f"无法读取 yt-dlp 版本：{error}",
            )

    detected = str(version or "").strip()
    parsed = tuple(int(value) for value in re.findall(r"\d+", detected)[:3])
    required_text = ".".join(str(value) for value in MIN_YT_DLP_VERSION)
    if len(parsed) != 3:
        return CheckResult(
            "yt_dlp_version",
            "yt-dlp 版本",
            False,
            f"无法识别版本“{detected}”；最低要求为 {required_text}",
        )

    ok = parsed >= MIN_YT_DLP_VERSION
    message = (
        f"yt-dlp {detected}，满足最低要求 {required_text}"
        if ok
        else f"yt-dlp {detected} 过低；请重新运行安装.bat，最低要求为 {required_text}"
    )
    return CheckResult("yt_dlp_version", "yt-dlp 版本", ok, message)


def check_core_modules(
    importer: Callable[[str], object] | None = None,
    profile: str = "core",
) -> CheckResult:
    importer = importer or importlib.import_module
    modules = dict(CORE_MODULES)
    if profile == "vocal":
        modules.update(VOCAL_MODULES)
    failures = []
    for module_name, label in modules.items():
        try:
            importer(module_name)
        except Exception as error:  # 原生 DLL 缺失也必须计入失败
            failures.append(f"{label}: {error}")

    if failures:
        return CheckResult(
            "python_modules",
            "Python 核心依赖",
            False,
            "缺失或无法加载：" + "；".join(failures),
        )
    return CheckResult(
        "python_modules", "Python 核心依赖", True, "核心依赖均可导入"
    )


def find_nvidia_smi() -> Path | None:
    located = shutil.which("nvidia-smi")
    if located:
        return Path(located)

    windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidate = windows_dir / "System32" / "nvidia-smi.exe"
    return candidate if candidate.is_file() else None


def check_nvidia_gpu(
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> CheckResult:
    executable = find_nvidia_smi()
    if not executable:
        return CheckResult(
            "nvidia_gpu",
            "NVIDIA 显卡",
            False,
            "未找到 nvidia-smi，请安装或更新 NVIDIA 官方驱动",
        )

    runner = runner or subprocess.run
    command = [
        str(executable),
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return CheckResult("nvidia_gpu", "NVIDIA 显卡", False, f"检测失败：{error}")

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        return CheckResult("nvidia_gpu", "NVIDIA 显卡", False, detail)

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_mb = int(float(parts[1]))
        except ValueError:
            continue
        gpus.append((parts[0], memory_mb, parts[2]))

    if not gpus:
        return CheckResult(
            "nvidia_gpu", "NVIDIA 显卡", False, "nvidia-smi 未返回可识别的显卡信息"
        )

    selected = max(gpus, key=lambda item: item[1])
    name, memory_mb, driver = selected
    ok = memory_mb >= MIN_NVIDIA_MEMORY_MB
    message = f"{name}，显存 {memory_mb} MiB，驱动 {driver}"
    if not ok:
        message += f"；large-v3 float16 建议至少 {MIN_NVIDIA_MEMORY_MB} MiB 显存"
    return CheckResult("nvidia_gpu", "NVIDIA 显卡", ok, message)


def check_cuda_float16() -> CheckResult:
    try:
        preload_bundled_cuda()
        import ctranslate2

        compute_types = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as error:
        return CheckResult(
            "cuda_float16",
            "CUDA float16",
            False,
            f"CTranslate2 无法使用 CUDA：{error}",
        )

    ok = "float16" in compute_types
    available = ", ".join(sorted(compute_types)) or "无"
    return CheckResult(
        "cuda_float16",
        "CUDA float16",
        ok,
        f"可用计算类型：{available}"
        if ok
        else f"不支持 float16；可用计算类型：{available}",
    )


def check_bundled_cuda_runtime(cuda_bin_dir: Path | None = None) -> CheckResult:
    cuda_bin_dir = Path(cuda_bin_dir or CUDA_BIN_DIR)
    missing = [
        filename
        for filename in CUDA_REQUIRED_FILES
        if not (cuda_bin_dir / filename).is_file()
        or (cuda_bin_dir / filename).stat().st_size <= 0
    ]
    if missing:
        preview = "、".join(missing[:3])
        if len(missing) > 3:
            preview += f" 等 {len(missing)} 个文件"
        return CheckResult(
            "bundled_cuda",
            "随包 CUDA 运行库",
            False,
            f"{cuda_bin_dir} 缺失：{preview}",
        )
    return CheckResult(
        "bundled_cuda",
        "随包 CUDA 运行库",
        True,
        f"10 个 CUDA 12 / cuDNN 9 DLL 均存在：{cuda_bin_dir}",
    )


def find_ffmpeg() -> Path | None:
    if FFMPEG_EXECUTABLE.is_file():
        return FFMPEG_EXECUTABLE
    located = shutil.which("ffmpeg")
    return Path(located) if located else None


def check_ffmpeg(
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> CheckResult:
    executable = find_ffmpeg()
    if not executable:
        return CheckResult(
            "ffmpeg",
            "FFmpeg",
            False,
            "未找到随包 FFmpeg，也未在系统 PATH 中找到 ffmpeg",
        )

    runner = runner or subprocess.run
    try:
        completed = runner(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return CheckResult("ffmpeg", "FFmpeg", False, f"执行失败：{error}")

    first_line = (completed.stdout or completed.stderr or "").splitlines()
    detail = first_line[0].strip() if first_line else str(executable)
    return CheckResult(
        "ffmpeg",
        "FFmpeg",
        completed.returncode == 0,
        detail if completed.returncode == 0 else f"无法执行：{detail}",
    )


def find_whisper_snapshot(model_root: Path | None = None) -> Path | None:
    model_root = Path(model_root or WHISPER_MODEL_DIR)
    cache_name = "models--" + WHISPER_MODEL_REPO_ID.replace("/", "--")
    snapshots_dir = model_root / cache_name / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
        if snapshot.is_dir() and all(
            (snapshot / filename).is_file() and (snapshot / filename).stat().st_size > 0
            for filename in WHISPER_REQUIRED_FILES
        ):
            return snapshot
    return None


def check_whisper_model() -> CheckResult:
    snapshot = find_whisper_snapshot()
    if not snapshot:
        return CheckResult(
            "whisper_model",
            "Whisper large-v3 模型",
            False,
            f"未在 {WHISPER_MODEL_DIR} 找到完整模型；请先运行后续安装器",
        )
    return CheckResult(
        "whisper_model", "Whisper large-v3 模型", True, f"模型目录：{snapshot}"
    )


def check_speaker_models() -> CheckResult:
    missing = [
        filename
        for filename in SPEAKER_REQUIRED_FILES
        if not (SPEAKER_MODEL_DIR / filename).is_file()
        or (SPEAKER_MODEL_DIR / filename).stat().st_size <= 0
    ]
    if missing:
        return CheckResult(
            "speaker_models",
            "说话人识别模型",
            False,
            "缺失或为空：" + "、".join(missing),
        )
    return CheckResult(
        "speaker_models", "说话人识别模型", True, "两个标准 ONNX 模型均存在"
    )


def check_artifact_integrity(
    full_hash: bool = False,
    include_optional: bool = False,
) -> CheckResult:
    try:
        manifest = artifact_manifest.load_manifest()
        results = artifact_manifest.verify_artifacts(
            manifest,
            root=Path(__file__).resolve().parent,
            scopes=("installed",),
            include_optional=include_optional,
            full_hash=full_hash,
        )
    except (artifact_manifest.ManifestError, OSError) as error:
        return CheckResult(
            "artifact_integrity",
            "资源清单",
            False,
            f"无法完成校验：{error}",
        )

    failures = [result for result in results if not result.ok]
    if failures:
        preview = "；".join(
            f"{result.key}({result.status})" for result in failures[:3]
        )
        if len(failures) > 3:
            preview += f"；另有 {len(failures) - 3} 项"
        return CheckResult(
            "artifact_integrity",
            "资源清单",
            False,
            preview,
        )

    mode = "SHA-256 完整校验" if full_hash else "文件大小快速校验"
    return CheckResult(
        "artifact_integrity",
        "含可选资源的完整清单" if include_optional else "标准资源清单",
        bool(results),
        f"{mode}通过，共 {len(results)} 个安装文件",
    )


def check_vocal_separation(required: bool = False) -> CheckResult:
    module_ready = importlib.util.find_spec("audio_separator") is not None
    model_path = VOCAL_MODEL_DIR / "Kim_Vocal_2.onnx"
    model_ready = model_path.is_file() and model_path.stat().st_size > 0
    ok = module_ready and model_ready
    if ok:
        message = "可选人声分离组件已安装"
    else:
        missing = []
        if not module_ready:
            missing.append("audio-separator 依赖")
        if not model_ready:
            missing.append("Kim_Vocal_2.onnx")
        message = "可选组件未启用，缺少：" + "、".join(missing)
    return CheckResult(
        "vocal_separation",
        "人声分离" if required else "人声分离（可选）",
        ok,
        message,
        required=required,
    )


def check_workspace_writable() -> CheckResult:
    try:
        AUDIO_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=AUDIO_WORKSPACE_DIR, prefix="write_test_"):
            pass
    except OSError as error:
        return CheckResult(
            "workspace", "临时工作目录", False, f"不可写：{AUDIO_WORKSPACE_DIR}；{error}"
        )
    return CheckResult(
        "workspace", "临时工作目录", True, f"可写：{AUDIO_WORKSPACE_DIR}"
    )


def check_disk_space() -> CheckResult:
    try:
        free_bytes = shutil.disk_usage(AUDIO_WORKSPACE_DIR).free
    except OSError as error:
        return CheckResult("disk_space", "运行磁盘空间", False, f"检测失败：{error}")

    free_gib = free_bytes / 1024**3
    ok = free_bytes >= MIN_RUNTIME_FREE_BYTES
    return CheckResult(
        "disk_space",
        "运行磁盘空间",
        ok,
        f"可用 {free_gib:.1f} GiB"
        if ok
        else f"仅剩 {free_gib:.1f} GiB，至少需要 2 GiB 临时空间",
    )


def check_port_available(host: str, port: int, key: str, label: str) -> CheckResult:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, port))
    except OSError as error:
        return CheckResult(key, label, False, f"{host}:{port} 已占用或不可绑定：{error}")
    finally:
        sock.close()
    return CheckResult(key, label, True, f"{host}:{port} 可用")


def run_all_checks(
    full_artifacts: bool = False,
    profile: str = "core",
) -> list[CheckResult]:
    if profile not in VALID_PROFILES:
        raise ValueError(f"未知运行环境：{profile}")
    vocal_enabled = profile == "vocal"
    return [
        check_platform(),
        check_python_runtime(),
        check_core_modules(profile=profile),
        check_yt_dlp_version(),
        check_nvidia_gpu(),
        check_bundled_cuda_runtime(),
        check_cuda_float16(),
        check_ffmpeg(),
        check_whisper_model(),
        check_speaker_models(),
        check_artifact_integrity(
            full_hash=full_artifacts,
            include_optional=vocal_enabled,
        ),
        check_workspace_writable(),
        check_disk_space(),
        check_port_available(API_HOST, API_PORT, "api_port", "FastAPI 端口"),
        check_port_available(
            STREAMLIT_HOST, STREAMLIT_PORT, "streamlit_port", "Streamlit 端口"
        ),
        check_vocal_separation(required=vocal_enabled),
    ]


def render_text(results: Sequence[CheckResult]) -> str:
    lines = []
    for result in results:
        if result.ok:
            marker = "通过"
        elif result.required:
            marker = "失败"
        else:
            marker = "可选"
        lines.append(f"[{marker}] {result.label}：{result.message}")
    lines.append("运行环境检查通过" if results_pass(results) else "运行环境检查未通过")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI 视频总结助手分享版运行环境自检")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument(
        "--full-artifacts",
        action="store_true",
        help="对标准资源计算 SHA-256；Whisper 大模型会使检查明显变慢",
    )
    parser.add_argument(
        "--profile",
        choices=("core", "vocal"),
        default=os.environ.get("AI_VIDEO_PROFILE", "core"),
        help="按标准版或人声分离增强版检查运行环境",
    )
    args = parser.parse_args(argv)

    results = run_all_checks(
        full_artifacts=args.full_artifacts,
        profile=args.profile,
    )
    if args.json:
        payload = {
            "ok": results_pass(results),
            "checks": [asdict(result) for result in results],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(results))
    return 0 if results_pass(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
