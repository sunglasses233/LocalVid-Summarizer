"""分享版统一启动器。"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from config import (
    BASE_DIR,
    STREAMLIT_HOST,
    STREAMLIT_MAX_UPLOAD_MB,
    STREAMLIT_PORT,
    TASK_API_URL,
)


REQUIRED_MODULES = (
    "fastapi",
    "pydantic",
    "uvicorn",
    "streamlit",
    "requests",
    "openai",
    "faster_whisper",
    "yt_dlp",
    "sherpa_onnx",
)
VOCAL_REQUIRED_MODULES = (*REQUIRED_MODULES, "audio_separator", "onnxruntime", "torch")
VALID_PROFILES = frozenset({"core", "vocal"})
MIN_YT_DLP_VERSION = (2026, 7, 4)
STREAMLIT_SERVICE_NAME = "Streamlit 界面"
STREAMLIT_STARTUP_TIMEOUT_SECONDS = 60
STREAMLIT_POLL_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class RuntimeSelection:
    profile: str
    executable: str


def configure_console() -> None:
    for output_stream in (sys.stdout, sys.stderr):
        if hasattr(output_stream, "reconfigure"):
            output_stream.reconfigure(errors="replace")


def read_active_profile(project_root: Path | str = BASE_DIR) -> str:
    active_path = Path(project_root) / "runtime" / "active-profile.txt"
    try:
        profile = active_path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "core"
    return profile if profile in VALID_PROFILES else "core"


def profile_python_path(profile: str, project_root: Path | str = BASE_DIR) -> Path:
    if profile not in VALID_PROFILES:
        raise ValueError(f"未知运行环境：{profile}")
    return Path(project_root) / "runtime" / "envs" / profile / "Scripts" / "python.exe"


def iter_runtime_candidates(project_root: Path | str = BASE_DIR):
    """只返回项目内的活动环境；vocal 不可用时允许回退到 core。"""

    active_profile = read_active_profile(project_root)
    profiles = [active_profile]
    if active_profile == "vocal":
        profiles.append("core")
    for profile in profiles:
        yield RuntimeSelection(profile, str(profile_python_path(profile, project_root)))


def interpreter_has_dependencies(
    executable: str,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    *,
    profile: str = "core",
) -> bool:
    modules = VOCAL_REQUIRED_MODULES if profile == "vocal" else REQUIRED_MODULES
    import_statement = "; ".join(f"import {module}" for module in modules)
    minimum_version = repr(MIN_YT_DLP_VERSION)
    runtime_probe = (
        f"{import_statement}; import re; "
        "from yt_dlp.version import __version__ as yt_dlp_version; "
        "parsed_yt_dlp_version = tuple(int(value) for value in "
        "re.findall(r'\\d+', yt_dlp_version)[:3]); "
        f"assert parsed_yt_dlp_version >= {minimum_version}"
    )
    runner = runner or subprocess.run
    try:
        result = runner(
            [executable, "-c", runtime_probe],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def select_runtime_environment(
    project_root: Path | str = BASE_DIR,
) -> RuntimeSelection:
    checked: list[RuntimeSelection] = []
    for selection in iter_runtime_candidates(project_root):
        checked.append(selection)
        if os.path.isfile(selection.executable) and interpreter_has_dependencies(
            selection.executable,
            profile=selection.profile,
        ):
            return selection

    checked_text = "\n".join(
        f"  - {item.profile}: {item.executable}" for item in checked
    )
    raise RuntimeError(
        "未找到可用的项目私有运行环境。\n"
        f"已检查：\n{checked_text}\n"
        f"最低 yt-dlp 版本：{'.'.join(str(value) for value in MIN_YT_DLP_VERSION)}\n"
        "请双击“安装.bat”完成安装或修复；分享版不会使用系统 Python。"
    )


def select_python_executable() -> str:
    """保留旧调用接口，实际选择范围严格限制在项目内。"""

    return select_runtime_environment().executable


def build_child_environment(
    source: dict[str, str] | None = None,
    *,
    selection: RuntimeSelection | None = None,
    project_root: Path | str = BASE_DIR,
) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    project_root = Path(project_root)
    runtime_root = project_root / "runtime"
    cuda_dir = project_root / "tools" / "cuda" / "bin"
    ffmpeg_dir = project_root / "tools" / "ffmpeg" / "bin"
    current_parts = [
        part for part in environment.get("PATH", "").split(os.pathsep) if part
    ]
    requested_parts = [str(cuda_dir), str(ffmpeg_dir), *current_parts]
    path_parts: list[str] = []
    seen: set[str] = set()
    for part in requested_parts:
        normalized = os.path.normcase(os.path.abspath(part))
        if normalized not in seen:
            seen.add(normalized)
            path_parts.append(part)

    environment.update(
        {
            "PATH": os.pathsep.join(path_parts),
            "TEMP": str(runtime_root / "tmp"),
            "TMP": str(runtime_root / "tmp"),
            "PIP_CACHE_DIR": str(runtime_root / "pip-cache"),
            "HF_HOME": str(runtime_root / "huggingface-cache"),
            "HUGGINGFACE_HUB_CACHE": str(runtime_root / "huggingface-cache" / "hub"),
            "TORCH_HOME": str(runtime_root / "torch-cache"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AI_VIDEO_CUDA_BIN_DIR": str(cuda_dir),
        }
    )
    if selection is not None:
        environment["AI_VIDEO_PROFILE"] = selection.profile
        environment["AI_VIDEO_PYTHON"] = selection.executable
    return environment


def run_runtime_check(
    python_executable: str,
    environment: dict[str, str],
    profile: str = "core",
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> bool:
    runner = runner or subprocess.run
    result = runner(
        [
            python_executable,
            str(BASE_DIR / "runtime_check.py"),
            "--profile",
            profile,
        ],
        cwd=str(BASE_DIR),
        env=environment,
        check=False,
    )
    return result.returncode == 0


def service_commands(python_executable: str) -> list[tuple[str, list[str]]]:
    return [
        (
            "FastAPI 调度中心",
            [python_executable, str(BASE_DIR / "server.py")],
        ),
        (
            "后台处理进程",
            [python_executable, str(BASE_DIR / "worker.py")],
        ),
        (
            STREAMLIT_SERVICE_NAME,
            [
                python_executable,
                "-m",
                "streamlit",
                "run",
                str(BASE_DIR / "app.py"),
                "--server.address",
                STREAMLIT_HOST,
                "--server.port",
                str(STREAMLIT_PORT),
                "--server.headless",
                "true",
                "--server.maxUploadSize",
                str(STREAMLIT_MAX_UPLOAD_MB),
                "--browser.gatherUsageStats",
                "false",
            ],
        ),
    ]


def cleanup_processes(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def start_services(
    python_executable: str,
    environment: dict[str, str],
    process_factory: Callable[..., subprocess.Popen] | None = None,
) -> tuple[list[subprocess.Popen], list[tuple[str, subprocess.Popen]]]:
    process_factory = process_factory or subprocess.Popen
    processes: list[subprocess.Popen] = []
    services: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, command in service_commands(python_executable):
            print(f"正在启动 [{name}]...")
            process = process_factory(
                command,
                cwd=str(BASE_DIR),
                env=environment,
            )
            processes.append(process)
            services.append((name, process))
    except Exception:
        cleanup_processes(processes)
        raise
    return processes, services


def detect_early_failures(
    services: Sequence[tuple[str, subprocess.Popen]], delay_seconds: float = 2
) -> list[str]:
    time.sleep(delay_seconds)
    return [
        f"{name}（退出码 {process.returncode}）"
        for name, process in services
        if process.poll() is not None
    ]


def wait_for_streamlit_ready(
    process: subprocess.Popen,
    *,
    host: str = STREAMLIT_HOST,
    port: int = STREAMLIT_PORT,
    timeout_seconds: float = STREAMLIT_STARTUP_TIMEOUT_SECONDS,
    poll_interval_seconds: float = STREAMLIT_POLL_INTERVAL_SECONDS,
    connector: Callable[..., object] | None = None,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> None:
    connector = connector or socket.create_connection
    clock = clock or time.monotonic
    sleeper = sleeper or time.sleep
    deadline = clock() + max(timeout_seconds, 0)
    last_error: OSError | None = None

    while True:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"{STREAMLIT_SERVICE_NAME}启动失败（退出码 {exit_code}）"
            )

        try:
            connection = connector((host, port), timeout=1)
        except OSError as error:
            last_error = error
        else:
            close = getattr(connection, "close", None)
            if callable(close):
                close()
            return

        if clock() >= deadline:
            detail = f"；最后一次连接错误：{last_error}" if last_error else ""
            raise TimeoutError(
                f"等待 {STREAMLIT_SERVICE_NAME} 超时：{host}:{port}{detail}"
            )
        sleeper(max(poll_interval_seconds, 0))


def submit_local_file(path: str) -> str:
    clean_path = path.strip().strip('"').strip("'")
    if not os.path.isfile(clean_path):
        return "读取失败：找不到该文件，请确认路径是否正确。"

    data = json.dumps(
        {
            "source_type": "local_file",
            "source_path": clean_path,
            "title": os.path.splitext(os.path.basename(clean_path))[0],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        TASK_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status == 200:
                return f"已提交：{os.path.basename(clean_path)}"
            return f"提交失败，服务器状态码：{response.status}"
    except Exception as error:
        return f"无法连接到本地调度中心：{error}"


def interactive_loop(input_func: Callable[[str], str] = input) -> None:
    print("系统已启动。可以把本地视频或音频文件拖入此窗口并按回车。")
    print("输入 q 可关闭全部服务。")
    while True:
        try:
            user_input = input_func("拖入文件并按回车（或输入 q 退出）：\n> ").strip()
        except EOFError:
            print("输入流已关闭，准备停止服务。")
            return
        if user_input.lower() in {"q", "exit", "quit"}:
            return
        if user_input:
            print(submit_local_file(user_input))


def print_header() -> None:
    print("=" * 50)
    print("AI 视频总结助手（分享版）")
    print("=" * 50)


def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    arguments = list(sys.argv[1:] if argv is None else argv)
    processes: list[subprocess.Popen] = []
    print_header()

    try:
        print("正在检查 Python 运行环境...")
        selection = select_runtime_environment()
        python_executable = selection.executable
        print(f"使用 Python：{python_executable}")
        print(f"使用环境：{selection.profile}")
        environment = build_child_environment(selection=selection)

        print("正在执行分享版运行环境自检...")
        if not run_runtime_check(
            python_executable,
            environment,
            profile=selection.profile,
        ):
            print("运行环境未通过，请根据上方提示完成安装或修复。")
            return 1

        if "--check-runtime" in arguments:
            print("运行环境检查通过。")
            return 0

        processes, services = start_services(python_executable, environment)
        failures = detect_early_failures(services)
        if failures:
            raise RuntimeError("以下服务启动失败：" + "；".join(failures))

        streamlit_process = next(
            (
                process
                for name, process in services
                if name == STREAMLIT_SERVICE_NAME
            ),
            None,
        )
        if streamlit_process is None:
            raise RuntimeError(f"未找到{STREAMLIT_SERVICE_NAME}进程")

        print("正在等待网页界面就绪...")
        wait_for_streamlit_ready(streamlit_process)
        web_url = f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
        print(f"网页地址：{web_url}")
        try:
            browser_opened = webbrowser.open(web_url)
        except Exception as error:
            browser_opened = False
            print(f"自动打开浏览器失败：{error}")
        if not browser_opened:
            print("浏览器未自动打开，请手动访问上方网页地址。")
        interactive_loop()
        return 0
    except KeyboardInterrupt:
        print("收到中断指令，准备关闭系统。")
        return 130
    except Exception as error:
        print(f"系统启动失败：{error}")
        return 1
    finally:
        if processes:
            print("正在关闭后台服务...")
            cleanup_processes(processes)
            print("后台服务已关闭。")


if __name__ == "__main__":
    raise SystemExit(main())
