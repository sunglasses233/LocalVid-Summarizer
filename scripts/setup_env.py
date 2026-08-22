"""为分享版创建项目内 Python 环境并安装锁定依赖。

该脚本只处理 Python 环境，不下载模型、CUDA 或 FFmpeg。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PYTHON = (3, 10)
PIP_VERSION = "26.2.1"
CHINA_PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
OFFICIAL_PYPI_INDEX = "https://pypi.org/simple"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
VALID_PROFILES = frozenset({"core", "vocal"})


class SetupError(RuntimeError):
    """环境准备无法继续。"""


def check_base_python(
    version_info: Sequence[int] | None = None,
    maxsize: int | None = None,
) -> None:
    version_info = sys.version_info if version_info is None else version_info
    maxsize = sys.maxsize if maxsize is None else maxsize
    detected = tuple(version_info[:3])
    if detected[:2] != SUPPORTED_PYTHON or maxsize <= 2**32:
        version = ".".join(map(str, detected))
        raise SetupError(
            f"需要 Python 3.10 64 位，当前为 Python {version}；"
            "推荐使用已经验证的 3.10.11。"
        )


def environment_python(
    profile: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if profile not in VALID_PROFILES:
        raise SetupError(f"未知环境档位：{profile}")
    return project_root / "runtime" / "envs" / profile / "Scripts" / "python.exe"


def child_environment(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    runtime_root = project_root / "runtime"
    temp_dir = runtime_root / "tmp"
    pip_cache = runtime_root / "pip-cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
            "PIP_CACHE_DIR": str(pip_cache),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def query_python_version(
    python_path: Path,
    *,
    environment: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[int, int, int, int] | None:
    try:
        completed = runner(
            [
                str(python_path),
                "-c",
                (
                    "import struct, sys; "
                    "print(*sys.version_info[:3], struct.calcsize('P') * 8)"
                ),
            ],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        major, minor, patch, bits = map(int, completed.stdout.strip().split())
    except (TypeError, ValueError):
        return None
    return major, minor, patch, bits


def ensure_environment(
    profile: str,
    *,
    project_root: Path = PROJECT_ROOT,
    environment: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    python_path = environment_python(profile, project_root)
    environment_dir = python_path.parents[1]
    if not python_path.is_file():
        if environment_dir.exists():
            raise SetupError(
                f"环境目录存在但缺少 python.exe：{environment_dir}\n"
                "请确认其中没有需要保留的内容，再手动删除该目录并重试。"
            )
        environment_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[环境] 创建 {profile}：{environment_dir}")
        runner(
            [sys.executable, "-m", "venv", str(environment_dir)],
            env=environment,
            check=True,
        )

    detected = query_python_version(
        python_path,
        environment=environment,
        runner=runner,
    )
    if detected is None or detected[:2] != SUPPORTED_PYTHON or detected[3] != 64:
        raise SetupError(
            f"现有 {profile} 环境不是 Python 3.10 64 位：{environment_dir}\n"
            "请手动删除该环境目录，再用 Python 3.10 重新运行安装入口。"
        )
    return python_path


def build_pip_command(
    python_path: Path | str,
    arguments: Sequence[str],
    *,
    index_url: str,
    profile: str,
) -> list[str]:
    command = [
        str(python_path),
        "-m",
        "pip",
        *arguments,
        "--index-url",
        index_url,
        "--no-cache-dir",
        "--disable-pip-version-check",
    ]
    if profile == "vocal":
        command.extend(["--extra-index-url", PYTORCH_CUDA_INDEX])
    return command


def run_pip_with_fallback(
    python_path: Path,
    arguments: Sequence[str],
    *,
    profile: str,
    environment: dict[str, str],
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for index_url in (CHINA_PYPI_INDEX, OFFICIAL_PYPI_INDEX):
        print(f"[pip] 使用索引：{index_url}")
        try:
            runner(
                build_pip_command(
                    python_path,
                    arguments,
                    index_url=index_url,
                    profile=profile,
                ),
                env=environment,
                check=True,
            )
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            print(f"[pip] 当前索引失败，准备回退：退出码 {error.returncode}")
    raise SetupError(f"pip 在国内镜像和官方源均安装失败：{last_error}")


def activate_profile(profile: str, project_root: Path = PROJECT_ROOT) -> None:
    active_path = project_root / "runtime" / "active-profile.txt"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = active_path.with_name(active_path.name + ".tmp")
    temporary.write_text(profile + "\n", encoding="utf-8")
    os.replace(temporary, active_path)


def setup_profile(profile: str, project_root: Path = PROJECT_ROOT) -> None:
    check_base_python()
    lock_path = project_root / "requirements" / f"{profile}.lock"
    if not lock_path.is_file():
        raise SetupError(f"缺少依赖锁定文件：{lock_path}")

    environment = child_environment(project_root)
    python_path = ensure_environment(
        profile,
        project_root=project_root,
        environment=environment,
    )
    run_pip_with_fallback(
        python_path,
        ["install", "--upgrade", f"pip=={PIP_VERSION}"],
        profile="core",
        environment=environment,
    )
    run_pip_with_fallback(
        python_path,
        ["install", "--requirement", str(lock_path)],
        profile=profile,
        environment=environment,
    )
    subprocess.run(
        [str(python_path), "-m", "pip", "check"],
        env=environment,
        check=True,
    )
    activate_profile(profile, project_root)
    print(f"[完成] {profile} Python 依赖已经安装。")
    print("下一步请按 docs/RESOURCE_DOWNLOADS.md 放置外部资源，然后运行环境检查。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备分享版 Python 依赖环境")
    parser.add_argument("--profile", choices=("core", "vocal"), default="core")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        setup_profile(args.profile)
    except (SetupError, OSError, subprocess.SubprocessError) as error:
        print(f"环境准备失败：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
