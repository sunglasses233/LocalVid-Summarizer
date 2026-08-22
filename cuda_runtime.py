"""Load the project-pinned CUDA runtime before importing GPU frameworks."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CUDA_DLL_NAMES = (
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
)
LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008


class CudaRuntimeError(RuntimeError):
    """The pinned CUDA runtime cannot be loaded safely."""


_load_lock = threading.Lock()
_loaded_directory: Path | None = None
_loaded_paths: tuple[Path, ...] = ()
_loaded_handles: tuple[object, ...] = ()
_dll_directory_handle: object | None = None


def resolve_cuda_bin_dir() -> Path:
    raw_value = os.environ.get("AI_VIDEO_CUDA_BIN_DIR", "").strip().strip('"').strip("'")
    if raw_value:
        expanded = os.path.expandvars(os.path.expanduser(raw_value))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
    else:
        candidate = BASE_DIR / "tools" / "cuda" / "bin"
    return candidate.resolve(strict=False)


def validate_cuda_files(cuda_bin_dir: Path | str | None = None) -> tuple[Path, ...]:
    directory = Path(cuda_bin_dir or resolve_cuda_bin_dir()).resolve(strict=False)
    files = tuple(directory / name for name in CUDA_DLL_NAMES)
    missing = [path.name for path in files if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        preview = "、".join(missing[:4])
        if len(missing) > 4:
            preview += f" 等 {len(missing)} 个文件"
        raise CudaRuntimeError(f"随包 CUDA 运行库不完整：{directory} 缺少 {preview}")
    return files


def loaded_cuda_paths() -> tuple[Path, ...]:
    """Return paths retained by the current process, mainly for diagnostics."""

    return _loaded_paths


def preload_bundled_cuda(
    cuda_bin_dir: Path | str | None = None,
    *,
    required: bool = True,
) -> tuple[Path, ...]:
    """Load every pinned CUDA DLL by absolute path and retain all handles.

    Merely prepending PATH is insufficient on Windows: an existing CUDA Toolkit
    or a package-local DLL can still win the loader search order.  Absolute
    preloading makes the selected runtime deterministic for this process.
    """

    if sys.platform != "win32":
        if required:
            raise CudaRuntimeError(
                f"随包 CUDA 运行库只支持 Windows，当前平台为 {sys.platform}"
            )
        return ()

    directory = Path(cuda_bin_dir or resolve_cuda_bin_dir()).resolve(strict=False)
    if not required and not directory.is_dir():
        return ()
    files = validate_cuda_files(directory)

    global _loaded_directory
    global _loaded_paths
    global _loaded_handles
    global _dll_directory_handle

    with _load_lock:
        if _loaded_handles:
            if _loaded_directory != directory:
                raise CudaRuntimeError(
                    "当前进程已经加载另一套 CUDA 运行库："
                    f"{_loaded_directory}；拒绝切换到 {directory}"
                )
            return _loaded_paths

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is None:
            raise CudaRuntimeError("当前 Python 不支持 os.add_dll_directory")

        directory_handle = None
        handles: list[object] = []
        try:
            directory_handle = add_dll_directory(str(directory))
            for path in files:
                try:
                    handle = ctypes.WinDLL(
                        str(path),
                        winmode=LOAD_WITH_ALTERED_SEARCH_PATH,
                    )
                except OSError as error:
                    raise CudaRuntimeError(
                        f"无法加载随包 CUDA 文件 {path.name}：{error}"
                    ) from error
                handles.append(handle)
        except Exception:
            if directory_handle is not None and hasattr(directory_handle, "close"):
                directory_handle.close()
            raise

        _loaded_directory = directory
        _loaded_paths = files
        _loaded_handles = tuple(handles)
        _dll_directory_handle = directory_handle
        return _loaded_paths
