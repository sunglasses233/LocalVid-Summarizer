from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = BASE_DIR / "manifests" / "artifacts.json"
VALID_SCOPES = frozenset({"download", "installed"})
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """资源清单格式无效。"""


@dataclass(frozen=True)
class ArtifactFile:
    artifact_id: str
    file_id: str
    required: bool
    scope: str
    relative_path: str
    size: int
    sha256: str
    urls: tuple[str, ...]
    derived_from: str | None

    @property
    def key(self) -> str:
        return f"{self.artifact_id}/{self.file_id}"


@dataclass(frozen=True)
class VerificationResult:
    key: str
    path: str
    status: str
    ok: bool
    detail: str


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{location} 必须是对象")
    return value


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location} 必须是非空字符串")
    return value


def _validate_identifier(value: Any, location: str) -> str:
    identifier = _require_nonempty_string(value, location)
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ManifestError(f"{location} 不是合法标识符：{identifier!r}")
    return identifier


def _validate_relative_path(value: Any, location: str) -> str:
    path_text = _require_nonempty_string(value, location)
    if "\\" in path_text:
        raise ManifestError(f"{location} 必须使用正斜杠：{path_text!r}")

    posix_path = PurePosixPath(path_text)
    windows_path = PureWindowsPath(path_text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ManifestError(f"{location} 必须是安全的相对路径：{path_text!r}")
    return path_text


def _validate_https_urls(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{location} 必须是至少含一个地址的数组")

    urls: list[str] = []
    for index, item in enumerate(value):
        url = _require_nonempty_string(item, f"{location}[{index}]")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ManifestError(f"{location}[{index}] 必须是 HTTPS 地址：{url!r}")
        urls.append(url)
    return tuple(urls)


def validate_manifest(manifest: Any) -> dict[str, Any]:
    root = _require_mapping(manifest, "清单")
    if root.get("schema_version") != 1:
        raise ManifestError("schema_version 必须为 1")
    _require_nonempty_string(root.get("generated_at"), "generated_at")
    _require_nonempty_string(root.get("release_profile"), "release_profile")

    artifacts = root.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("artifacts 必须是非空数组")

    artifact_ids: set[str] = set()
    file_keys: set[str] = set()
    derived_references: list[tuple[str, str]] = []

    for artifact_index, artifact_value in enumerate(artifacts):
        location = f"artifacts[{artifact_index}]"
        artifact = _require_mapping(artifact_value, location)
        artifact_id = _validate_identifier(artifact.get("id"), f"{location}.id")
        if artifact_id in artifact_ids:
            raise ManifestError(f"资源 id 重复：{artifact_id}")
        artifact_ids.add(artifact_id)

        if not isinstance(artifact.get("required"), bool):
            raise ManifestError(f"{location}.required 必须是布尔值")
        _require_nonempty_string(artifact.get("kind"), f"{location}.kind")
        _require_nonempty_string(artifact.get("version"), f"{location}.version")

        license_info = _require_mapping(artifact.get("license"), f"{location}.license")
        _require_nonempty_string(license_info.get("spdx"), f"{location}.license.spdx")
        _validate_https_urls(
            [license_info.get("source_url")], f"{location}.license.source_url"
        )
        _require_nonempty_string(
            license_info.get("redistribution"),
            f"{location}.license.redistribution",
        )

        files = artifact.get("files")
        if not isinstance(files, list) or not files:
            raise ManifestError(f"{location}.files 必须是非空数组")

        local_file_ids: set[str] = set()
        for file_index, file_value in enumerate(files):
            file_location = f"{location}.files[{file_index}]"
            file_info = _require_mapping(file_value, file_location)
            file_id = _validate_identifier(file_info.get("id"), f"{file_location}.id")
            if file_id in local_file_ids:
                raise ManifestError(f"{artifact_id} 中的文件 id 重复：{file_id}")
            local_file_ids.add(file_id)

            key = f"{artifact_id}/{file_id}"
            file_keys.add(key)
            scope = file_info.get("scope")
            if scope not in VALID_SCOPES:
                raise ManifestError(
                    f"{file_location}.scope 必须是 download 或 installed"
                )
            _validate_relative_path(file_info.get("path"), f"{file_location}.path")

            size = file_info.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ManifestError(f"{file_location}.size 必须是正整数")
            sha256 = file_info.get("sha256")
            if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
                raise ManifestError(f"{file_location}.sha256 必须是小写 SHA-256")

            urls = file_info.get("urls")
            derived_from = file_info.get("derived_from")
            if urls is None and derived_from is None:
                raise ManifestError(
                    f"{file_location} 必须提供 urls 或 derived_from"
                )
            if urls is not None:
                _validate_https_urls(urls, f"{file_location}.urls")
            if derived_from is not None:
                reference = _require_nonempty_string(
                    derived_from, f"{file_location}.derived_from"
                )
                if "/" not in reference:
                    reference = f"{artifact_id}/{reference}"
                derived_references.append((key, reference))

    for key, reference in derived_references:
        if reference not in file_keys:
            raise ManifestError(f"{key} 引用了不存在的来源文件：{reference}")

    return root


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"无法读取资源清单 {manifest_path}：{error}") from error
    return validate_manifest(manifest)


def iter_files(
    manifest: dict[str, Any],
    scopes: Iterable[str] = ("installed",),
    include_optional: bool = False,
) -> Iterator[ArtifactFile]:
    validate_manifest(manifest)
    selected_scopes = frozenset(scopes)
    invalid_scopes = selected_scopes - VALID_SCOPES
    if invalid_scopes:
        raise ManifestError(f"未知校验范围：{', '.join(sorted(invalid_scopes))}")

    for artifact in manifest["artifacts"]:
        required = artifact["required"]
        if not required and not include_optional:
            continue
        for file_info in artifact["files"]:
            if file_info["scope"] not in selected_scopes:
                continue
            yield ArtifactFile(
                artifact_id=artifact["id"],
                file_id=file_info["id"],
                required=required,
                scope=file_info["scope"],
                relative_path=file_info["path"],
                size=file_info["size"],
                sha256=file_info["sha256"],
                urls=tuple(file_info.get("urls", ())),
                derived_from=file_info.get("derived_from"),
            )


def calculate_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root: Path, relative_path: str) -> Path | None:
    resolved_root = root.resolve()
    target = (resolved_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return None
    return target


def verify_file(
    file_info: ArtifactFile,
    root: Path | str,
    full_hash: bool = False,
) -> VerificationResult:
    target = _safe_target(Path(root), file_info.relative_path)
    if target is None:
        return VerificationResult(
            file_info.key,
            file_info.relative_path,
            "unsafe_path",
            False,
            "目标路径超出校验根目录",
        )
    if not target.is_file():
        return VerificationResult(
            file_info.key,
            str(target),
            "missing",
            False,
            "文件不存在",
        )

    actual_size = target.stat().st_size
    if actual_size != file_info.size:
        return VerificationResult(
            file_info.key,
            str(target),
            "size_mismatch",
            False,
            f"大小不符：实际 {actual_size}，预期 {file_info.size}",
        )

    if full_hash:
        actual_hash = calculate_sha256(target)
        if actual_hash != file_info.sha256:
            return VerificationResult(
                file_info.key,
                str(target),
                "hash_mismatch",
                False,
                f"SHA-256 不符：实际 {actual_hash}",
            )

    detail = "大小与 SHA-256 均匹配" if full_hash else "文件存在且大小匹配"
    return VerificationResult(file_info.key, str(target), "ok", True, detail)


def verify_artifacts(
    manifest: dict[str, Any],
    root: Path | str,
    scopes: Iterable[str] = ("installed",),
    include_optional: bool = False,
    full_hash: bool = False,
) -> list[VerificationResult]:
    return [
        verify_file(file_info, root=root, full_hash=full_hash)
        for file_info in iter_files(
            manifest,
            scopes=scopes,
            include_optional=include_optional,
        )
    ]


def _selected_scopes(scope: str) -> Sequence[str]:
    return tuple(VALID_SCOPES) if scope == "all" else (scope,)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验分享版外部资源清单和文件")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--root", type=Path, default=BASE_DIR)
    parser.add_argument(
        "--scope",
        choices=("installed", "download", "all"),
        default="installed",
    )
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--full", action="store_true", help="计算 SHA-256；大模型会较慢")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        results = verify_artifacts(
            manifest,
            root=args.root,
            scopes=_selected_scopes(args.scope),
            include_optional=args.include_optional,
            full_hash=args.full,
        )
    except ManifestError as error:
        print(f"资源清单错误：{error}", file=sys.stderr)
        return 2

    if args.json_output:
        print(
            json.dumps(
                {
                    "ok": all(result.ok for result in results),
                    "checked": len(results),
                    "results": [asdict(result) for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for result in results:
            marker = "通过" if result.ok else "失败"
            print(f"[{marker}] {result.key}: {result.detail}")
        print(
            f"校验完成：{sum(result.ok for result in results)}/{len(results)} 项通过"
        )
    return 0 if results and all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
