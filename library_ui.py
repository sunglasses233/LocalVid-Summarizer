import os
import re
import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List


TASK_SUFFIX_PATTERN = re.compile(r"_[0-9a-fA-F]{8}$")
SPEAKER_DISPLAY_PREFIX_PATTERN = re.compile(
    r"(?m)^(\s*)\[(?:说话人\d+|其他声音|说话人未知)\]\s*"
)
CHAT_HISTORY_MARKER = "====================== AI_CHAT_HISTORY ======================"
SRT_TIMESTAMP_PATTERN = re.compile(r"(\d{2}:\d{2}:\d{2}),(\d{3})")
SRT_TIMELINE_PATTERN = re.compile(
    r"(?m)^(\d{2}:\d{2}:\d{2}),(\d{3})\s*-->\s*"
    r"(\d{2}:\d{2}:\d{2}),(\d{3})\s*$"
)
PLAYER_SUBTITLE_LINES = {
    "顶部": "12%",
    "中部": "50%",
    "底部": "85%",
}


def clean_library_file_label(file_path: str) -> str:
    file_name = os.path.basename(str(file_path or ""))
    stem, extension = os.path.splitext(file_name)
    display_name = stem if extension.lower() == ".srt" else file_name
    display_name = TASK_SUFFIX_PATTERN.sub("", display_name)
    return display_name or stem or file_name


def build_library_file_labels(file_paths: Iterable[str]) -> Dict[str, str]:
    """Build unique labels; input order must be newest first."""
    ordered_paths: List[str] = list(file_paths)
    groups: Dict[str, List[str]] = defaultdict(list)
    for file_path in ordered_paths:
        groups[clean_library_file_label(file_path)].append(file_path)

    candidates: Dict[str, str] = {}
    for base_label, grouped_paths in groups.items():
        if len(grouped_paths) == 1:
            candidates[grouped_paths[0]] = base_label
            continue
        for index, file_path in enumerate(grouped_paths):
            version_label = "最新" if index == 0 else f"历史 {index}"
            candidates[file_path] = f"{base_label}（{version_label}）"

    labels: Dict[str, str] = {}
    used_labels = set()
    for file_path in ordered_paths:
        candidate = candidates[file_path]
        unique_label = candidate
        duplicate_number = 2
        while unique_label in used_labels:
            unique_label = f"{candidate}（版本 {duplicate_number}）"
            duplicate_number += 1
        labels[file_path] = unique_label
        used_labels.add(unique_label)
    return labels


def clean_player_subtitle_content(content: str, position: str = "底部") -> str:
    """生成播放器专用 WebVTT，不改变磁盘中的 SRT 和说话人标签。"""
    subtitle_content = str(content or "").lstrip("\ufeff")
    subtitle_content = subtitle_content.split(CHAT_HISTORY_MARKER, 1)[0].strip()
    subtitle_content = SPEAKER_DISPLAY_PREFIX_PATTERN.sub(r"\1", subtitle_content)
    if not subtitle_content:
        return ""
    cue_line = PLAYER_SUBTITLE_LINES.get(position, PLAYER_SUBTITLE_LINES["底部"])
    subtitle_content = SRT_TIMELINE_PATTERN.sub(
        rf"\1.\2 --> \3.\4 line:{cue_line} position:50% align:center",
        subtitle_content,
    )
    subtitle_content = SRT_TIMESTAMP_PATTERN.sub(r"\1.\2", subtitle_content)
    return f"WEBVTT\n\n{subtitle_content}\n"


def build_video_subtitle_tracks(
    current_srt: str,
    original_srt: str = "",
    correction_applied: bool = False,
    position: str = "底部",
) -> Dict[str, str]:
    current_track = clean_player_subtitle_content(current_srt, position)
    if not current_track:
        return {}

    current_label = "AI 校正版" if correction_applied else "识别字幕"
    tracks = {current_label: current_track}
    original_track = clean_player_subtitle_content(original_srt, position)
    if original_track and original_track != current_track:
        tracks["原始识别版"] = original_track
    return tracks


def build_video_player_fingerprint(
    video_path: str,
    subtitle_tracks: Dict[str, str],
    position: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(video_path or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(position or "").encode("utf-8"))
    for label, content in subtitle_tracks.items():
        digest.update(b"\0")
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(content).encode("utf-8"))
    return digest.hexdigest()[:20]
