import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence


DEEPSEEK_V4_SINGLE_INPUT_CHARS = 600_000
DEEPSEEK_V4_CHUNK_CHARS = 450_000
FALLBACK_SINGLE_INPUT_CHARS = 20_000
FALLBACK_CHUNK_CHARS = 20_000
CHUNK_CONTEXT_SEGMENTS = 5
DEEPSEEK_MAX_OUTPUT_TOKENS = 32_000
FALLBACK_MAX_OUTPUT_TOKENS = 8_000
SPEAKER_TAG_PATTERN = re.compile(
    r"^\s*\[(?P<label>说话人\d+|其他声音|说话人未知)\]\s*"
)

ProgressCallback = Callable[[str, int, int, str], None]


class SubtitleCorrectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubtitleEntry:
    subtitle_id: str
    timeline: str
    text: str


@dataclass(frozen=True)
class ModelCorrectionProfile:
    single_input_chars: int
    chunk_chars: int
    json_output: bool
    max_output_tokens: int


@dataclass(frozen=True)
class CorrectionChunk:
    context_entries: List[SubtitleEntry]
    target_ids: List[str]


@dataclass(frozen=True)
class SubtitleCorrectionResult:
    corrected_srt: str
    changed_count: int
    total_segments: int
    changed_ids: List[str]
    chunk_count: int
    model: str


@dataclass(frozen=True)
class SubtitleTextChange:
    subtitle_id: str
    timeline: str
    before: str
    after: str


def split_speaker_tag(text: str) -> tuple[str, str]:
    value = str(text or "")
    match = SPEAKER_TAG_PATTERN.match(value)
    if not match:
        return "", value.strip()
    return f"[{match.group('label')}]", value[match.end() :].strip()


def preserve_speaker_tag(original_text: str, corrected_text: str) -> str:
    original_tag, _ = split_speaker_tag(original_text)
    _, corrected_body = split_speaker_tag(corrected_text)
    if original_tag:
        return f"{original_tag} {corrected_body}".rstrip()
    return corrected_body


def resolve_correction_profile(settings: Dict[str, Any]) -> ModelCorrectionProfile:
    model = str(settings.get("model") or "").strip().lower()
    if model in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return ModelCorrectionProfile(
            single_input_chars=DEEPSEEK_V4_SINGLE_INPUT_CHARS,
            chunk_chars=DEEPSEEK_V4_CHUNK_CHARS,
            json_output=True,
            max_output_tokens=DEEPSEEK_MAX_OUTPUT_TOKENS,
        )
    return ModelCorrectionProfile(
        single_input_chars=FALLBACK_SINGLE_INPUT_CHARS,
        chunk_chars=FALLBACK_CHUNK_CHARS,
        json_output=False,
        max_output_tokens=FALLBACK_MAX_OUTPUT_TOKENS,
    )


def parse_srt(srt_content: str) -> List[SubtitleEntry]:
    normalized = str(srt_content or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip().lstrip("\ufeff")
    if not normalized:
        raise SubtitleCorrectionError("字幕内容为空")

    entries = []
    seen_ids = set()
    for block_index, block in enumerate(re.split(r"\n\s*\n", normalized), start=1):
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            raise SubtitleCorrectionError(f"第 {block_index} 个字幕块格式不正确")

        subtitle_id = lines[0].strip()
        timeline = lines[1].strip()
        text = "\n".join(lines[2:]).strip()
        if not subtitle_id:
            raise SubtitleCorrectionError(f"第 {block_index} 个字幕块缺少编号")
        if subtitle_id in seen_ids:
            raise SubtitleCorrectionError(f"字幕编号重复: {subtitle_id}")
        seen_ids.add(subtitle_id)
        entries.append(SubtitleEntry(subtitle_id, timeline, text))

    return entries


def render_srt(entries: Sequence[SubtitleEntry]) -> str:
    if not entries:
        raise SubtitleCorrectionError("没有可写入的字幕块")
    blocks = [
        f"{entry.subtitle_id}\n{entry.timeline}\n{entry.text}"
        for entry in entries
    ]
    return "\n\n".join(blocks) + "\n"


def compare_srt_versions(
    original_srt: str,
    corrected_srt: str,
) -> List[SubtitleTextChange]:
    original_entries = parse_srt(original_srt)
    corrected_entries = parse_srt(corrected_srt)
    _validate_srt_identity(original_entries, corrected_entries)
    return [
        SubtitleTextChange(
            subtitle_id=original.subtitle_id,
            timeline=original.timeline,
            before=original.text,
            after=corrected.text,
        )
        for original, corrected in zip(original_entries, corrected_entries)
        if original.text != corrected.text
    ]


def build_correction_chunks(
    entries: Sequence[SubtitleEntry],
    settings: Dict[str, Any],
) -> List[CorrectionChunk]:
    if not entries:
        return []

    profile = resolve_correction_profile(settings)
    if _entries_size(entries) <= profile.single_input_chars:
        return [
            CorrectionChunk(
                context_entries=list(entries),
                target_ids=[entry.subtitle_id for entry in entries],
            )
        ]

    target_budget = min(
        profile.chunk_chars,
        max(1, int(profile.single_input_chars * 0.8)),
    )
    target_ranges = []
    start = 0
    current_size = 0
    for index, entry in enumerate(entries):
        entry_size = _entry_size(entry)
        if index > start and current_size + entry_size > target_budget:
            target_ranges.append((start, index))
            start = index
            current_size = 0
        current_size += entry_size
    target_ranges.append((start, len(entries)))

    chunks = []
    for target_start, target_end in target_ranges:
        context_start = target_start
        context_end = target_end
        context_size = _entries_size(entries[context_start:context_end])
        for _ in range(CHUNK_CONTEXT_SEGMENTS):
            if context_start > 0:
                left_size = _entry_size(entries[context_start - 1])
                if context_size + left_size <= profile.single_input_chars:
                    context_start -= 1
                    context_size += left_size
            if context_end < len(entries):
                right_size = _entry_size(entries[context_end])
                if context_size + right_size <= profile.single_input_chars:
                    context_end += 1
                    context_size += right_size
        chunks.append(
            CorrectionChunk(
                context_entries=list(entries[context_start:context_end]),
                target_ids=[
                    entry.subtitle_id for entry in entries[target_start:target_end]
                ],
            )
        )
    return chunks


def correct_srt_with_llm(
    srt_content: str,
    title: str,
    client: Any,
    settings: Dict[str, Any],
    progress: ProgressCallback | None = None,
) -> SubtitleCorrectionResult:
    entries = parse_srt(srt_content)
    chunks = build_correction_chunks(entries, settings)
    if not chunks:
        raise SubtitleCorrectionError("没有可校对的字幕")

    profile = resolve_correction_profile(settings)
    corrections: Dict[str, str] = {}
    total_chunks = len(chunks)
    _notify(progress, "preparing", 0, total_chunks, "正在整理字幕上下文")

    for chunk_index, chunk in enumerate(chunks, start=1):
        _notify(
            progress,
            "correcting",
            chunk_index - 1,
            total_chunks,
            f"正在结合上下文校对第 {chunk_index}/{total_chunks} 段",
        )
        chunk_corrections = _request_chunk_corrections(
            client=client,
            settings=settings,
            profile=profile,
            title=title,
            chunk=chunk,
        )
        corrections.update(chunk_corrections)
        _notify(
            progress,
            "validating",
            chunk_index,
            total_chunks,
            f"已校验第 {chunk_index}/{total_chunks} 段",
        )

    corrected_entries = [
        replace(entry, text=corrections.get(entry.subtitle_id, entry.text))
        for entry in entries
    ]
    changed_ids = [
        entry.subtitle_id
        for entry in entries
        if corrections.get(entry.subtitle_id, entry.text) != entry.text
    ]
    corrected_srt = render_srt(corrected_entries)
    _validate_srt_identity(entries, parse_srt(corrected_srt))
    _notify(progress, "completed", total_chunks, total_chunks, "字幕校对完成")
    return SubtitleCorrectionResult(
        corrected_srt=corrected_srt,
        changed_count=len(changed_ids),
        total_segments=len(entries),
        changed_ids=changed_ids,
        chunk_count=total_chunks,
        model=str(settings.get("model") or ""),
    )


def persist_corrected_srt(
    srt_path: str | Path,
    original_srt: str,
    corrected_srt: str,
) -> Path:
    path = Path(srt_path)
    original_entries = parse_srt(original_srt)
    corrected_entries = parse_srt(corrected_srt)
    _validate_srt_identity(original_entries, corrected_entries)

    path.parent.mkdir(parents=True, exist_ok=True)
    raw_backup_path = Path(str(path) + ".raw.bak")
    if not raw_backup_path.exists():
        _atomic_write_text(raw_backup_path, render_srt(original_entries))
    _atomic_write_text(path, render_srt(corrected_entries))
    return raw_backup_path


def load_persisted_correction(
    srt_path: str | Path,
    model: str = "",
) -> SubtitleCorrectionResult | None:
    path = Path(srt_path)
    raw_backup_path = Path(str(path) + ".raw.bak")
    if not path.exists() or not raw_backup_path.exists():
        return None

    original_entries = parse_srt(raw_backup_path.read_text(encoding="utf-8"))
    corrected_srt = path.read_text(encoding="utf-8")
    corrected_entries = parse_srt(corrected_srt)
    _validate_srt_identity(original_entries, corrected_entries)
    changed_ids = [
        original.subtitle_id
        for original, corrected in zip(original_entries, corrected_entries)
        if original.text != corrected.text
    ]
    return SubtitleCorrectionResult(
        corrected_srt=render_srt(corrected_entries),
        changed_count=len(changed_ids),
        total_segments=len(corrected_entries),
        changed_ids=changed_ids,
        chunk_count=0,
        model=str(model or ""),
    )


def _request_chunk_corrections(
    client: Any,
    settings: Dict[str, Any],
    profile: ModelCorrectionProfile,
    title: str,
    chunk: CorrectionChunk,
) -> Dict[str, str]:
    messages = _build_correction_messages(title, chunk)
    raw = _call_chat(
        client,
        settings,
        messages,
        profile.max_output_tokens,
        profile.json_output,
    )
    target_id_set = set(chunk.target_ids)
    target_entries = {
        entry.subtitle_id: entry
        for entry in chunk.context_entries
        if entry.subtitle_id in target_id_set
    }
    try:
        payload = _parse_json_response(raw)
        return _validate_correction_payload(payload, target_entries)
    except SubtitleCorrectionError as first_error:
        repair_messages = _build_repair_messages(raw, target_entries, first_error)
        repaired = _call_chat(
            client,
            settings,
            repair_messages,
            profile.max_output_tokens,
            profile.json_output,
        )
        try:
            payload = _parse_json_response(repaired)
            return _validate_correction_payload(payload, target_entries)
        except SubtitleCorrectionError as second_error:
            raise SubtitleCorrectionError(
                "模型两次都没有返回可安全应用的字幕校对结果。"
                f"首次错误: {first_error}; 修复错误: {second_error}"
            ) from second_error


def _build_correction_messages(
    title: str,
    chunk: CorrectionChunk,
) -> List[Dict[str, str]]:
    target_ids = set(chunk.target_ids)
    subtitle_payload = [
        {
            "id": entry.subtitle_id,
            "time": entry.timeline,
            "text": entry.text,
            "target": entry.subtitle_id in target_ids,
        }
        for entry in chunk.context_entries
    ]
    system_prompt = (
        "你是谨慎的 ASR 字幕校对员。请结合视频标题、前后字幕和全文语义，只修正明确的错别字、"
        "同音误写、明显误听以及能由上下文确定的专有名词。不得总结、翻译、润色观点、补充事实、"
        "删除口语信息或改变说话人的原意。target=false 的字幕仅供理解上下文，绝对不能修改。"
        "字幕可能以[说话人1]、[说话人2]或[其他声音]开头；这些标签属于不可修改的结构信息，"
        "返回完整字幕文本时必须原样保留，不得删除、增加或更换标签。"
        "只返回 JSON 对象，格式为 {\"corrections\":[{\"id\":\"字幕编号\"," 
        "\"text\":\"修正后的完整字幕文本\",\"reason\":\"简短原因\"}]}。"
        "只列出确实需要修改且 target=true 的字幕；没有修改时返回空数组。不得返回时间轴。"
    )
    user_prompt = (
        f"视频标题：{title or '未命名视频'}\n"
        "请校对下面的字幕数组。数组顺序就是视频语境顺序：\n"
        + json.dumps(subtitle_payload, ensure_ascii=False, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_repair_messages(
    raw: str,
    target_entries: Dict[str, SubtitleEntry],
    error: Exception,
) -> List[Dict[str, str]]:
    allowed = {
        subtitle_id: entry.text for subtitle_id, entry in target_entries.items()
    }
    system_prompt = (
        "修复下面的字幕校对输出，使其成为可安全应用的 JSON 对象。顶层只能包含 corrections 数组；"
        "每项只能使用允许的字幕编号，并包含非空 text。不要新增校对内容，只修复 JSON 和结构问题。"
        "只输出 JSON 对象。"
    )
    user_prompt = (
        f"检测到的问题：{error}\n"
        "允许修改的原字幕："
        + json.dumps(allowed, ensure_ascii=False, separators=(",", ":"))
        + "\n待修复输出：\n"
        + str(raw or "")
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_chat(
    client: Any,
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_tokens: int,
    json_output: bool,
) -> str:
    kwargs = {
        "model": settings["model"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise SubtitleCorrectionError("模型返回了空内容")
    return content


def _parse_json_response(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip().lstrip("\ufeff")
    fence = chr(96) * 3
    if text.startswith(fence):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text[len(fence) :]
        if text.endswith(fence):
            text = text[: -len(fence)]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SubtitleCorrectionError("输出中没有 JSON 对象")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise SubtitleCorrectionError(f"JSON 解析失败: {error}") from error
    if not isinstance(payload, dict):
        raise SubtitleCorrectionError("JSON 顶层必须是对象")
    return payload


def _validate_correction_payload(
    payload: Dict[str, Any],
    target_entries: Dict[str, SubtitleEntry],
) -> Dict[str, str]:
    raw_corrections = payload.get("corrections")
    if not isinstance(raw_corrections, list):
        raise SubtitleCorrectionError("corrections 必须是数组")

    result = {}
    for index, item in enumerate(raw_corrections, start=1):
        if not isinstance(item, dict):
            raise SubtitleCorrectionError(f"第 {index} 条校对结果不是对象")
        subtitle_id = str(item.get("id") or "").strip()
        if subtitle_id not in target_entries:
            raise SubtitleCorrectionError(f"模型返回了非目标字幕编号: {subtitle_id or '空'}")
        if subtitle_id in result:
            raise SubtitleCorrectionError(f"模型重复返回字幕编号: {subtitle_id}")

        raw_corrected_text = str(item.get("text") or "").strip()
        if not raw_corrected_text:
            raise SubtitleCorrectionError(f"字幕 {subtitle_id} 的修正文本为空")
        original_text = target_entries[subtitle_id].text
        corrected_text = preserve_speaker_tag(original_text, raw_corrected_text)
        _validate_text_change(subtitle_id, original_text, corrected_text)
        if corrected_text != original_text:
            result[subtitle_id] = corrected_text
    return result


def _validate_text_change(subtitle_id: str, original: str, corrected: str) -> None:
    original_length = len(original.strip())
    corrected_length = len(corrected.strip())
    maximum_length = max(original_length * 3, original_length + 80, 20)
    if corrected_length > maximum_length:
        raise SubtitleCorrectionError(f"字幕 {subtitle_id} 的修正文本异常变长")
    if original_length >= 12 and corrected_length < max(1, original_length // 5):
        raise SubtitleCorrectionError(f"字幕 {subtitle_id} 的修正文本异常缩短")


def _validate_srt_identity(
    original_entries: Sequence[SubtitleEntry],
    corrected_entries: Sequence[SubtitleEntry],
) -> None:
    original_identity = [
        (entry.subtitle_id, entry.timeline) for entry in original_entries
    ]
    corrected_identity = [
        (entry.subtitle_id, entry.timeline) for entry in corrected_entries
    ]
    if original_identity != corrected_identity:
        raise SubtitleCorrectionError("校对结果改变了字幕编号、数量或时间轴")


def _entry_size(entry: SubtitleEntry) -> int:
    return len(entry.subtitle_id) + len(entry.timeline) + len(entry.text) + 48


def _entries_size(entries: Sequence[SubtitleEntry]) -> int:
    return sum(_entry_size(entry) for entry in entries)


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback:
        callback(stage, current, total, message)


def _atomic_write_text(path: Path, content: str) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
