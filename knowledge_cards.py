import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from config import (
    KNOWLEDGE_CARD_DIR,
    KNOWLEDGE_CARD_DRAFTS_DIR,
    KNOWLEDGE_CARD_FILES_DIR,
    KNOWLEDGE_CARD_HISTORY_DIR,
    KNOWLEDGE_CARD_INDEX_PATH,
)


SCHEMA_VERSION = 1
CARD_TYPES = (
    "action_guide",
    "resource_index",
    "viewpoint_profile",
    "general_knowledge",
)
CARD_TYPE_LABELS = {
    "action_guide": "行动指南",
    "resource_index": "资源索引",
    "viewpoint_profile": "观点档案",
    "general_knowledge": "通用知识",
}
RESOURCE_TITLE_PREFIX = "资源-"

DEEPSEEK_V4_SINGLE_INPUT_CHARS = 600_000
DEEPSEEK_V4_CHUNK_CHARS = 450_000
FALLBACK_SINGLE_INPUT_CHARS = 20_000
FALLBACK_CHUNK_CHARS = 20_000
CHUNK_OVERLAP_BLOCKS = 5
MAX_GENERATED_CARDS = 30
MAX_BUNDLE_OUTPUT_TOKENS = 32_000
MAX_SINGLE_CARD_OUTPUT_TOKENS = 8_000

SINGLE_CARD_EDITABLE_FIELDS = frozenset(
    {
        "title",
        "summary",
        "tags",
        "aliases",
        "retrieval_questions",
        "body",
        "body_markdown",
        "evidence",
        "ai_supplement_markdown",
        "caveats",
    }
)

METADATA_START = "<!-- KNOWLEDGE_CARD_METADATA\n"
METADATA_END = "\n-->\n"
CHINA_TIMEZONE = timezone(timedelta(hours=8))
TIMESTAMP_BLOCK_PATTERN = re.compile(r"\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]\s*")
TIMESTAMP_VALUE_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")

ProgressCallback = Callable[[str, int, int, str], None]


class KnowledgeCardError(RuntimeError):
    pass


@dataclass
class ModelContextProfile:
    single_input_chars: int
    chunk_chars: int
    json_output: bool
    max_bundle_output_tokens: int = MAX_BUNDLE_OUTPUT_TOKENS
    max_single_output_tokens: int = MAX_SINGLE_CARD_OUTPUT_TOKENS


@dataclass
class CardSource:
    source_id: str
    title: str
    url: str
    srt_file: str
    collection: str
    transcript: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CardSource":
        return cls(
            source_id=str(data.get("source_id") or ""),
            title=str(data.get("title") or ""),
            url=str(data.get("url") or ""),
            srt_file=str(data.get("srt_file") or ""),
            collection=str(data.get("collection") or "默认收藏夹"),
            transcript=str(data.get("transcript") or ""),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class KnowledgeCard:
    card_id: str
    logical_key: str
    card_type: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    retrieval_questions: List[str] = field(default_factory=list)
    body_markdown: str = ""
    evidence: List[Dict[str, str]] = field(default_factory=list)
    ai_supplement_markdown: str = ""
    caveats: List[str] = field(default_factory=list)
    parent_card_id: str = ""
    child_card_ids: List[str] = field(default_factory=list)
    selected: bool = True
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeCard":
        return cls(
            card_id=str(data.get("card_id") or ""),
            logical_key=str(data.get("logical_key") or ""),
            card_type=str(data.get("card_type") or "general_knowledge"),
            title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""),
            tags=_normalize_string_list(data.get("tags")),
            aliases=_normalize_string_list(data.get("aliases")),
            retrieval_questions=_normalize_string_list(data.get("retrieval_questions")),
            body_markdown=str(data.get("body_markdown") or ""),
            evidence=_normalize_evidence(data.get("evidence")),
            ai_supplement_markdown=str(data.get("ai_supplement_markdown") or ""),
            caveats=_normalize_string_list(data.get("caveats")),
            parent_card_id=str(data.get("parent_card_id") or ""),
            child_card_ids=_normalize_string_list(data.get("child_card_ids")),
            selected=bool(data.get("selected", True)),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class CardBundle:
    source: CardSource
    detected_type: str
    cards: List[KnowledgeCard]
    generated_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CardBundle":
        source = CardSource.from_dict(data.get("source") or {})
        cards = [KnowledgeCard.from_dict(item) for item in data.get("cards") or []]
        return cls(
            source=source,
            detected_type=str(data.get("detected_type") or "general_knowledge"),
            cards=cards,
            generated_at=str(data.get("generated_at") or now_china_iso()),
            updated_at=str(data.get("updated_at") or now_china_iso()),
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
        )


def now_china_iso() -> str:
    return datetime.now(CHINA_TIMEZONE).isoformat(timespec="seconds")


def derive_source_id(srt_file: str) -> str:
    stem = Path(srt_file).stem
    suffix = stem.rsplit("_", 1)[-1]
    if re.fullmatch(r"[0-9a-fA-F]{8,64}", suffix):
        return suffix.lower()
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]


def derive_video_title(srt_file: str) -> str:
    stem = Path(srt_file).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{8,64}", parts[1]):
        return parts[0]
    return stem


def resolve_model_context_profile(settings: Dict[str, Any]) -> ModelContextProfile:
    model = str(settings.get("model") or "").strip().lower()
    if model in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return ModelContextProfile(
            single_input_chars=DEEPSEEK_V4_SINGLE_INPUT_CHARS,
            chunk_chars=DEEPSEEK_V4_CHUNK_CHARS,
            json_output=True,
        )
    return ModelContextProfile(
        single_input_chars=FALLBACK_SINGLE_INPUT_CHARS,
        chunk_chars=FALLBACK_CHUNK_CHARS,
        json_output=False,
    )


def split_timestamp_blocks(transcript: str) -> List[str]:
    text = str(transcript or "").strip()
    if not text:
        return []

    matches = list(TIMESTAMP_BLOCK_PATTERN.finditer(text))
    if not matches:
        return [text]

    blocks = []
    prefix = text[: matches[0].start()].strip()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        if index == 0 and prefix:
            block = f"{prefix} {block}"
        if block:
            blocks.append(block)
    return blocks


def chunk_transcript(
    transcript: str,
    max_chars: int,
    overlap_blocks: int = CHUNK_OVERLAP_BLOCKS,
) -> List[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    raw_blocks = split_timestamp_blocks(transcript)
    blocks = []
    for block in raw_blocks:
        if len(block) <= max_chars:
            blocks.append(block)
            continue
        for start in range(0, len(block), max_chars):
            blocks.append(block[start : start + max_chars])

    if not blocks:
        return []

    chunks = []
    start = 0
    while start < len(blocks):
        current = []
        current_length = 0
        end = start
        while end < len(blocks):
            separator_length = 1 if current else 0
            candidate_length = current_length + separator_length + len(blocks[end])
            if current and candidate_length > max_chars:
                break
            current.append(blocks[end])
            current_length = candidate_length
            end += 1

        chunks.append(" ".join(current))
        if end >= len(blocks):
            break
        start = max(start + 1, end - max(0, overlap_blocks))
    return chunks


def transcript_chunks_for_model(
    transcript: str,
    settings: Dict[str, Any],
) -> List[str]:
    profile = resolve_model_context_profile(settings)
    if len(transcript) <= profile.single_input_chars:
        return [transcript]
    return chunk_transcript(transcript, profile.chunk_chars)


def generate_card_bundle(
    client: Any,
    settings: Dict[str, Any],
    source: CardSource,
    forced_type: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> CardBundle:
    if not source.transcript.strip():
        raise KnowledgeCardError("当前视频没有可用于生成知识卡片的字幕")
    if forced_type and forced_type not in CARD_TYPES:
        raise KnowledgeCardError(f"不支持的知识卡片类型: {forced_type}")

    profile = resolve_model_context_profile(settings)
    chunks = transcript_chunks_for_model(source.transcript, settings)
    if len(chunks) == 1:
        _notify(progress, "generating", 1, 1, "正在分析完整字幕并生成知识卡片")
        payload = _call_json_with_repair(
            client,
            settings,
            _build_generation_messages(source, chunks[0], forced_type),
            profile.max_bundle_output_tokens,
            profile.json_output,
        )
    else:
        partial_payloads = []
        for index, chunk in enumerate(chunks, start=1):
            _notify(progress, "extracting", index, len(chunks), f"正在提取第 {index}/{len(chunks)} 段字幕")
            partial_payloads.append(
                _call_json_with_repair(
                    client,
                    settings,
                    _build_generation_messages(source, chunk, forced_type, partial=True),
                    profile.max_bundle_output_tokens,
                    profile.json_output,
                )
            )

        _notify(progress, "merging", len(chunks), len(chunks), "正在合并、去重并建立卡片关系")
        payload = _call_json_with_repair(
            client,
            settings,
            _build_merge_messages(source, partial_payloads, forced_type),
            profile.max_bundle_output_tokens,
            profile.json_output,
        )

    _notify(progress, "validating", 1, 1, "正在校验知识卡片结构")
    bundle = _bundle_with_structure_repair(
        client,
        settings,
        payload,
        source,
        forced_type,
        profile.max_bundle_output_tokens,
        profile.json_output,
    )
    save_draft(bundle)
    _notify(progress, "draft_saved", 1, 1, "草稿已自动保存")
    return bundle


def revise_card(
    client: Any,
    settings: Dict[str, Any],
    bundle: CardBundle,
    card_id: str,
    instruction: str,
    progress: Optional[ProgressCallback] = None,
) -> CardBundle:
    target = next((card for card in bundle.cards if card.card_id == card_id), None)
    if target is None:
        raise KnowledgeCardError("没有找到要修改的知识卡片")
    if not instruction.strip():
        raise KnowledgeCardError("请输入修改要求")

    profile = resolve_model_context_profile(settings)
    evidence_context = extract_evidence_context(bundle.source.transcript, target.evidence)
    _notify(progress, "revising", 1, 1, "正在修改当前知识卡片")
    payload = _call_json_with_repair(
        client,
        settings,
        _build_single_revision_messages(bundle.source, target, evidence_context, instruction),
        profile.max_single_output_tokens,
        profile.json_output,
    )

    payload = _normalize_single_card_revision_payload(payload, target)
    revised_bundle = _bundle_with_structure_repair(
        client,
        settings,
        payload,
        bundle.source,
        target.card_type,
        profile.max_single_output_tokens,
        profile.json_output,
    )
    revised = revised_bundle.cards[0]
    revised.card_id = target.card_id
    revised.logical_key = target.logical_key
    revised.parent_card_id = target.parent_card_id
    revised.child_card_ids = target.child_card_ids
    revised.selected = target.selected
    revised.created_at = target.created_at

    bundle.cards = [revised if card.card_id == card_id else card for card in bundle.cards]
    bundle.updated_at = now_china_iso()
    save_draft(bundle)
    _notify(progress, "draft_saved", 1, 1, "修改结果已保存到草稿")
    return bundle


def _normalize_single_card_revision_payload(
    payload: Dict[str, Any],
    target: KnowledgeCard,
) -> Dict[str, Any]:
    raw_card = _extract_single_card_revision(payload)
    if raw_card is None:
        fields = ", ".join(str(key) for key in list(payload)[:12]) or "空对象"
        raise KnowledgeCardError(
            f"模型没有返回可识别的修改后卡片（顶层字段：{fields}）"
        )

    updates = {
        key: value
        for key, value in raw_card.items()
        if key in SINGLE_CARD_EDITABLE_FIELDS
    }
    if "body" in updates and "body_markdown" not in updates:
        updates["body_markdown"] = updates["body"]
    updates.pop("body", None)
    if not updates:
        fields = ", ".join(str(key) for key in list(raw_card)[:12]) or "空对象"
        raise KnowledgeCardError(
            f"模型返回的卡片不含可修改字段（卡片字段：{fields}）"
        )

    merged = asdict(target)
    merged.update(updates)
    merged["logical_key"] = target.logical_key
    merged["card_type"] = target.card_type
    return {"detected_type": target.card_type, "cards": [merged]}


def _extract_single_card_revision(
    payload: Any,
    depth: int = 0,
) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict) or depth > 2:
        return None

    raw_cards = payload.get("cards")
    if isinstance(raw_cards, list) and raw_cards and isinstance(raw_cards[0], dict):
        return raw_cards[0]
    if isinstance(raw_cards, dict):
        return raw_cards

    for key in ("card", "updated_card"):
        raw_card = payload.get(key)
        if isinstance(raw_card, dict):
            return raw_card

    for key in ("result", "data"):
        nested = _extract_single_card_revision(payload.get(key), depth + 1)
        if nested is not None:
            return nested

    if SINGLE_CARD_EDITABLE_FIELDS.intersection(payload):
        return payload
    return None


def revise_bundle(
    client: Any,
    settings: Dict[str, Any],
    bundle: CardBundle,
    instruction: str,
    include_full_transcript: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> CardBundle:
    if not instruction.strip():
        raise KnowledgeCardError("请输入修改要求")
    profile = resolve_model_context_profile(settings)
    _notify(progress, "revising", 1, 1, "正在修改整组知识卡片")
    payload = _call_json_with_repair(
        client,
        settings,
        _build_bundle_revision_messages(bundle, instruction, include_full_transcript),
        profile.max_bundle_output_tokens,
        profile.json_output,
    )
    revised = _bundle_with_structure_repair(
        client,
        settings,
        payload,
        bundle.source,
        bundle.detected_type,
        profile.max_bundle_output_tokens,
        profile.json_output,
    )

    previous_by_key = {card.logical_key: card for card in bundle.cards}
    for card in revised.cards:
        previous = previous_by_key.get(card.logical_key)
        if previous:
            card.selected = previous.selected
            card.created_at = previous.created_at
    revised.generated_at = bundle.generated_at
    revised.updated_at = now_china_iso()
    save_draft(revised)
    _notify(progress, "draft_saved", 1, 1, "整组修改结果已保存到草稿")
    return revised


def _bundle_with_structure_repair(
    client: Any,
    settings: Dict[str, Any],
    payload: Dict[str, Any],
    source: CardSource,
    forced_type: Optional[str],
    max_tokens: int,
    json_output: bool,
) -> CardBundle:
    try:
        return bundle_from_model_payload(payload, source, forced_type)
    except KnowledgeCardError as first_error:
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "修复知识卡片JSON的结构。顶层必须包含detected_type和非空cards数组；"
                    "每张卡必须包含logical_key、card_type、title、summary、body_markdown、"
                    "tags、aliases、retrieval_questions、evidence、ai_supplement_markdown、caveats。"
                    "只修复结构，不新增知识事实，只输出JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"校验错误：{first_error}\n待修复内容：\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]
        repaired_payload = _parse_json_response(
            _call_chat(client, settings, repair_messages, max_tokens, json_output)
        )
        try:
            return bundle_from_model_payload(repaired_payload, source, forced_type)
        except KnowledgeCardError as second_error:
            raise KnowledgeCardError(
                f"知识卡片结构修复后仍无效。首次错误: {first_error}; 修复错误: {second_error}"
            ) from second_error


def bundle_from_model_payload(
    payload: Dict[str, Any],
    source: CardSource,
    forced_type: Optional[str] = None,
) -> CardBundle:
    if not isinstance(payload, dict):
        raise KnowledgeCardError("知识卡片结果不是JSON对象")

    detected_type = forced_type or str(payload.get("detected_type") or "general_knowledge")
    if detected_type not in CARD_TYPES:
        detected_type = "general_knowledge"

    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        raise KnowledgeCardError("模型没有生成任何知识卡片")
    if len(raw_cards) > MAX_GENERATED_CARDS:
        raw_cards = raw_cards[:MAX_GENERATED_CARDS]
    if detected_type in {"action_guide", "viewpoint_profile"}:
        raw_cards = raw_cards[:1]
    elif detected_type == "general_knowledge":
        raw_cards = _merge_general_knowledge_raw_cards(raw_cards, source)

    cards = []
    relationships = []
    used_keys = set()
    timestamp = now_china_iso()
    for position, raw in enumerate(raw_cards):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        body = str(raw.get("body_markdown") or raw.get("body") or "").strip()
        if not title or not body:
            continue

        card_type = forced_type or str(raw.get("card_type") or detected_type)
        if card_type not in CARD_TYPES:
            card_type = detected_type
        if detected_type == "resource_index":
            card_type = "resource_index"
        if card_type == "resource_index":
            title = ensure_resource_title_prefix(title)

        logical_key = _normalize_logical_key(raw.get("logical_key"), title, card_type, position)
        base_key = logical_key
        duplicate_number = 2
        while logical_key in used_keys:
            logical_key = f"{base_key}-{duplicate_number}"
            duplicate_number += 1
        used_keys.add(logical_key)

        card = KnowledgeCard(
            card_id=build_card_id(source.source_id, card_type, logical_key),
            logical_key=logical_key,
            card_type=card_type,
            title=title,
            summary=str(raw.get("summary") or "").strip(),
            tags=_normalize_string_list(raw.get("tags")),
            aliases=_normalize_string_list(raw.get("aliases")),
            retrieval_questions=_normalize_string_list(raw.get("retrieval_questions")),
            body_markdown=body,
            evidence=_validate_evidence(raw.get("evidence"), source.transcript),
            ai_supplement_markdown=_strip_supplement_timestamps(
                raw.get("ai_supplement_markdown")
            ),
            caveats=_normalize_string_list(raw.get("caveats")),
            selected=bool(raw.get("selected", True)),
            created_at=timestamp,
            updated_at=timestamp,
        )
        cards.append(card)
        relationships.append(
            {
                "card": card,
                "parent": str(raw.get("parent_logical_key") or "").strip(),
                "children": _normalize_string_list(raw.get("child_logical_keys")),
            }
        )

    if not cards:
        raise KnowledgeCardError("生成结果缺少有效的标题或正文")

    cards_by_key = {card.logical_key: card for card in cards}
    for relation in relationships:
        card = relation["card"]
        parent = cards_by_key.get(relation["parent"])
        if parent:
            card.parent_card_id = parent.card_id
            if card.card_id not in parent.child_card_ids:
                parent.child_card_ids.append(card.card_id)
        for child_key in relation["children"]:
            child = cards_by_key.get(child_key)
            if child:
                if child.card_id not in card.child_card_ids:
                    card.child_card_ids.append(child.card_id)
                child.parent_card_id = card.card_id

    if detected_type == "resource_index" and len(cards) > 1:
        has_relationships = any(card.parent_card_id or card.child_card_ids for card in cards)
        if not has_relationships:
            parent = cards[0]
            for child in cards[1:]:
                parent.child_card_ids.append(child.card_id)
                child.parent_card_id = parent.card_id

    _apply_high_risk_caveats(cards, source)

    return CardBundle(
        source=source,
        detected_type=detected_type,
        cards=cards,
        generated_at=timestamp,
        updated_at=timestamp,
    )


def ensure_resource_title_prefix(title: str) -> str:
    normalized_title = str(title or "").strip()
    if normalized_title.startswith(RESOURCE_TITLE_PREFIX):
        return normalized_title
    return f"{RESOURCE_TITLE_PREFIX}{normalized_title}"


def _merge_general_knowledge_raw_cards(
    raw_cards: List[Any],
    source: CardSource,
) -> List[Dict[str, Any]]:
    valid_cards = [card for card in raw_cards if isinstance(card, dict)]
    if len(valid_cards) <= 1:
        return valid_cards

    body_sections = []
    supplement_sections = []
    for card in valid_cards:
        title = str(card.get("title") or "").strip()
        body = str(card.get("body_markdown") or card.get("body") or "").strip()
        if body:
            body_sections.append(f"## {title}\n\n{body}" if title else body)
        supplement = str(card.get("ai_supplement_markdown") or "").strip()
        if supplement:
            supplement_sections.append(
                f"### {title}\n\n{supplement}" if title else supplement
            )

    merged = {
        "logical_key": "core-knowledge",
        "card_type": "general_knowledge",
        "title": str(source.title or valid_cards[0].get("title") or "核心知识").strip(),
        "summary": "；".join(
            _unique_text_values(card.get("summary") for card in valid_cards)
        ),
        "tags": _merge_raw_list_field(valid_cards, "tags"),
        "aliases": _merge_raw_list_field(valid_cards, "aliases"),
        "retrieval_questions": _merge_raw_list_field(
            valid_cards,
            "retrieval_questions",
        ),
        "body_markdown": "\n\n".join(body_sections),
        "evidence": _merge_raw_evidence(valid_cards),
        "ai_supplement_markdown": "\n\n".join(supplement_sections),
        "caveats": _merge_raw_list_field(valid_cards, "caveats"),
        "parent_logical_key": "",
        "child_logical_keys": [],
    }
    return [merged]


def _unique_text_values(values: Iterable[Any]) -> List[str]:
    result = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _merge_raw_list_field(cards: List[Dict[str, Any]], field_name: str) -> List[str]:
    values = []
    for card in cards:
        raw_value = card.get(field_name)
        if isinstance(raw_value, list):
            values.extend(raw_value)
        elif raw_value:
            values.append(raw_value)
    return _unique_text_values(values)


def _merge_raw_evidence(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for card in cards:
        raw_evidence = card.get("evidence")
        if not isinstance(raw_evidence, list):
            continue
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def build_card_id(source_id: str, card_type: str, logical_key: str) -> str:
    raw = f"{source_id}|{card_type}|{logical_key}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def save_draft(bundle: CardBundle) -> Path:
    KNOWLEDGE_CARD_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle.updated_at = now_china_iso()
    path = draft_path(bundle.source.source_id)
    _atomic_write_text(path, json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
    return path


def load_draft(source_id: str) -> Optional[CardBundle]:
    path = draft_path(source_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return CardBundle.from_dict(json.load(file))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise KnowledgeCardError(f"知识卡片草稿无法读取: {error}") from error


def load_committed_bundle(source: CardSource) -> Optional[CardBundle]:
    index_data = load_index()
    matching_entries = [
        item
        for item in index_data.get("cards", [])
        if item.get("source_id") == source.source_id
    ]
    cards = []
    for entry in matching_entries:
        path = _entry_path(entry)
        if not path or not path.exists():
            continue
        try:
            metadata = parse_card_markdown_metadata(path.read_text(encoding="utf-8"))
        except (OSError, KnowledgeCardError):
            continue
        card = KnowledgeCard.from_dict(metadata)
        if card.card_id and card.title and card.body_markdown:
            card.selected = True
            cards.append(card)

    if not cards:
        return None

    generated_at = min((card.created_at for card in cards if card.created_at), default=now_china_iso())
    updated_at = max((card.updated_at for card in cards if card.updated_at), default=generated_at)
    detected_type = cards[0].card_type if cards[0].card_type in CARD_TYPES else "general_knowledge"
    return CardBundle(
        source=source,
        detected_type=detected_type,
        cards=cards,
        generated_at=generated_at,
        updated_at=updated_at,
    )


def delete_draft(source_id: str) -> None:
    path = draft_path(source_id)
    if path.exists():
        path.unlink()


def draft_path(source_id: str) -> Path:
    safe_id = re.sub(r"[^0-9a-zA-Z_-]", "", source_id) or "unknown"
    return KNOWLEDGE_CARD_DRAFTS_DIR / f"{safe_id}.json"


def render_card_markdown(card: KnowledgeCard, source: CardSource) -> str:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "card_id": card.card_id,
        "logical_key": card.logical_key,
        "card_type": card.card_type,
        "title": card.title,
        "summary": card.summary,
        "tags": card.tags,
        "aliases": card.aliases,
        "retrieval_questions": card.retrieval_questions,
        "body_markdown": card.body_markdown,
        "evidence": card.evidence,
        "ai_supplement_markdown": card.ai_supplement_markdown,
        "caveats": card.caveats,
        "parent_card_id": card.parent_card_id,
        "child_card_ids": card.child_card_ids,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
        "selected": card.selected,
        "source": {
            "source_id": source.source_id,
            "video_title": source.title,
            "url": source.url,
            "srt_file": source.srt_file,
            "collection": source.collection,
        },
    }
    lines = [
        METADATA_START.rstrip("\n"),
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "-->",
        "",
        f"# {card.title}",
        "",
        f"> {card.summary or '暂无摘要'}",
        "",
        "## 适合在这些问题下检索",
        "",
    ]
    lines.extend(f"- {question}" for question in card.retrieval_questions)
    if not card.retrieval_questions:
        lines.append("- 暂无")
    lines.extend(["", "## 卡片内容", "", card.body_markdown or "暂无", "", "## 视频依据", ""])
    if card.evidence:
        for item in card.evidence:
            timestamp = item.get("timestamp") or ""
            point = item.get("point") or ""
            lines.append(f"- {point} {timestamp}".rstrip())
    else:
        lines.append("- 暂无可定位的字幕依据")
    lines.extend(
        [
            "",
            "## AI 补充（未联网核验）",
            "",
            card.ai_supplement_markdown or "无",
            "",
            "## 注意事项与适用边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in card.caveats)
    if not card.caveats:
        lines.append("- 请结合实际情况判断")
    lines.extend(
        [
            "",
            "## 来源",
            "",
            f"- 视频：{source.title}",
            f"- 网页：{source.url or '本地导入，无网页地址'}",
            f"- 字幕：{source.srt_file}",
            f"- 卡片类型：{CARD_TYPE_LABELS.get(card.card_type, card.card_type)}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_card_markdown_metadata(content: str) -> Dict[str, Any]:
    if not content.startswith(METADATA_START):
        raise KnowledgeCardError("知识卡片缺少元数据头")
    end = content.find(METADATA_END, len(METADATA_START))
    if end < 0:
        raise KnowledgeCardError("知识卡片元数据头不完整")
    raw = content[len(METADATA_START) : end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise KnowledgeCardError(f"知识卡片元数据损坏: {error}") from error


def commit_bundle(
    bundle: CardBundle,
    selected_card_ids: Optional[Iterable[str]] = None,
) -> List[Path]:
    if selected_card_ids is None:
        selected_ids = {card.card_id for card in bundle.cards if card.selected}
    else:
        selected_ids = set(selected_card_ids)
    selected_cards = [card for card in bundle.cards if card.card_id in selected_ids]
    if not selected_cards:
        raise KnowledgeCardError("请至少选择一张知识卡片再保存")

    for directory in (
        KNOWLEDGE_CARD_DIR,
        KNOWLEDGE_CARD_FILES_DIR,
        KNOWLEDGE_CARD_DRAFTS_DIR,
        KNOWLEDGE_CARD_HISTORY_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    index_data = load_index()
    existing_entries = {item.get("card_id"): item for item in index_data.get("cards", [])}
    timestamp = now_china_iso()
    history_stamp = datetime.now(CHINA_TIMEZONE).strftime("%Y%m%d_%H%M%S")
    pending_files = []
    originals: Dict[Path, Optional[bytes]] = {}
    created_history_files = []

    for card in selected_cards:
        card.updated_at = timestamp
        existing_entry = existing_entries.get(card.card_id) or {}
        existing_path = _entry_path(existing_entry)
        if existing_path and existing_path.exists():
            try:
                old_metadata = parse_card_markdown_metadata(existing_path.read_text(encoding="utf-8"))
                card.created_at = str(old_metadata.get("created_at") or card.created_at or timestamp)
            except (OSError, KnowledgeCardError):
                card.created_at = card.created_at or timestamp
        else:
            card.created_at = card.created_at or timestamp

        destination = KNOWLEDGE_CARD_FILES_DIR / f"{_safe_filename(card.title)}__{card.card_id}.md"
        content = render_card_markdown(card, bundle.source)
        metadata = parse_card_markdown_metadata(content)
        if metadata.get("card_id") != card.card_id:
            raise KnowledgeCardError(f"知识卡片写入前校验失败: {card.title}")
        pending_files.append((card, destination, content, existing_path))

    new_entries = dict(existing_entries)
    for card, destination, _content, _existing_path in pending_files:
        new_entries[card.card_id] = _build_index_entry(card, bundle.source, destination)
    new_index = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": timestamp,
        "cards": sorted(new_entries.values(), key=lambda item: (item.get("title") or "").lower()),
    }

    touched_paths = {KNOWLEDGE_CARD_INDEX_PATH}
    for _card, destination, _content, existing_path in pending_files:
        touched_paths.add(destination)
        if existing_path:
            touched_paths.add(existing_path)
    for path in touched_paths:
        originals[path] = path.read_bytes() if path.exists() else None

    try:
        for card, destination, content, existing_path in pending_files:
            if existing_path and existing_path.exists():
                history_dir = KNOWLEDGE_CARD_HISTORY_DIR / card.card_id
                history_dir.mkdir(parents=True, exist_ok=True)
                history_path = history_dir / f"{history_stamp}.md"
                suffix = 2
                while history_path.exists():
                    history_path = history_dir / f"{history_stamp}_{suffix}.md"
                    suffix += 1
                _atomic_write_bytes(history_path, existing_path.read_bytes())
                created_history_files.append(history_path)

            _atomic_write_text(destination, content)
            if existing_path and existing_path != destination and existing_path.exists():
                existing_path.unlink()

        _atomic_write_text(
            KNOWLEDGE_CARD_INDEX_PATH,
            json.dumps(new_index, ensure_ascii=False, indent=2),
        )
    except Exception as error:
        for path, original in originals.items():
            try:
                if original is None:
                    if path.exists():
                        path.unlink()
                else:
                    _atomic_write_bytes(path, original)
            except OSError:
                pass
        for path in created_history_files:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        raise KnowledgeCardError(f"知识卡片保存失败，已恢复原文件: {error}") from error

    delete_draft(bundle.source.source_id)
    return [destination for _card, destination, _content, _existing in pending_files]


def load_index() -> Dict[str, Any]:
    if not KNOWLEDGE_CARD_INDEX_PATH.exists():
        empty_index = {"schema_version": SCHEMA_VERSION, "updated_at": "", "cards": []}
        KNOWLEDGE_CARD_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            KNOWLEDGE_CARD_INDEX_PATH,
            json.dumps(empty_index, ensure_ascii=False, indent=2),
        )
        return empty_index
    try:
        with KNOWLEDGE_CARD_INDEX_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeCardError(f"知识卡片索引无法读取: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("cards"), list):
        raise KnowledgeCardError("知识卡片索引结构无效")
    return data


def rebuild_index() -> Dict[str, Any]:
    entries = []
    for path in KNOWLEDGE_CARD_FILES_DIR.glob("*.md"):
        try:
            metadata = parse_card_markdown_metadata(path.read_text(encoding="utf-8"))
        except (OSError, KnowledgeCardError):
            continue
        entries.append(_index_entry_from_metadata(metadata, path))
    data = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_china_iso(),
        "cards": sorted(entries, key=lambda item: (item.get("title") or "").lower()),
    }
    _atomic_write_text(KNOWLEDGE_CARD_INDEX_PATH, json.dumps(data, ensure_ascii=False, indent=2))
    return data


def update_source_path(source_id: str, srt_file: str, collection: str) -> int:
    index_data = load_index()
    matching_entries = [item for item in index_data.get("cards", []) if item.get("source_id") == source_id]
    originals: Dict[Path, bytes] = {}
    original_index = (
        KNOWLEDGE_CARD_INDEX_PATH.read_bytes()
        if KNOWLEDGE_CARD_INDEX_PATH.exists()
        else None
    )
    updated_files = []

    try:
        for entry in matching_entries:
            path = _entry_path(entry)
            if not path or not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            metadata = parse_card_markdown_metadata(content)
            end = content.find(METADATA_END, len(METADATA_START))
            source = metadata.setdefault("source", {})
            previous_srt_file = str(source.get("srt_file") or "")
            source["srt_file"] = srt_file
            source["collection"] = collection
            originals[path] = content.encode("utf-8")
            visible_content = content[end:]
            if previous_srt_file:
                visible_content = visible_content.replace(
                    f"- 字幕：{previous_srt_file}",
                    f"- 字幕：{srt_file}",
                    1,
                )
            updated_content = (
                METADATA_START
                + json.dumps(metadata, ensure_ascii=False, indent=2)
                + visible_content
            )
            _atomic_write_text(path, updated_content)
            entry["source_srt"] = srt_file
            entry["source_collection"] = collection
            updated_files.append(path)

        if matching_entries:
            index_data["updated_at"] = now_china_iso()
            _atomic_write_text(
                KNOWLEDGE_CARD_INDEX_PATH,
                json.dumps(index_data, ensure_ascii=False, indent=2),
            )

        draft = load_draft(source_id)
        if draft:
            draft.source.srt_file = srt_file
            draft.source.collection = collection
            save_draft(draft)
    except Exception as error:
        for path, original in originals.items():
            try:
                _atomic_write_bytes(path, original)
            except OSError:
                pass
        try:
            if original_index is None:
                if KNOWLEDGE_CARD_INDEX_PATH.exists():
                    KNOWLEDGE_CARD_INDEX_PATH.unlink()
            else:
                _atomic_write_bytes(KNOWLEDGE_CARD_INDEX_PATH, original_index)
        except OSError:
            pass
        raise KnowledgeCardError(f"知识卡片来源路径更新失败: {error}") from error
    return len(updated_files)


def extract_evidence_context(
    transcript: str,
    evidence: List[Dict[str, str]],
    neighbor_blocks: int = 1,
) -> str:
    blocks = split_timestamp_blocks(transcript)
    if not blocks or not evidence:
        return ""
    wanted = set()
    evidence_times = {
        _timestamp_plain(item.get("timestamp") or "")
        for item in evidence
        if _timestamp_plain(item.get("timestamp") or "")
    }
    for index, block in enumerate(blocks):
        match = TIMESTAMP_BLOCK_PATTERN.match(block)
        block_time = _timestamp_plain(match.group(0) if match else "")
        if block_time in evidence_times:
            for selected in range(max(0, index - neighbor_blocks), min(len(blocks), index + neighbor_blocks + 1)):
                wanted.add(selected)
    return " ".join(blocks[index] for index in sorted(wanted))


def _build_generation_messages(
    source: CardSource,
    transcript: str,
    forced_type: Optional[str],
    partial: bool = False,
) -> List[Dict[str, str]]:
    mode = "这是长视频的一段字幕。只提取本段候选卡片，保留可合并的 logical_key。" if partial else "分析完整视频并生成最终知识卡片。"
    type_rule = f"卡片类型强制为 {forced_type}。" if forced_type else "自动判断最合适的卡片类型。"
    system = (
        "你是知识库编辑器。任务是把视频内容整理成可检索、可复用、可追溯的知识卡片。"
        "视频内容与模型补充必须严格分开；模型补充不得伪装成视频观点，也不得声称已经联网。"
        "所有视频证据必须使用字幕中真实存在的时间戳。只输出JSON对象。"
    )
    user = (
        f"{mode}\n{type_rule}\n"
        f"允许的类型：{', '.join(CARD_TYPES)}。\n"
        "分类优先级：如果视频集中介绍多个项目、软件、工具、网站、论文或其他可枚举资源，必须优先选择resource_index；"
        "action_guide生成一张行动指南，正文必须含触发场景、快速判断、行动步骤、证据清单、升级路径、风险边界；"
        "resource_index生成一期总索引，并为每个项目或资源生成可独立检索的子卡，写明名称、用途、亮点、适用对象和原始地址（仅在视频提供时填写）；总索引和所有子卡的标题必须以资源-开头；"
        "viewpoint_profile生成一张观点档案，正文必须含作者主张、论据、前提、案例、适用边界和反方视角；"
        "general_knowledge只能生成一张主卡，围绕视频最核心的可复用知识主题组织内容，次要方向必须作为同一卡片的章节，不得拆成多张卡。\n"
        "返回结构：detected_type，以及 cards 数组。每张卡必须包含 logical_key、card_type、title、summary、"
        "tags、aliases、retrieval_questions、body_markdown、evidence、ai_supplement_markdown、caveats、"
        "parent_logical_key、child_logical_keys。evidence是包含timestamp和point的对象数组。\n"
        "高风险法律、医疗、金融内容必须在caveats中提醒核验时效和专业意见。\n\n"
        f"【视频标题】{source.title}\n"
        f"【网页网址】{source.url or '本地导入'}\n"
        f"【个人笔记】{source.notes or '无'}\n"
        f"【带时间戳字幕】\n{transcript}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_merge_messages(
    source: CardSource,
    partial_payloads: List[Dict[str, Any]],
    forced_type: Optional[str],
) -> List[Dict[str, str]]:
    type_rule = f"最终类型强制为 {forced_type}。" if forced_type else "综合判断最终类型。"
    system = (
        "你是知识库总编辑。请合并长视频各片段的候选知识卡，去重、补全父子关系并统一结构。"
        "不得新增候选材料中不存在的视频事实或时间戳。模型补充必须单独放入ai_supplement_markdown。"
        "只输出JSON对象。"
    )
    user = (
        f"{type_rule}\n"
        "数量规则：action_guide、viewpoint_profile和general_knowledge最终都只能保留一张；"
        "general_knowledge必须把不同片段的次要主题合并成一张主卡；"
        "只有resource_index可以生成总索引和多个资源子卡。"
        "包含多个项目、软件、工具、网站、论文或资源推荐时，优先判定为resource_index。"
        "resource_index总索引和所有子卡的标题必须以资源-开头。\n"
        "返回detected_type和cards数组，字段与候选保持一致。\n"
        f"视频标题：{source.title}\n网页网址：{source.url or '本地导入'}\n个人笔记：{source.notes or '无'}\n"
        "候选结果：\n"
        + json.dumps(partial_payloads, ensure_ascii=False)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_single_revision_messages(
    source: CardSource,
    card: KnowledgeCard,
    evidence_context: str,
    instruction: str,
) -> List[Dict[str, str]]:
    system = (
        "你是知识卡片编辑器。根据用户要求修改当前卡片，保持事实来源边界。"
        "视频内容、个人笔记与AI补充不得混淆；不得发明时间戳。只输出JSON对象。"
    )
    user = (
        f"修改要求：{instruction}\n"
        f"视频标题：{source.title}\n网页网址：{source.url or '本地导入'}\n个人笔记：{source.notes or '无'}\n"
        f"相关字幕证据：{evidence_context or '未找到额外字幕片段，请只使用卡片已有证据'}\n"
        "当前卡片：\n"
        + json.dumps(asdict(card), ensure_ascii=False)
        + (
            '\n必须返回单个JSON对象，顶层只能包含"card"字段，格式为'
            '{"card": {"title": "...", "summary": "...", "body_markdown": "..."}}。'
            "card内请返回与当前卡片一致的完整字段，不要把卡片字段直接放在顶层。"
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_bundle_revision_messages(
    bundle: CardBundle,
    instruction: str,
    include_full_transcript: bool = False,
) -> List[Dict[str, str]]:
    system = (
        "你是知识卡片总编辑。根据用户要求修改整组草稿，保留logical_key和父子关系。"
        "不得把AI补充写成视频事实，不得发明时间戳。只输出JSON对象。"
    )
    draft_payload = {
        "detected_type": bundle.detected_type,
        "cards": [asdict(card) for card in bundle.cards],
    }
    transcript_context = (
        f"\n完整带时间戳字幕：\n{bundle.source.transcript}\n"
        if include_full_transcript
        else "\n本次为草稿编辑，不附带完整字幕；只能使用卡片已有事实和证据。\n"
    )
    user = (
        f"修改要求：{instruction}\n"
        f"视频标题：{bundle.source.title}\n网页网址：{bundle.source.url or '本地导入'}\n"
        f"个人笔记：{bundle.source.notes or '无'}\n"
        f"{transcript_context}"
        "当前草稿：\n"
        + json.dumps(draft_payload, ensure_ascii=False)
        + "\n返回detected_type和cards数组。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_json_with_repair(
    client: Any,
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_tokens: int,
    json_output: bool,
) -> Dict[str, Any]:
    raw = _call_chat(client, settings, messages, max_tokens, json_output)
    try:
        return _parse_json_response(raw)
    except KnowledgeCardError as first_error:
        repair_messages = [
            {
                "role": "system",
                "content": "修复下面的输出，使其成为语法正确的JSON对象。不要增删知识内容，只输出JSON对象。",
            },
            {"role": "user", "content": raw},
        ]
        repaired = _call_chat(client, settings, repair_messages, max_tokens, json_output)
        try:
            return _parse_json_response(repaired)
        except KnowledgeCardError as second_error:
            raise KnowledgeCardError(
                f"模型两次都没有返回有效JSON。首次错误: {first_error}; 修复错误: {second_error}"
            ) from second_error


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
        "temperature": min(float(settings.get("temperature", 0.3)), 0.3),
        "max_tokens": max_tokens,
    }
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise KnowledgeCardError("模型返回了空内容")
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
        raise KnowledgeCardError("输出中没有JSON对象")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise KnowledgeCardError(f"JSON解析失败: {error}") from error
    if not isinstance(payload, dict):
        raise KnowledgeCardError("JSON顶层必须是对象")
    return payload


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = re.split(r"[,，\n]", value)
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    result = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_evidence(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        point = str(item.get("point") or item.get("text") or "").strip()
        timestamp = _normalize_timestamp(item.get("timestamp"))
        if point:
            result.append({"timestamp": timestamp, "point": point})
    return result


def _validate_evidence(value: Any, transcript: str) -> List[Dict[str, str]]:
    valid = []
    for item in _normalize_evidence(value):
        timestamp = _timestamp_plain(item.get("timestamp") or "")
        if timestamp and f"[{timestamp}]" in transcript:
            valid.append(item)
    return valid


def _strip_supplement_timestamps(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    bracketed_timestamp = re.compile(
        r"[\(\[]\s*(?:\d{1,2}:)?\d{1,2}:\d{2}"
        r"(?:\s*[-~到至]\s*(?:\d{1,2}:)?\d{1,2}:\d{2})?\s*[\)\]]"
    )
    return re.sub(r"\s{2,}", " ", bracketed_timestamp.sub("", text)).strip()


def _apply_high_risk_caveats(cards: List[KnowledgeCard], source: CardSource) -> None:
    searchable = "\n".join(
        [source.title, source.transcript]
        + [f"{card.title}\n{card.summary}\n{card.body_markdown}" for card in cards]
    ).lower()
    risk_rules = (
        (
            ("法律", "维权", "诉讼", "合同", "赔偿", "消费者权益", "劳动仲裁"),
            "法律与政策可能变化，请核验内容时效；重要事项应咨询具备资质的法律专业人士。",
        ),
        (
            ("医疗", "疾病", "诊断", "治疗", "药物", "用药", "手术", "hpv"),
            "医疗信息不能替代诊断或治疗建议，请核验内容时效并咨询具备资质的医疗专业人士。",
        ),
        (
            ("金融", "投资", "理财", "股票", "基金", "保险", "贷款", "收益率"),
            "金融市场、产品与监管规则可能变化，请核验最新信息，并在重要决策前咨询具备资质的专业人士。",
        ),
    )
    for keywords, caveat in risk_rules:
        if any(keyword in searchable for keyword in keywords):
            for card in cards:
                if caveat not in card.caveats:
                    card.caveats.append(caveat)


def _normalize_timestamp(value: Any) -> str:
    match = TIMESTAMP_VALUE_PATTERN.search(str(value or ""))
    return f"({match.group(1)})" if match else ""


def _timestamp_plain(value: str) -> str:
    match = TIMESTAMP_VALUE_PATTERN.search(str(value or ""))
    return match.group(1) if match else ""


def _normalize_logical_key(value: Any, title: str, card_type: str, position: int) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        if card_type == "action_guide":
            raw = "action"
        elif card_type == "viewpoint_profile":
            raw = "viewpoint"
        else:
            raw = title
    normalized = re.sub(r"\s+", "-", raw)
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff:_-]", "", normalized)
    return normalized[:80] or f"card-{position + 1}"


def _safe_filename(value: str) -> str:
    safe = re.sub(r'[\\/*?:"<>|]', "", str(value or "")).strip().rstrip(".")
    return safe[:80] or "知识卡片"


def _build_index_entry(card: KnowledgeCard, source: CardSource, path: Path) -> Dict[str, Any]:
    return {
        "card_id": card.card_id,
        "logical_key": card.logical_key,
        "card_type": card.card_type,
        "title": card.title,
        "summary": card.summary,
        "tags": card.tags,
        "aliases": card.aliases,
        "retrieval_questions": card.retrieval_questions,
        "source_id": source.source_id,
        "source_title": source.title,
        "source_url": source.url,
        "source_srt": source.srt_file,
        "source_collection": source.collection,
        "parent_card_id": card.parent_card_id,
        "child_card_ids": card.child_card_ids,
        "path": path.relative_to(KNOWLEDGE_CARD_DIR).as_posix(),
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    }


def _index_entry_from_metadata(metadata: Dict[str, Any], path: Path) -> Dict[str, Any]:
    source = metadata.get("source") or {}
    return {
        "card_id": metadata.get("card_id") or "",
        "logical_key": metadata.get("logical_key") or "",
        "card_type": metadata.get("card_type") or "general_knowledge",
        "title": metadata.get("title") or path.stem,
        "summary": metadata.get("summary") or "",
        "tags": metadata.get("tags") or [],
        "aliases": metadata.get("aliases") or [],
        "retrieval_questions": metadata.get("retrieval_questions") or [],
        "source_id": source.get("source_id") or "",
        "source_title": source.get("video_title") or "",
        "source_url": source.get("url") or "",
        "source_srt": source.get("srt_file") or "",
        "source_collection": source.get("collection") or "",
        "parent_card_id": metadata.get("parent_card_id") or "",
        "child_card_ids": metadata.get("child_card_ids") or [],
        "path": path.relative_to(KNOWLEDGE_CARD_DIR).as_posix(),
        "created_at": metadata.get("created_at") or "",
        "updated_at": metadata.get("updated_at") or "",
    }


def _entry_path(entry: Dict[str, Any]) -> Optional[Path]:
    relative = str(entry.get("path") or "").strip()
    if not relative:
        return None
    candidate = (KNOWLEDGE_CARD_DIR / relative).resolve()
    root = KNOWLEDGE_CARD_DIR.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise KnowledgeCardError("知识卡片索引包含越界路径")
    return candidate


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("wb") as file:
            file.write(content)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _notify(
    progress: Optional[ProgressCallback],
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress:
        progress(stage, current, total, message)
