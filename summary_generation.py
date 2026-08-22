import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set


SUMMARY_PROFILES = {
    "roundup_digest": "资讯/栏目汇总",
    "tutorial_guide": "教程/行动指南",
    "review_comparison": "测评/对比",
    "knowledge_explainer": "知识讲解",
    "viewpoint_analysis": "观点分析",
    "narrative_interview": "叙事/访谈",
    "general_summary": "通用总结",
}

SUMMARY_PROFILE_DESCRIPTIONS = {
    "roundup_digest": "固定栏目、周报、新闻盘点、项目推荐、榜单，包含多个相互独立条目。",
    "tutorial_guide": "围绕一个目标给出操作步骤、流程或解决方案。",
    "review_comparison": "围绕产品、软件或方案进行体验、评测或横向比较。",
    "knowledge_explainer": "系统解释一个概念、机制、历史或科学问题。",
    "viewpoint_analysis": "围绕中心主张展开论证、评论或价值判断。",
    "narrative_interview": "以人物、事件进程、故事或访谈问答为主。",
    "general_summary": "确实无法归入以上类型时使用。",
}

DEFAULT_PROFILE = "general_summary"
CLASSIFICATION_MAX_TOKENS = 800
DEFAULT_SUMMARY_CHAR_LIMIT = 26_000
DEEPSEEK_V4_SUMMARY_CHAR_LIMIT = 600_000
MIN_SUMMARY_CHAR_LIMIT = 20_000
MAX_SUMMARY_CHAR_LIMIT = 600_000
SUMMARY_CHAR_LIMIT_STEP = 10_000
_TIMESTAMP_PATTERN = re.compile(r"\[(?:\d{1,3}:)?\d{1,2}:\d{2}\]")
_TRUNCATION_NOTICE = " ...(后续字幕已按设置的字符上限截断)"


@dataclass(frozen=True)
class SummaryRoute:
    profile: str
    label: str
    confidence: float = 0.0
    reason: str = ""
    source: str = "model"
    error: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptLimitResult:
    text: str
    limit: int
    original_chars: int
    retained_chars: int
    truncated: bool
    last_timestamp: str = ""


def recommended_summary_char_limit(settings: Optional[Dict[str, Any]] = None) -> int:
    model = str((settings or {}).get("model") or "").lower()
    if "deepseek" in model and "v4" in model:
        return DEEPSEEK_V4_SUMMARY_CHAR_LIMIT
    return DEFAULT_SUMMARY_CHAR_LIMIT


def normalize_summary_char_limit(
    value: Any,
    settings: Optional[Dict[str, Any]] = None,
) -> int:
    fallback = recommended_summary_char_limit(settings)
    if isinstance(value, bool):
        return fallback
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(MIN_SUMMARY_CHAR_LIMIT, min(MAX_SUMMARY_CHAR_LIMIT, normalized))


def resolve_summary_char_limit(settings: Optional[Dict[str, Any]] = None) -> int:
    active = settings or {}
    return normalize_summary_char_limit(active.get("summary_max_chars"), active)


def limit_transcript_to_complete_segments(
    transcript: str,
    limit: int,
) -> TranscriptLimitResult:
    text = str(transcript or "").strip()
    normalized_limit = normalize_summary_char_limit(limit, {})
    if len(text) <= normalized_limit:
        timestamps = list(_TIMESTAMP_PATTERN.finditer(text))
        return TranscriptLimitResult(
            text=text,
            limit=normalized_limit,
            original_chars=len(text),
            retained_chars=len(text),
            truncated=False,
            last_timestamp=timestamps[-1].group(0).strip("[]") if timestamps else "",
        )

    timestamps = list(_TIMESTAMP_PATTERN.finditer(text))
    complete_boundaries = [
        match.start()
        for match in timestamps[1:]
        if match.start() <= normalized_limit
    ]
    if complete_boundaries:
        cut_at = complete_boundaries[-1]
    else:
        cut_at = text.rfind(" ", 0, normalized_limit + 1)
        if cut_at <= 0:
            cut_at = normalized_limit

    retained = text[:cut_at].rstrip()
    retained_timestamps = list(_TIMESTAMP_PATTERN.finditer(retained))
    last_timestamp = (
        retained_timestamps[-1].group(0).strip("[]")
        if retained_timestamps
        else ""
    )
    return TranscriptLimitResult(
        text=retained + _TRUNCATION_NOTICE,
        limit=normalized_limit,
        original_chars=len(text),
        retained_chars=len(retained),
        truncated=True,
        last_timestamp=last_timestamp,
    )


def fallback_summary_route(error: Any) -> SummaryRoute:
    return SummaryRoute(
        profile=DEFAULT_PROFILE,
        label=SUMMARY_PROFILES[DEFAULT_PROFILE],
        source="fallback",
        error=str(error)[:500],
        reason="内容形式识别失败，使用通用总结兜底。",
    )


def classify_summary_profile(
    title: str,
    transcript: str,
    client: Any,
    settings: Dict[str, Any],
) -> SummaryRoute:
    messages = build_classification_messages(title, transcript)
    try:
        return _request_summary_route(client, settings, messages)
    except Exception as error:
        return fallback_summary_route(error)


def resolve_summary_baseline_route(
    title: str,
    transcript: str,
    current_profile: str,
    client: Any,
    settings: Dict[str, Any],
) -> SummaryRoute:
    if current_profile in SUMMARY_PROFILES:
        return SummaryRoute(
            profile=current_profile,
            label=SUMMARY_PROFILES[current_profile],
            confidence=1.0,
            reason="沿用现有总结记录中的板块。",
            source="existing_summary",
        )
    return classify_summary_profile(title, transcript, client, settings)


def classify_alternative_summary_profile(
    title: str,
    transcript: str,
    current_profile: str,
    client: Any,
    settings: Dict[str, Any],
) -> SummaryRoute:
    if current_profile not in SUMMARY_PROFILES:
        raise ValueError(f"无法识别当前总结类型: {current_profile or '空'}")
    messages = build_classification_messages(
        title,
        transcript,
        excluded_profiles={current_profile},
    )
    return _request_summary_route(
        client,
        settings,
        messages,
        excluded_profiles={current_profile},
        source="alternative_model",
    )


def _request_summary_route(
    client: Any,
    settings: Dict[str, Any],
    messages: List[Dict[str, str]],
    excluded_profiles: Optional[Set[str]] = None,
    source: str = "model",
) -> SummaryRoute:
    request = {
        "model": settings["model"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": CLASSIFICATION_MAX_TOKENS,
    }
    if supports_json_output(settings):
        request["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or ""
    payload = parse_json_object(content)
    profile = str(payload.get("profile") or "").strip()
    if profile not in SUMMARY_PROFILES:
        raise ValueError(f"未知总结类型: {profile or '空'}")
    if profile in (excluded_profiles or set()):
        raise ValueError(f"模型仍返回了已排除的总结类型: {profile}")
    return SummaryRoute(
        profile=profile,
        label=SUMMARY_PROFILES[profile],
        confidence=_bounded_confidence(payload.get("confidence")),
        reason=str(payload.get("reason") or "").strip()[:300],
        source=source,
    )


def build_classification_messages(
    title: str,
    transcript: str,
    excluded_profiles: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    excluded = {
        profile
        for profile in (excluded_profiles or [])
        if profile in SUMMARY_PROFILES
    }
    available_profiles = [
        profile for profile in SUMMARY_PROFILES if profile not in excluded
    ]
    if not available_profiles:
        raise ValueError("没有可用的替代总结类型")
    option_lines = "\n".join(
        f"- {profile}：{SUMMARY_PROFILE_DESCRIPTIONS[profile]}"
        for profile in available_profiles
    )
    excluded_instruction = ""
    if excluded:
        excluded_labels = "、".join(
            f"{profile}（{SUMMARY_PROFILES[profile]}）" for profile in sorted(excluded)
        )
        excluded_instruction = (
            f"当前总结已使用：{excluded_labels}。本次必须排除这些类型，"
            "从剩余选项中选择与视频组织形式最接近的唯一类型。\n"
        )
    priority_instruction = (
        "优先规则：只要视频按栏目连续介绍多个独立项目、资源或新闻事件，即使内容跨越多个行业，"
        "也必须选择roundup_digest，不要按其中某一条新闻的行业分类。\n"
        if "roundup_digest" in available_profiles
        else "优先规则：判断剩余类型中最接近原视频组织形式的一种，不要返回已排除类型。\n"
    )
    system = (
        "你是视频内容结构路由器。你判断的是视频的组织形式，而不是所属行业。"
        "只输出JSON对象，不要生成视频总结。"
    )
    user = (
        "请根据标题和完整字幕选择唯一的总结类型。\n"
        f"{excluded_instruction}"
        f"可选类型：\n{option_lines}\n\n"
        f"{priority_instruction}"
        "返回字段：profile、confidence（0到1）、reason。\n\n"
        f"【视频标题】{title}\n"
        f"【带时间戳字幕】\n{transcript}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_summary_messages(
    title: str,
    transcript: str,
    route: SummaryRoute,
) -> List[Dict[str, str]]:
    profile_instruction = _profile_instruction(route.profile)
    system = (
        "你是专业的视频内容编辑。请把带时间戳的ASR字幕整理为准确、完整、便于浏览的中文笔记。\n\n"
        f"【视频标题】{title}\n"
        f"【带时间戳字幕】\n{transcript}\n\n"
        "【事实约束】\n"
        "1. 只能把字幕中出现的内容写成视频事实；不要声称联网核验，不要补造链接、数据或观点。\n"
        "2. 所有要点和Markdown表格的数据行都要附真实时间戳，格式为(MM:SS)或(HH:MM:SS)。\n"
        "3. 时间段使用(开始-结束)，有小时位时必须保留小时位。\n"
        "4. 可以静默修正常见ASR错字，但不改变原意。"
    )
    user = (
        f"已识别的内容形式：{route.label}。\n"
        "请直接输出总结，不要复述任务要求，不要用‘好的’开场。\n"
        "优先保证完整覆盖和清晰导航；不要为了套模板制造原视频不存在的案例、金句或结论。\n\n"
        f"{profile_instruction}\n\n"
        "【通用排版要求】\n"
        "- 使用清晰的Markdown标题、短段落和列表。\n"
        "- 需要比较或密集展示同类条目时使用标准换行Markdown表格。\n"
        "- 每个独立条目只出现一次，相关信息放在同一处，避免跨章节重复。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _profile_instruction(profile: str) -> str:
    instructions = {
        "roundup_digest": (
            "按原视频的栏目边界和出场顺序整理，禁止为了追求所谓深度而打乱栏目。\n"
            "### 本期速览：概括有哪些栏目、多少个主要条目以及最值得关注的内容。\n"
            "### 项目与资源速览：如果存在软件、开源项目、网站、论文或产品推荐，必须逐项覆盖，"
            "优先用表格列出名称、平台/类型、解决的问题、核心亮点或使用门槛、适合对象、时间戳。\n"
            "### 科技事件与行业动态：按原节目中的主题栏目分组，每条独立新闻单独成点，保留关键数据、"
            "影响和作者明确表达的判断，不得把多条新闻合成一个泛泛结论。\n"
            "### 其他固定板块：继续覆盖免费游戏、活动、书影音推荐等未归入前两节的内容；没有则省略。\n"
            "### 值得关注的信号：最后只提炼2到5条跨条目的趋势或风险，并明确这是归纳，不替代前面的逐项索引。"
        ),
        "tutorial_guide": (
            "围绕用户要完成的目标组织内容。依次输出：目标与适用场景、前置条件、分步操作、"
            "关键参数或判断点、常见错误与风险、完成后的检查方法。步骤保持可执行，不能遗漏必要条件。"
        ),
        "review_comparison": (
            "先交代评测对象和结论，再用表格呈现核心参数、体验差异、优缺点和适用人群；随后说明测试场景、"
            "影响结论的限制、购买或选择建议。没有实测的数据不得写成实测结果。"
        ),
        "knowledge_explainer": (
            "依次输出：一句话解释、核心概念、底层机制或因果链、关键证据与案例、容易误解之处、"
            "适用边界与未解决问题。保持概念关系清楚，不要改写成操作教程。"
        ),
        "viewpoint_analysis": (
            "依次输出：核心主张、论证结构、主要证据与案例、隐含前提、适用边界、反方视角或争议点。"
            "严格区分作者明确观点与编辑归纳，不为作者补造立场。"
        ),
        "narrative_interview": (
            "保留人物和事件关系，依次输出：人物/背景、事件或访谈脉络、关键转折、重要原话或观点、"
            "结果与后续影响。访谈内容按主题聚合，但不能混淆不同说话者。"
        ),
        "general_summary": (
            "输出：核心摘要、内容脉络、关键事实与细节、作者观点或结论、风险和适用边界。"
            "内容存在清晰的原始章节时优先保留，不要强行重排。"
        ),
    }
    return instructions.get(profile, instructions[DEFAULT_PROFILE])


def supports_json_output(settings: Dict[str, Any]) -> bool:
    model = str(settings.get("model") or "").lower()
    return "deepseek" in model and "v4" in model


def parse_json_object(content: str) -> Dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ValueError("模型未返回分类结果")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("分类结果中没有有效JSON对象")


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
