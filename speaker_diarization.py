import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "speaker_diarization"
DEFAULT_OUTPUT_DIR = BASE_DIR / "实验输出"
SEGMENTATION_MODEL_NAME = "segmentation.onnx"
EMBEDDING_MODEL_NAME = "speaker_embedding.onnx"
OTHER_SOUND_LABEL = "其他声音"
MIN_INTERNAL_CLUSTERS = 8
AI_HISTORY_MARKER = "====================== AI_CHAT_HISTORY ======================"
CHINA_TZ = timezone(timedelta(hours=8))
SRT_TIMELINE_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})$"
)


class DiarizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPaths:
    segmentation: Path
    embedding: Path


@dataclass(frozen=True)
class SpeakerSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    timeline: str
    text: str


def now_china_iso() -> str:
    return datetime.now(CHINA_TZ).isoformat(timespec="seconds")


def validate_model_files(model_dir: Path | str) -> ModelPaths:
    root = Path(model_dir).expanduser().resolve()
    paths = ModelPaths(
        segmentation=root / SEGMENTATION_MODEL_NAME,
        embedding=root / EMBEDDING_MODEL_NAME,
    )
    missing = [path.name for path in asdict(paths).values() if not Path(path).is_file()]
    if missing:
        raise DiarizationError(
            "缺少说话人分离模型："
            + "、".join(missing)
            + f"。请放入目录：{root}"
        )
    empty = [
        path.name
        for path in (paths.segmentation, paths.embedding)
        if path.stat().st_size <= 0
    ]
    if empty:
        raise DiarizationError("模型文件为空：" + "、".join(empty))
    return paths


def import_sherpa_onnx():
    try:
        import sherpa_onnx
    except ImportError as error:
        raise DiarizationError(
            "未安装 sherpa-onnx。请使用项目 .venv 中的Python运行实验。"
        ) from error
    return sherpa_onnx


def build_diarizer(
    model_paths: ModelPaths,
    num_speakers: int = -1,
    threshold: float = 0.5,
    num_threads: int = 4,
):
    sherpa_onnx = import_sherpa_onnx()
    segmentation = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(model_paths.segmentation)
        ),
        num_threads=max(1, int(num_threads)),
        debug=False,
        provider="cpu",
    )
    embedding = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_paths.embedding),
        num_threads=max(1, int(num_threads)),
        debug=False,
        provider="cpu",
    )
    clustering = sherpa_onnx.FastClusteringConfig(
        num_clusters=int(num_speakers),
        threshold=float(threshold),
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        clustering=clustering,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise DiarizationError("说话人分离配置校验失败，请检查模型是否与sherpa-onnx兼容。")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def extract_mono_wav(media_path: Path, wav_path: Path, sample_rate: int = 16000) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise DiarizationError("未找到ffmpeg，无法从视频提取音频。") from error
    if completed.returncode != 0 or not wav_path.is_file():
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        raise DiarizationError(f"音频提取失败：{detail[-500:]}")


def read_pcm16_mono(wav_path: Path, expected_sample_rate: int) -> array:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (OSError, wave.Error) as error:
        raise DiarizationError(f"无法读取临时音频：{error}") from error

    if channels != 1 or sample_width != 2:
        raise DiarizationError("临时音频必须是单声道16位PCM格式。")
    if sample_rate != expected_sample_rate:
        raise DiarizationError(
            f"音频采样率为{sample_rate}Hz，模型要求{expected_sample_rate}Hz。"
        )

    pcm = array("h")
    pcm.frombytes(frames)
    if sys.byteorder != "little":
        pcm.byteswap()
    return array("f", (sample / 32768.0 for sample in pcm))


def compact_speaker_labels(raw_speakers: Iterable[object]) -> List[str]:
    label_map = {}
    labels = []
    for raw_speaker in raw_speakers:
        key = str(raw_speaker)
        if key not in label_map:
            label_map[key] = f"说话人{len(label_map) + 1}"
        labels.append(label_map[key])
    return labels


def internal_cluster_count(primary_speaker_count: int) -> int:
    if primary_speaker_count < 0:
        return -1
    return max(MIN_INTERNAL_CLUSTERS, int(primary_speaker_count))


def assign_primary_speaker_labels(
    raw_segments: Sequence[Tuple[object, float, float]],
    primary_speaker_count: int,
) -> List[str]:
    if primary_speaker_count < 0:
        return compact_speaker_labels(segment[0] for segment in raw_segments)

    durations = {}
    first_seen = {}
    for position, (raw_speaker, start, end) in enumerate(raw_segments):
        key = str(raw_speaker)
        durations[key] = durations.get(key, 0.0) + max(0.0, float(end) - float(start))
        first_seen.setdefault(key, position)

    ranked_keys = sorted(
        durations,
        key=lambda key: (-durations[key], first_seen[key]),
    )
    primary_keys = set(ranked_keys[:primary_speaker_count])
    ordered_primary_keys = sorted(primary_keys, key=lambda key: first_seen[key])
    primary_labels = {
        key: f"说话人{index + 1}"
        for index, key in enumerate(ordered_primary_keys)
    }
    return [
        primary_labels.get(str(raw_speaker), OTHER_SOUND_LABEL)
        for raw_speaker, _, _ in raw_segments
    ]


def diarize_media(
    media_path: Path | str,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    num_speakers: int = -1,
    threshold: float = 0.5,
    num_threads: int = 4,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[List[SpeakerSegment], str]:
    source = Path(media_path).expanduser().resolve()
    if not source.is_file():
        raise DiarizationError(f"找不到输入视频或音频：{source}")
    models = validate_model_files(model_dir)
    cluster_count = internal_cluster_count(num_speakers)
    diarizer = build_diarizer(models, cluster_count, threshold, num_threads)
    sample_rate = int(diarizer.sample_rate)

    with tempfile.TemporaryDirectory(prefix="speaker_diarization_") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        print("[1/3] 正在提取16kHz单声道音频...", flush=True)
        extract_mono_wav(source, wav_path, sample_rate)
        print("[2/3] 正在分析说话人，请稍候...", flush=True)
        segments = _run_diarizer_on_wav(
            wav_path,
            diarizer,
            num_speakers,
            progress,
        )

    _print_diarization_summary(segments)
    return segments, str(getattr(import_sherpa_onnx(), "__version__", "unknown"))


def diarize_audio_file(
    audio_path: Path | str,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    num_speakers: int = -1,
    threshold: float = 0.5,
    num_threads: int = 4,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[List[SpeakerSegment], str]:
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise DiarizationError(f"找不到待分析音频：{source}")
    models = validate_model_files(model_dir)
    cluster_count = internal_cluster_count(num_speakers)
    diarizer = build_diarizer(models, cluster_count, threshold, num_threads)
    segments = _run_diarizer_on_wav(
        source,
        diarizer,
        num_speakers,
        progress,
    )
    _print_diarization_summary(segments)
    return segments, str(getattr(import_sherpa_onnx(), "__version__", "unknown"))


def _run_diarizer_on_wav(
    wav_path: Path,
    diarizer,
    primary_speaker_count: int,
    progress: Optional[Callable[[int], None]] = None,
) -> List[SpeakerSegment]:
    sample_rate = int(diarizer.sample_rate)
    samples = read_pcm16_mono(wav_path, sample_rate)
    last_percent = [-1]

    def report_progress(processed_chunks, num_chunks):
        if num_chunks <= 0:
            return 0
        percent = min(100, int(processed_chunks * 100 / num_chunks))
        if percent >= last_percent[0] + 5:
            last_percent[0] = percent
            print(f"      说话人分析进度：{percent}%", flush=True)
            if progress:
                progress(percent)
        return 0

    result = diarizer.process(samples, report_progress)
    raw_segments = [
        segment
        for segment in result.sort_by_start_time()
        if float(segment.end) > float(segment.start)
    ]
    raw_segment_data = [
        (segment.speaker, float(segment.start), float(segment.end))
        for segment in raw_segments
    ]
    compact_labels = assign_primary_speaker_labels(
        raw_segment_data,
        primary_speaker_count,
    )
    return [
        SpeakerSegment(
            speaker=label,
            start=round(float(segment.start), 3),
            end=round(float(segment.end), 3),
        )
        for segment, label in zip(raw_segments, compact_labels)
    ]


def _print_diarization_summary(segments: Sequence[SpeakerSegment]) -> None:
    detected_speakers = len(
        {
            segment.speaker
            for segment in segments
            if segment.speaker != OTHER_SOUND_LABEL
        }
    )
    other_sound_seconds = sum(
        segment.duration
        for segment in segments
        if segment.speaker == OTHER_SOUND_LABEL
    )
    other_sound_text = (
        f"，其他声音约{other_sound_seconds:.1f}秒"
        if other_sound_seconds > 0
        else ""
    )
    print(
        f"[3/3] 分析完成：{detected_speakers}位主要说话人，"
        f"{len(segments)}个片段{other_sound_text}。",
        flush=True,
    )


def timestamp_to_seconds(value: str) -> float:
    normalized = value.replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def seconds_to_timestamp(value: float) -> str:
    total_milliseconds = max(0, round(float(value) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt(content: str) -> List[SubtitleCue]:
    clean_content = str(content or "").split(AI_HISTORY_MARKER, 1)[0].strip()
    if not clean_content:
        return []
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", clean_content):
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 3 or not lines[0].strip().isdigit():
            continue
        timeline = lines[1].strip()
        match = SRT_TIMELINE_PATTERN.match(timeline)
        if not match:
            continue
        cues.append(
            SubtitleCue(
                index=int(lines[0].strip()),
                start=timestamp_to_seconds(match.group("start")),
                end=timestamp_to_seconds(match.group("end")),
                timeline=timeline.replace(".", ","),
                text="\n".join(lines[2:]).strip(),
            )
        )
    return cues


def interval_overlap(start: float, end: float, segment: SpeakerSegment) -> float:
    return max(0.0, min(end, segment.end) - max(start, segment.start))


def choose_speaker_for_cue(
    cue: SubtitleCue,
    segments: Sequence[SpeakerSegment],
    nearest_tolerance: float = 1.5,
) -> str:
    if not segments:
        return "说话人未知"
    overlaps = [(interval_overlap(cue.start, cue.end, segment), segment) for segment in segments]
    overlap, best_segment = max(overlaps, key=lambda item: item[0])
    if overlap > 0:
        return best_segment.speaker

    midpoint = (cue.start + cue.end) / 2
    nearest = min(
        segments,
        key=lambda segment: min(
            abs(midpoint - segment.start),
            abs(midpoint - segment.end),
        ),
    )
    distance = min(abs(midpoint - nearest.start), abs(midpoint - nearest.end))
    return nearest.speaker if distance <= nearest_tolerance else "说话人未知"


def label_subtitle_records(
    subtitles: Sequence[Dict[str, object]],
    segments: Sequence[SpeakerSegment],
) -> List[Dict[str, object]]:
    labeled = []
    for position, subtitle in enumerate(subtitles, 1):
        copied = dict(subtitle)
        try:
            start = timestamp_to_seconds(str(copied.get("start_time") or ""))
            end = timestamp_to_seconds(str(copied.get("end_time") or ""))
        except (TypeError, ValueError) as error:
            raise DiarizationError(
                f"第{position}条字幕时间轴无效，无法匹配说话人。"
            ) from error
        if end < start:
            raise DiarizationError(f"第{position}条字幕结束时间早于开始时间。")
        cue = SubtitleCue(
            index=position,
            start=start,
            end=end,
            timeline="",
            text=str(copied.get("text") or ""),
        )
        copied["speaker"] = choose_speaker_for_cue(cue, segments)
        labeled.append(copied)
    return labeled


def render_speaker_srt(
    cues: Sequence[SubtitleCue],
    segments: Sequence[SpeakerSegment],
) -> str:
    blocks = []
    for output_index, cue in enumerate(cues, 1):
        speaker = choose_speaker_for_cue(cue, segments)
        blocks.append(
            f"{output_index}\n{cue.timeline}\n[{speaker}] {cue.text}".rstrip()
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_segments_as_srt(segments: Sequence[SpeakerSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        timeline = (
            f"{seconds_to_timestamp(segment.start)} --> "
            f"{seconds_to_timestamp(segment.end)}"
        )
        blocks.append(f"{index}\n{timeline}\n[{segment.speaker}]")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_outputs(
    media_path: Path,
    segments: Sequence[SpeakerSegment],
    runtime_version: str,
    output_dir: Path,
    source_srt: Optional[Path] = None,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{media_path.stem}_说话人.json"
    srt_path = output_dir / f"{media_path.stem}_说话人.srt"
    payload = {
        "schema_version": 1,
        "created_at": now_china_iso(),
        "source_media": str(media_path),
        "runtime": {"name": "sherpa-onnx", "version": runtime_version},
        "speaker_count": len(
            {
                segment.speaker
                for segment in segments
                if segment.speaker != OTHER_SOUND_LABEL
            }
        ),
        "has_other_sound": any(
            segment.speaker == OTHER_SOUND_LABEL for segment in segments
        ),
        "segments": [asdict(segment) for segment in segments],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if source_srt and source_srt.is_file():
        cues = parse_srt(source_srt.read_text(encoding="utf-8"))
        if not cues:
            raise DiarizationError(f"未能从字幕文件解析出有效片段：{source_srt}")
        output_srt = render_speaker_srt(cues, segments)
    else:
        output_srt = render_segments_as_srt(segments)
    srt_path.write_text(output_srt, encoding="utf-8")
    return json_path, srt_path


def parse_num_speakers(value: str) -> int:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"", "auto", "自动", "-1"}:
        return -1
    try:
        number = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("说话人数必须是auto或2以上的整数") from error
    if number < 2:
        raise argparse.ArgumentTypeError("说话人数必须是auto或2以上的整数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="纯本地多人对话说话人分离实验")
    parser.add_argument("input", help="本地视频或音频路径")
    parser.add_argument("--srt", help="可选的原始SRT；默认寻找视频同名SRT")
    parser.add_argument(
        "--models",
        default=str(DEFAULT_MODEL_DIR),
        help="模型目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="实验输出目录",
    )
    parser.add_argument(
        "--speakers",
        type=parse_num_speakers,
        default=-1,
        help="主要说话人数；内部会额外保留音乐和噪声候选，使用auto完全自动推断",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="自动聚类阈值")
    parser.add_argument("--threads", type=int, default=4, help="CPU线程数")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    media_path = Path(args.input).expanduser().resolve()
    source_srt = Path(args.srt).expanduser().resolve() if args.srt else None
    if source_srt is None:
        same_name_srt = media_path.with_suffix(".srt")
        source_srt = same_name_srt if same_name_srt.is_file() else None
    try:
        segments, runtime_version = diarize_media(
            media_path,
            model_dir=args.models,
            num_speakers=args.speakers,
            threshold=args.threshold,
            num_threads=args.threads,
        )
        json_path, srt_path = write_outputs(
            media_path,
            segments,
            runtime_version,
            Path(args.output_dir).expanduser().resolve(),
            source_srt,
        )
    except DiarizationError as error:
        print(f"实验失败：{error}", file=sys.stderr)
        return 1
    print(f"JSON：{json_path}")
    print(f"字幕：{srt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
