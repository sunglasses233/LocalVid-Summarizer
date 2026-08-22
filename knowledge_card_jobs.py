import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import KNOWLEDGE_CARD_DIR
from knowledge_cards import (
    CARD_TYPES,
    CardSource,
    generate_card_bundle,
    now_china_iso,
)


ACTIVE_JOB_STATUSES = {"queued", "running"}
STATUS_WRITE_MAX_ATTEMPTS = 5
STATUS_WRITE_RETRY_SECONDS = 0.05


def _load_llm_settings() -> Dict[str, Any]:
    from llm_settings import load_llm_settings

    return load_llm_settings()


def _create_llm_client(settings: Dict[str, Any]) -> Any:
    from llm_settings import create_llm_client

    return create_llm_client(settings)


class KnowledgeCardJobManager:
    def __init__(
        self,
        status_dir: Optional[Path] = None,
        max_workers: int = 1,
        generate_func: Callable[..., Any] = generate_card_bundle,
        settings_loader: Callable[[], Dict[str, Any]] = _load_llm_settings,
        client_factory: Callable[[Dict[str, Any]], Any] = _create_llm_client,
    ):
        self.status_dir = Path(status_dir or (KNOWLEDGE_CARD_DIR / "_jobs"))
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self._generate_func = generate_func
        self._settings_loader = settings_loader
        self._client_factory = client_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="knowledge-card",
        )
        self._lock = threading.RLock()

    def submit(
        self,
        source: CardSource,
        forced_type: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        if not source.source_id.strip():
            raise ValueError("知识卡片任务缺少 source_id")
        if not source.transcript.strip():
            raise ValueError("知识卡片任务缺少完整字幕")
        if forced_type and forced_type not in CARD_TYPES:
            raise ValueError(f"不支持的知识卡片类型: {forced_type}")

        with self._lock:
            existing = self.get(source.source_id)
            if existing and existing.get("status") in ACTIVE_JOB_STATUSES:
                return existing, False

            timestamp = now_china_iso()
            status = {
                "job_id": uuid.uuid4().hex,
                "source_id": source.source_id,
                "source_title": source.title,
                "forced_type": forced_type or "",
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "current": 0,
                "total": 1,
                "message": "已加入知识卡片后台队列",
                "card_count": 0,
                "error": "",
                "created_at": timestamp,
                "updated_at": timestamp,
                "finished_at": "",
            }
            self._write_status(status)
            try:
                future = self._executor.submit(
                    self._run_job,
                    source,
                    forced_type,
                    status["job_id"],
                )
            except Exception as error:
                status.update(
                    {
                        "status": "failed",
                        "stage": "failed",
                        "message": "知识卡片后台任务启动失败",
                        "error": str(error),
                        "updated_at": now_china_iso(),
                        "finished_at": now_china_iso(),
                    }
                )
                self._write_status(status)
                raise
            future.add_done_callback(
                lambda completed, source_id=source.source_id, job_id=status["job_id"]: (
                    self._handle_future_done(source_id, job_id, completed)
                )
            )
            return dict(status), True

    def get(self, source_id: str) -> Optional[Dict[str, Any]]:
        safe_id = self._safe_source_id(source_id)
        path = self.status_dir / f"{safe_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def list(self, active_only: bool = False, limit: int = 20) -> List[Dict[str, Any]]:
        statuses = []
        for path in self.status_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if active_only and data.get("status") not in ACTIVE_JOB_STATUSES:
                continue
            statuses.append(data)
        statuses.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return statuses[: max(1, int(limit))]

    def recover_interrupted(self) -> int:
        recovered = 0
        with self._lock:
            for status in self.list(active_only=True, limit=10_000):
                status.update(
                    {
                        "status": "interrupted",
                        "stage": "interrupted",
                        "message": "服务重启导致任务中断，请重新提交",
                        "error": "知识卡片后台服务已重启",
                        "updated_at": now_china_iso(),
                        "finished_at": now_china_iso(),
                    }
                )
                self._write_status(status)
                recovered += 1
        return recovered

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_job(
        self,
        source: CardSource,
        forced_type: Optional[str],
        job_id: str,
    ) -> None:
        def progress(stage: str, current: int, total: int, message: str) -> None:
            self._update_status(
                source.source_id,
                job_id,
                status="running",
                stage=stage,
                progress=self._progress_percent(stage, current, total),
                current=current,
                total=max(1, total),
                message=message,
            )

        try:
            started_status = self._update_status(
                source.source_id,
                job_id,
                status="running",
                stage="starting",
                progress=2,
                message="后台任务已启动，正在连接模型",
            )
            if not started_status or started_status.get("job_id") != job_id:
                return

            settings = self._settings_loader()
            client = self._client_factory(settings)
            bundle = self._generate_func(
                client,
                settings,
                source,
                forced_type=forced_type,
                progress=progress,
            )
            self._update_status(
                source.source_id,
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                current=1,
                total=1,
                message=f"知识卡片草稿生成完成，共 {len(bundle.cards)} 张",
                card_count=len(bundle.cards),
                error="",
                finished_at=now_china_iso(),
            )
        except Exception as error:
            try:
                self._update_status(
                    source.source_id,
                    job_id,
                    status="failed",
                    stage="failed",
                    message="知识卡片后台生成失败，原草稿未被覆盖",
                    error=str(error),
                    finished_at=now_china_iso(),
                )
            except Exception as status_error:
                raise RuntimeError(
                    f"知识卡片任务失败，且状态写入失败: {status_error}"
                ) from error

    def _handle_future_done(self, source_id: str, job_id: str, future: Any) -> None:
        if future.cancelled():
            error = RuntimeError("知识卡片后台线程在执行前被取消")
        else:
            try:
                error = future.exception()
            except Exception as future_error:
                error = future_error
        if error is None:
            return

        print(f"[知识卡片任务 {source_id}] 后台线程异常: {error}")
        try:
            self._update_status(
                source_id,
                job_id,
                status="failed",
                stage="failed",
                message="知识卡片后台线程异常，原草稿未被覆盖",
                error=str(error),
                finished_at=now_china_iso(),
            )
        except Exception as status_error:
            print(
                f"[知识卡片任务 {source_id}] 无法写入失败状态: "
                f"{status_error}"
            )

    def _update_status(self, source_id: str, job_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self.get(source_id)
            if not status or status.get("job_id") != job_id:
                return status
            status.update(changes)
            status["updated_at"] = now_china_iso()
            self._write_status(status)
            return status

    def _write_status(self, status: Dict[str, Any]) -> None:
        source_id = self._safe_source_id(status.get("source_id") or "")
        path = self.status_dir / f"{source_id}.json"
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(status, ensure_ascii=False, indent=2)
        try:
            temp_path.write_text(payload, encoding="utf-8")
            for attempt in range(1, STATUS_WRITE_MAX_ATTEMPTS + 1):
                try:
                    os.replace(temp_path, path)
                    break
                except OSError as error:
                    is_last_attempt = attempt >= STATUS_WRITE_MAX_ATTEMPTS
                    if is_last_attempt or not self._is_retryable_replace_error(error):
                        raise
                    time.sleep(STATUS_WRITE_RETRY_SECONDS * attempt)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _safe_source_id(source_id: str) -> str:
        safe = "".join(character for character in str(source_id) if character.isalnum() or character in "_-")
        return safe or "unknown"

    @staticmethod
    def _is_retryable_replace_error(error: OSError) -> bool:
        return isinstance(error, PermissionError) or getattr(error, "winerror", None) in {
            5,
            32,
            33,
        }

    @staticmethod
    def _progress_percent(stage: str, current: int, total: int) -> int:
        if stage == "extracting":
            return min(70, 5 + int(62 * current / max(1, total)))
        return {
            "generating": 10,
            "merging": 72,
            "validating": 88,
            "draft_saved": 96,
        }.get(stage, 5)


knowledge_card_job_manager = KnowledgeCardJobManager()
