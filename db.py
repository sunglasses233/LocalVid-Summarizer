import sqlite3
import uuid
from typing import List, Dict, Any

DB_PATH = "tasks.db"

def get_conn() -> sqlite3.Connection:
    """获取数据库连接，配置为字典工厂以便直接返回 JSON 友好的数据"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """初始化数据库表结构"""
    with get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS video_tasks (
                id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

def reset_zombie_tasks() -> None:
    """系统重启时，将进行中（死掉）的任务重置为排队状态"""
    with get_conn() as conn:
        conn.execute('''
            UPDATE video_tasks 
            SET status = 'pending', progress = 0 
            WHERE status IN ('downloading', 'transcribing')
        ''')

def create_task(source_type: str, source_path: str, title: str) -> str:
    """创建一个新任务进入队列"""
    task_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO video_tasks (id, source_type, source_path, title, status) VALUES (?, ?, ?, ?, 'pending')",
            (task_id, source_type, source_path, title)
        )
    return task_id

def get_all_tasks() -> List[Dict[str, Any]]:
    """获取所有任务列表，按时间倒序排列"""
    with get_conn() as conn:
        cursor = conn.execute("SELECT * FROM video_tasks ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def update_task_status(task_id: str, status: str, progress: int = 0) -> None:
    """供未来的 Worker 节点调用，更新任务进度"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE video_tasks SET status = ?, progress = ? WHERE id = ?",
            (status, progress, task_id)
        )