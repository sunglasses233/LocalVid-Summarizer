from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import db
from config import API_HOST, API_PORT
from typing import Optional

from knowledge_card_jobs import knowledge_card_job_manager
from knowledge_cards import CardSource
from processing_settings import merge_processing_defaults

# ================= 生命周期管理 (现代 FastAPI 写法) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    db.init_db()
    db.reset_zombie_tasks()
    interrupted_card_jobs = knowledge_card_job_manager.recover_interrupted()
    print("✅ 调度中心已启动，数据库初始化完成。僵尸任务已重置。")
    if interrupted_card_jobs:
        print(f"⚠️ 已将 {interrupted_card_jobs} 个未完成的知识卡片任务标记为中断。")
    yield
    # 关闭时执行 (未来可在这里添加释放资源的代码)
    print("🛑 调度中心已安全关闭。")

app = FastAPI(
    title="Video Task Scheduler API", 
    description="视听内容处理调度中心",
    lifespan=lifespan
)

# ================= 中间件配置 =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 数据验证模型 =================
class TaskCreate(BaseModel):
    source_type: str
    source_path: str
    title: str
    # 【核心修复 1】：必须明确声明 options，否则 FastAPI 会丢弃前端传来的该字段
    options: Optional[dict] = None


class KnowledgeCardJobCreate(BaseModel):
    source: dict
    forced_type: Optional[str] = None

# ================= API 路由 =================
@app.post("/api/tasks")
def add_task(task: TaskCreate):
    # 包含了 direct_url 兼容
    if task.source_type not in ["url", "local_file", "direct_url"]:
        raise HTTPException(status_code=400, detail="非法的 source_type")
    
    try:
        # 创建时固化全局处理设置，后续修改不会影响已经排队的任务。
        task_options = merge_processing_defaults(task.options)
        task_id = db.create_task(
            task.source_type,
            task.source_path,
            task.title,
            task_options,
        )
        return {"status": "success", "task_id": task_id, "message": "任务已成功加入队列"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库写入失败: {str(e)}")


@app.post("/api/knowledge-card-jobs")
def add_knowledge_card_job(job: KnowledgeCardJobCreate):
    try:
        source = CardSource.from_dict(job.source)
        status, created = knowledge_card_job_manager.submit(source, job.forced_type)
        return {
            "status": "accepted" if created else "already_running",
            "job": status,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"知识卡片任务提交失败: {error}") from error


@app.get("/api/knowledge-card-jobs")
def list_knowledge_card_jobs(active_only: bool = False, limit: int = 20):
    return {
        "jobs": knowledge_card_job_manager.list(active_only=active_only, limit=limit),
    }


@app.get("/api/knowledge-card-jobs/{source_id}")
def get_knowledge_card_job(source_id: str):
    status = knowledge_card_job_manager.get(source_id)
    if not status:
        raise HTTPException(status_code=404, detail="未找到知识卡片任务")
    return {"job": status}

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
