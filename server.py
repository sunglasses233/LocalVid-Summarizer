from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import db
from config import API_HOST, API_PORT
from typing import Optional

# ================= 生命周期管理 (现代 FastAPI 写法) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    db.init_db()
    db.reset_zombie_tasks()
    print("✅ 调度中心已启动，数据库初始化完成。僵尸任务已重置。")
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
    options: Optional[dict] = {} 

# ================= API 路由 =================
@app.post("/api/tasks")
def add_task(task: TaskCreate):
    # 包含了 direct_url 兼容
    if task.source_type not in ["url", "local_file", "direct_url"]:
        raise HTTPException(status_code=400, detail="非法的 source_type")
    
    try:
        # 【核心修复 2】：将 task.options 准确透传给数据库引擎
        task_id = db.create_task(task.source_type, task.source_path, task.title, task.options)
        return {"status": "success", "task_id": task_id, "message": "任务已成功加入队列"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库写入失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)