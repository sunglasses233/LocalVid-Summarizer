from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import db

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

# ================= API 路由 =================
@app.post("/api/tasks")
def add_task(task: TaskCreate):
    if task.source_type not in ["url", "local_file"]:
        raise HTTPException(status_code=400, detail="非法的 source_type，必须为 url 或 local_file")
    
    try:
        task_id = db.create_task(task.source_type, task.source_path, task.title)
        return {"status": "success", "task_id": task_id, "message": "任务已成功加入队列"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库写入失败: {str(e)}")

@app.get("/api/tasks")
def list_tasks():
    try:
        tasks = db.get_all_tasks()
        return {"status": "success", "data": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库读取失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)