import subprocess
import sys
import os
import urllib.request
import json
import time

print("="*50)
print("🎬 视听内容 AI 助手 - 总控司令部")
print("="*50)
print("正在启动所有微服务...\n")

processes = []

try:
    # 1. 启动 API 调度中心
    print("🚀 启动 [FastAPI 调度中心]...")
    p1 = subprocess.Popen([sys.executable, "server.py"])
    processes.append(p1)

    # 2. 启动后台 Worker
    print("🚀 启动 [Whisper 消费进程]...")
    p2 = subprocess.Popen([sys.executable, "worker.py"])
    processes.append(p2)

    # 3. 启动 Streamlit UI (放宽上传限制，以备不时之需)
    print("🚀 启动 [Streamlit 界面]...\n")
    p3 = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py", 
        "--server.maxUploadSize", "2048"
    ])
    processes.append(p3)

    print("✅ 系统全部启动完毕！浏览器即将自动打开。")
    
    # ================= 🚀 新增：黑框物理拖拽交互区 =================
    # 给服务 2 秒钟的启动缓冲时间
    time.sleep(2)
    
    print("\n" + "🌟 "*17)
    print("💡 隐藏彩蛋：支持物理级拖拽识别！")
    print("   你可以直接把本地视频/音频文件拖拽到这个黑框里，")
    print("   然后按下【回车键】，任务就会瞬间推送到网页端！")
    print("   (输入 'q' 退出系统)")
    print("🌟 "*17 + "\n")

    while True:
        # 阻塞式等待用户拖入文件
        user_input = input("👇 拖入文件并按回车 (或输入 q 退出):\n> ").strip()

        if user_input.lower() in ['q', 'exit', 'quit']:
            print("\n🛑 收到退出指令，准备关闭系统...")
            break

        if not user_input:
            continue

        # 自动清洗 Windows 拖拽时自带的双引号和单引号
        clean_path = user_input.strip('"').strip("'")

        if os.path.exists(clean_path):
            # 将路径打包，通过 HTTP POST 瞬间塞给本地的 FastAPI
            url = "http://127.0.0.1:8000/api/tasks"
            data = json.dumps({
                "source_type": "local_file",
                "source_path": clean_path,
                "title": os.path.splitext(os.path.basename(clean_path))[0]
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

            try:
                with urllib.request.urlopen(req) as response:
                    if response.status == 200:
                        print(f"✅ 捷径成功！[{os.path.basename(clean_path)}] 已瞬间推送到网页后台！请看网页！\n")
                    else:
                        print(f"❌ 捷径失败，服务器状态码: {response.status}\n")
            except Exception as e:
                print(f"❌ 无法连接到本地调度中心: {e}\n")
        else:
            print("❌ 读取失败：找不到该文件，请确认是否拖拽正确。\n")

except KeyboardInterrupt:
    print("\n🛑 收到强行中断指令 (Ctrl+C)...")
finally:
    # 无论如何，保证销毁子进程，不留僵尸程序
    print("正在清理并关闭所有后台微服务...")
    for p in processes:
        p.terminate()
    print("清理完毕，再见！")