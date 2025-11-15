# api_server.py
import asyncio
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional
import re
import socketio
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv # [新增] 导入 dotenv

# --- 全局设置 ---
# [新增] 在程序开始时加载 .env 文件中的环境变量
load_dotenv()

# 创建一个目录用于存放上传的临时文件
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# --- FastAPI 和 Socket.IO 应用设置 ---
app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

client_tasks = {}

# --- 主页路由 ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>错误：index.html 未找到</h1>", status_code=404)

# --- 核心业务逻辑 ---
async def run_main_script(sid: str, pdf_path: str, params: dict):
    temp_upload_dir = Path(pdf_path).parent
    
    try:
        # [修改] 构建命令，包含所有从前端接收的参数
        command = [
            "python", "main.py",
            "--pdf", pdf_path,
            "--tier", params['tier']
        ]
        if params['force']:
            command.append("--force")
        
        # 将所有可选配置作为命令行参数传递
        for key, value in params.items():
            # tier 和 force 已经处理过，跳过
            if key in ['tier', 'force']:
                continue
            if value is not None:
                command.extend([f"--{key.replace('_', '-')}", str(value)])

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        async def stream_logs(stream, stream_name):
            # [核心修改] 新增信令检测逻辑
            report_signal_pattern = re.compile(r"\[REPORT_READY\](reviewer1|reviewer2):(.+)")

            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace').strip()
                
                # 检查是否是报告信令
                match = report_signal_pattern.match(line_str)
                if match:
                    report_type = match.group(1) # 'reviewer1' or 'reviewer2'
                    report_path_str = match.group(2).strip()
                    report_path = Path(report_path_str)
                    
                    print(f"[SID: {sid}] Detected signal for {report_type} report at {report_path}")
                    if report_path.exists():
                        try:
                            content = report_path.read_text(encoding='utf-8')
                            await sio.emit(f'{report_type}_report', {'content': content}, to=sid)
                        except Exception as e:
                            print(f"[SID: {sid}] Error reading report file {report_path}: {e}")
                    else:
                         print(f"[SID: {sid}] Report file not found at signaled path: {report_path}")

                    continue # 信令本身不作为日志发送给前端

                # 如果不是信令，则作为普通日志处理
                print(f"[SID: {sid}] {stream_name}: {line_str}")
                await sio.emit('log', {'data': line_str}, to=sid)
                
                # 状态更新逻辑 (保持不变)
                if "🚀 开始执行: Step 1" in line_str:
                    await sio.emit('status_update', {'step': 'step1', 'status': 'running'}, to=sid)
                elif "✅ Step 1" in line_str:
                    await sio.emit('status_update', {'step': 'step1', 'status': 'success'}, to=sid)
                elif "🚀 开始执行: Step 2" in line_str:
                    await sio.emit('status_update', {'step': 'step2', 'status': 'running'}, to=sid)
                elif "✅ Step 2" in line_str:
                    await sio.emit('status_update', {'step': 'step2', 'status': 'success'}, to=sid)
                elif "🚀 开始并行执行审稿分支" in line_str:
                    await sio.emit('status_update', {'step': 'parallel', 'status': 'running'}, to=sid)
                elif "✅ 所有并行审稿分支执行完毕" in line_str:
                    await sio.emit('status_update', {'step': 'parallel', 'status': 'success'}, to=sid)
                elif "🚀 开始执行: Meta Reviewer" in line_str:
                    await sio.emit('status_update', {'step': 'meta', 'status': 'running'}, to=sid)
                elif "✅ Meta Reviewer" in line_str or "🎉 Meta Reviewer 全部流程成功！" in line_str:
                    await sio.emit('status_update', {'step': 'meta', 'status': 'success'}, to=sid)

        await asyncio.gather(
            stream_logs(process.stdout, "LOG"),
            stream_logs(process.stderr, "ERROR")
        )

        await process.wait()

        if process.returncode == 0:
            try:
                # 这里的逻辑需要动态确定最终报告的路径
                # 假设 main.py 在成功时会打印出最终报告的路径
                # 为简化，我们先基于 paper_name 构建路径
                paper_name = Path(pdf_path).stem
                # 注意：这个路径需要和 main.py 中的输出路径一致
                # 我们需要从 config.ini 读取 OUTPUT_BASE_DIR
                import configparser
                config = configparser.ConfigParser()
                config.read('config.ini')
                output_base_dir = Path(config['PATHS']['OUTPUT_BASE_DIR'])
                report_path = output_base_dir / f"{paper_name}_output" / f"{paper_name}_meta_review.md"

                if report_path.exists():
                    report_content = report_path.read_text(encoding='utf-8')
                    await sio.emit('final_review', {'content': report_content}, to=sid)
                    await sio.emit('done', {'message': '🎉 流水线成功结束！最终报告已生成。'}, to=sid)
                else:
                    error_msg = f"错误：流水线声称成功，但未找到最终报告文件于 {report_path}"
                    await sio.emit('log', {'data': error_msg}, to=sid)
                    await sio.emit('error', {'message': error_msg}, to=sid)

            except Exception as e:
                error_msg = f"错误：读取最终报告文件时出错: {e}"
                await sio.emit('log', {'data': error_msg}, to=sid)
                await sio.emit('error', {'message': error_msg}, to=sid)
        else:
            await sio.emit('error', {'message': f'❌ 流水线执行失败，请检查日志获取详细信息。'}, to=sid)
    
    finally:
        if temp_upload_dir.exists():
            try:
                shutil.rmtree(temp_upload_dir)
                print(f"[SID: {sid}] 已清理临时上传目录: {temp_upload_dir}")
            except Exception as e:
                print(f"[SID: {sid}] 清理临时目录失败: {e}")
        
        if sid in client_tasks:
            del client_tasks[sid]

# --- API Endpoints ---
@app.post("/api/review")
async def start_review(
    sid: str = Form(...),
    tier: str = Form(...),
    force: str = Form(...), 
    pdf_file: UploadFile = File(...),
    # [新增] 接收所有高级设置参数
    max_papers_frontier: Optional[int] = Form(None),
    max_papers_openreview: Optional[int] = Form(None),
    relevance_threshold: Optional[float] = Form(None),
    similarity_threshold: Optional[float] = Form(None),
    arxiv_start_date: Optional[str] = Form(None),
    openreview_years: Optional[str] = Form(None),
):
    if sid in client_tasks:
        return {"error": "该会话已有任务在运行。"}

    session_upload_dir = UPLOADS_DIR / str(uuid.uuid4())
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = Path(pdf_file.filename).name
    pdf_path_on_server = session_upload_dir / file_name

    try:
        with open(pdf_path_on_server, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
    except Exception as e:
        error_msg = f"错误：无法保存上传的PDF文件: {e}"
        await sio.emit('error', {'message': error_msg}, to=sid)
        return {"error": error_msg}
    finally:
        pdf_file.file.close()

    # [修改] 将所有参数打包到一个字典中
    params = {
        'tier': tier,
        'force': (force == 'true'),
        'max_papers_frontier': max_papers_frontier,
        'max_papers_openreview': max_papers_openreview,
        'relevance_threshold': relevance_threshold,
        'similarity_threshold': similarity_threshold,
        'arxiv_start_date': arxiv_start_date,
        'openreview_years': openreview_years,
    }
    
    task = asyncio.create_task(run_main_script(sid, str(pdf_path_on_server), params))
    client_tasks[sid] = task
    
    return {"message": f"审稿流程已启动！正在分析文件: {file_name}"}

# --- WebSocket 事件 ---
@sio.event
async def connect(sid, environ):
    print(f"🔗 客户端已连接: {sid}")
    await sio.emit('sid', {'sid': sid}, to=sid)

@sio.event
def disconnect(sid):
    if sid in client_tasks:
        client_tasks[sid].cancel()
        del client_tasks[sid]
    print(f"🔌 客户端已断开: {sid}")

if __name__ == "__main__":
    # [修改] 检查必要的环境变量
    required_env_vars = ['DASHSCOPE_API_KEY', 'OPENREVIEW_EMAIL', 'OPENREVIEW_PASSWORD']
    if any(not os.getenv(var) for var in required_env_vars):
        print("❌ 启动错误: 缺少必要的环境变量。请确保您已创建 .env 文件并正确配置了以下变量: ")
        for var in required_env_vars:
            print(f"  - {var}")
    else:
        print("✅ 所有必要的环境变量已加载。")
        uvicorn.run(socket_app, host="0.0.0.0", port=50001)