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
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv 
import requests
import time
import random

# --- 全局设置 ---
# 加载 .env 文件中的环境变量 (用于加载 DASHSCOPE_API_KEY 等)
load_dotenv()

# 创建一个目录用于存放上传的临时文件
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ==========================================
#               光子支付配置
# ==========================================

# 1. [调试开关] 
# True = 开启模拟支付（跳过真实扣费，用于跑通流程）
# False = 开启真实扣费（需填写有效 SKU_ID 和真实的 ACCESS_KEY）
MOCK_PAYMENT_MODE = True 

# 2. [本地硬编码配置] 
# 当 Cookie 中无法获取时，将使用这些默认值
# 请将下方引号内的内容替换为您真实的 accessKey 和 clientName
DEV_ACCESS_KEY = "developer-key" 
CLIENT_NAME = "developer_name"

# 3. [商品配置]
SKU_ID = 111  # 申请到真实 ID 后请修改此处
PHOTON_API_URL = "https://openapi.dp.tech/openapi/v1/api/integral/consume"
CHARGE_AMOUNT = 1 

# ==========================================

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

# --- 核心业务逻辑 (保持不变) ---
async def run_main_script(sid: str, pdf_path: str, params: dict):
    temp_upload_dir = Path(pdf_path).parent
    
    try:
        command = [
            "python", "main.py",
            "--pdf", pdf_path,
            "--tier", params['tier']
        ]
        if params['force']:
            command.append("--force")
        
        for key, value in params.items():
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
            report_signal_pattern = re.compile(r"\[REPORT_READY\](reviewer1|reviewer2):(.+)")

            while True:
                line = await stream.readline()
                if not line:
                    break
                line_str = line.decode('utf-8', errors='replace').strip()
                
                match = report_signal_pattern.match(line_str)
                if match:
                    report_type = match.group(1)
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
                    continue

                print(f"[SID: {sid}] {stream_name}: {line_str}")
                await sio.emit('log', {'data': line_str}, to=sid)
                
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
                paper_name = Path(pdf_path).stem
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
    request: Request,
    sid: str = Form(...),
    tier: str = Form(...),
    force: str = Form(...), 
    pdf_file: UploadFile = File(...),
    max_papers_frontier: Optional[int] = Form(None),
    max_papers_openreview: Optional[int] = Form(None),
    relevance_threshold: Optional[float] = Form(None),
    similarity_threshold: Optional[float] = Form(None),
    arxiv_start_date: Optional[str] = Form(None),
    openreview_years: Optional[str] = Form(None),
):
    if sid in client_tasks:
        return {"error": "该会话已有任务在运行。"}

    # =======================================================
    #                 光子扣费逻辑 (已修改)
    # =======================================================
    
    # 逻辑：优先从 Cookie 获取，如果为空则使用文件顶部的硬编码变量
    access_key = request.cookies.get("appAccessKey") or DEV_ACCESS_KEY
    client_name = request.cookies.get("clientName") or CLIENT_NAME
    
    # --- 1. 模拟模式 (调试用) ---
    if MOCK_PAYMENT_MODE:
        print(f"[SID: {sid}] ⚠️ [调试模式] 模拟光子扣费成功 (未调用真实接口)")
        # 即使是模拟模式，打印一下当前使用的 key 信息也方便调试
        print(f"[SID: {sid}] Using Key: {access_key[:6]}***, Client: {client_name}")
        await sio.emit('log', {'data': f"💰 [调试模式] 虚拟扣除 {CHARGE_AMOUNT} 光子，跳过支付验证，直接开始..."}, to=sid)
    
    # --- 2. 真实扣费模式 ---
    else:
        # 必须要有 access_key 才能扣费
        if not access_key or access_key == "your_access_key_here":
            error_msg = "❌ 错误：未获取到有效的 AccessKey。请配置 Cookie 或在 api_server.py 中正确填写 DEV_ACCESS_KEY。"
            await sio.emit('error', {'message': error_msg}, to=sid)
            return {"error": error_msg}

        timestamp = int(time.time())
        rand_part = random.randint(1000, 9999)
        biz_no = int(f"{timestamp}{rand_part}")

        payload = {
            "bizNo": biz_no,
            "changeType": 1,
            "eventValue": CHARGE_AMOUNT,
            "skuId": SKU_ID, 
            "scene": "appCustomizeCharge"
        }

        # 这里的 client_name 对应文档中的 x-app-key header
        headers = {
            "accessKey": access_key,
            "x-app-key": client_name, 
            "Content-Type": "application/json"
        }

        try:
            print(f"[SID: {sid}] 正在请求光子扣费: {CHARGE_AMOUNT} 光子...")
            resp = requests.post(PHOTON_API_URL, headers=headers, json=payload, timeout=10)
            resp_data = resp.json()

            if resp_data.get("code") != 0:
                fail_reason = resp_data.get("msg") or resp_data.get("message") or "未知错误"
                error_msg = f"光子扣费失败: {fail_reason} (Code: {resp_data.get('code')})"
                print(f"[SID: {sid}] {error_msg}")
                await sio.emit('error', {'message': error_msg}, to=sid)
                return {"error": error_msg}
            
            print(f"[SID: {sid}] 光子扣费成功！BizNo: {biz_no}")
            await sio.emit('log', {'data': f"💰 已成功扣除 {CHARGE_AMOUNT} 光子，开始审稿流程..."}, to=sid)

        except Exception as e:
            error_msg = f"光子支付接口调用异常: {str(e)}"
            print(f"[SID: {sid}] {error_msg}")
            await sio.emit('error', {'message': error_msg}, to=sid)
            return {"error": error_msg}

    # ==========================
    # 2. 文件保存与任务启动
    # ==========================

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
    required_env_vars = ['DASHSCOPE_API_KEY', 'OPENREVIEW_EMAIL', 'OPENREVIEW_PASSWORD']
    if any(not os.getenv(var) for var in required_env_vars):
        print("❌ 启动错误: 缺少必要的环境变量。请确保您已创建 .env 文件并正确配置了以下变量: ")
        for var in required_env_vars:
            print(f"  - {var}")
    else:
        print("✅ 所有必要的环境变量已加载。")
        uvicorn.run(socket_app, host="0.0.0.0", port=50001)