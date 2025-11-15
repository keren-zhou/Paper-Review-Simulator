# run_review_cli.py
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path

def select_file_from_dir(directory: Path, file_extension: str) -> Path | None:
    """
    扫描指定目录中特定后缀的文件，将其以数字列表形式呈现给用户，
    并返回用户所选文件的Path对象。
    """
    print("="*60)
    print(f"🔍 正在 '{directory}' 目录中搜索 '{file_extension}' 文件...")
    
    # 查找所有PDF文件并排序
    files = sorted([f for f in directory.glob(f"*{file_extension}") if f.is_file()])
    
    if not files:
        print(f"❌ 在指定目录中未找到任何 '{file_extension}' 文件。")
        print("   请先将您的论文PDF上传到此目录。")
        return None
        
    print("📄 请选择您要审稿的论文:")
    for i, file_path in enumerate(files):
        print(f"   [{i + 1}] {file_path.name}")
        
    while True:
        try:
            choice = input(f"请输入您的选择 (数字 1-{len(files)}): ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(files):
                return files[choice_index]
            else:
                print("   无效的数字，请重新输入。")
        except (ValueError, IndexError):
            print("   输入无效，请输入列表中的数字。")

def select_conference_tier() -> str:
    """
    提示用户从预设列表中选择一个会议等级。
    """
    print("\n🎯 请选择目标会议等级:")
    tiers = ['CCF-A', 'CCF-B', 'CCF-C']
    for i, tier in enumerate(tiers):
        print(f"   [{i + 1}] {tier}")
        
    while True:
        try:
            choice = input(f"请输入您的选择 (数字 1-{len(tiers)}): ")
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(tiers):
                return tiers[choice_index]
            else:
                print("   无效的数字，请重新输入。")
        except (ValueError, IndexError):
            print("   输入无效，请输入列表中的数字。")

def ask_force_rerun() -> bool:
    """
    询问用户是否希望强制重新运行所有步骤。
    """
    while True:
        choice = input("\n🔄 是否强制重新运行所有步骤 (这将覆盖已有结果)？ (y/n): ").lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("   输入无效，请输入 'y' 或 'n'。")

def run_main_pipeline(pdf_path: Path, tier: str, force: bool):
    """
    构建并执行 main.py 脚本的命令，并实时将输出流式传输到控制台。
    """
    # 使用 sys.executable 确保我们用的是同一个Python解释器
    command = [
        sys.executable,
        "main.py",
        "--pdf", str(pdf_path),
        "--tier", tier
    ]
    if force:
        command.append("--force")
        
    print("\n" + "="*60)
    print("🚀 即将启动 Auto-Reviewer 自动化审稿流水线...")
    print(f"   - 审稿论文: {pdf_path.name}")
    print(f"   - 目标会议: {tier}")
    print(f"   - 强制重跑: {'是' if force else '否'}")
    print("="*60 + "\n")
    
    try:
        # 使用 Popen 以便实时获取和打印子进程的输出
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误输出重定向到标准输出
            text=True,
            encoding='utf-8',
            bufsize=1 # 设置行缓冲
        )
        
        # 逐行读取并打印输出
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\n❌ 流水线执行出错 (返回码: {return_code})。请检查上面的日志。")
        else:
            print("\n✅ 流水线执行成功完成！")
            
    except FileNotFoundError:
        print("❌ 错误: 未找到 'main.py'。请确保您在项目的根目录下运行此脚本。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

if __name__ == "__main__":
    # 定义用户上传PDF的目录
    upload_directory = Path("uploads")
    upload_directory.mkdir(exist_ok=True) # 确保目录存在
    
    # --- 步骤 1: 选择PDF文件 ---
    selected_pdf = select_file_from_dir(upload_directory, ".pdf")
    if not selected_pdf:
        sys.exit(1) # 如果没有选择文件，则退出程序
        
    # --- 步骤 2: 选择会议等级 ---
    selected_tier = select_conference_tier()
    
    # --- 步骤 3: 询问是否强制运行 ---
    force_rerun = ask_force_rerun()
    
    # --- 步骤 4: 运行主流水线 ---
    run_main_pipeline(selected_pdf, selected_tier, force_rerun)