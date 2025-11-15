# main.py
import os
import sys
import subprocess
import configparser
from pathlib import Path
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def get_conference_info(input_tier_or_name: str, config_path: str = 'venue_knowledge_base_ccf_auto.json') -> tuple[str | None, str | None]:
    # ... (此函数保持不变) ...
    input_upper = input_tier_or_name.upper()
    if input_upper in ['CCF-A', 'CCF-B', 'CCF-C']:
        return input_upper, input_upper
    try:
        if not Path(config_path).exists():
            print(f"错误：会议知识库文件 '{config_path}' 未找到！")
            return None, None
        with open(config_path, 'r', encoding='utf-8') as f:
            venues = json.load(f)
        for venue in venues:
            if venue.get('venue_abbr', '').upper() == input_upper:
                tier = venue.get('tier')
                if tier:
                    print(f"   -> 已识别会议: '{input_tier_or_name}' 属于 '{tier}' 等级。")
                    return input_tier_or_name, tier
                else:
                    print(f"错误：在知识库中找到了会议 '{input_tier_or_name}'，但它没有有效的CCF等级信息。")
                    return None, None
        print(f"错误：输入 '{input_tier_or_name}' 不是一个有效的CCF等级，也未在会议知识库中找到。")
        return None, None
    except json.JSONDecodeError:
        print(f"错误：无法解析会议知识库文件 '{config_path}'。")
        return None, None

def run_step(command: list, step_name: str, output_path: Path, force_run: bool = False, report_type: str | None = None):
    """
    通用函数，执行子进程步骤。
    [修改] 增加 report_type 参数，用于在成功后打印报告信令。
    """
    if not force_run and output_path.exists():
        print(f"\n{'='*25} ⏩ 跳过执行: {step_name} {'='*25}")
        print(f"   - 原因: 输出文件已存在 -> {output_path.name}")
        # [修改] 即使跳过，如果需要，也要发送报告信令
        if report_type:
             print(f"[REPORT_READY]{report_type}:{output_path.resolve()}", flush=True)
        return True

    # ... (其他打印逻辑保持不变) ...
    if force_run and output_path.exists():
        print(f"\n{'='*25} 💥 强制执行: {step_name} {'='*25}")
        print(f"   - 原因: 用户指定了 --force 参数，将覆盖现有文件。")
    else:
        print(f"\n{'='*25} 🚀 开始执行: {step_name} {'='*25}")

    print(f"   - 命令: {' '.join(command)}")

    # [修改] 从环境变量直接获取 API Key，不再依赖 config.ini
    env = os.environ.copy()
    
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', env=env
    )

    # ... (日志流处理逻辑保持不变) ...
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            sys.stdout.write(f"[{step_name}] {output}")
            sys.stdout.flush()

    process.wait()
    stderr_output = process.stderr.read()

    if process.returncode != 0:
        print(f"\n❌❌❌ 错误：步骤 '{step_name}' 执行失败！ ❌❌❌")
        print(f"返回码: {process.returncode}")
        print("--- [错误信息] ---")
        print(stderr_output)
        raise RuntimeError(f"步骤 '{step_name}' 失败")

    print(f"✅ {step_name} 执行成功。")
    
    # [核心修改] 如果步骤成功且是报告类型，则打印信令
    if report_type:
        print(f"[REPORT_READY]{report_type}:{output_path.resolve()}", flush=True)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="论文自动审稿流水线")
    # --- 核心参数 ---
    parser.add_argument("--pdf", type=str, required=True, help="待分析的论文PDF文件的完整路径。")
    parser.add_argument("--tier", type=str, required=True, help="目标会议的等级或名称。")
    parser.add_argument("--force", action="store_true", help="强制重新运行所有步骤。")
    
    # --- [新增] UI 可配置参数 ---
    parser.add_argument("--max-papers-frontier", type=int, default=None)
    parser.add_argument("--max-papers-openreview", type=int, default=None)
    parser.add_argument("--relevance-threshold", type=float, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--arxiv-start-date", type=str, default=None)
    parser.add_argument("--openreview-years", type=str, default=None)
    
    args = parser.parse_args()
    
    # --- 配置加载 ---
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    if not Path(config_path).exists():
        print(f"错误：配置文件 '{config_path}' 未找到！")
        sys.exit(1)
    config.read(config_path)

    # [修改] 更新 config 对象以包含来自命令行的参数（如果提供了的话）
    # 这使得下游脚本无需修改，仍然可以从 config 读取
    def override_config(section, key, value):
        if value is not None:
            if not config.has_section(section): config.add_section(section)
            config.set(section, key, str(value))
            # print(f"   -> [配置覆盖] 使用UI参数: {key} = {value}")

    print("\n--- 正在应用UI配置 ---")
    override_config('SETTINGS', 'MAX_PAPERS_FRONTIER', args.max_papers_frontier)
    override_config('SETTINGS', 'MAX_PAPERS_OPENREVIEW', args.max_papers_openreview)
    override_config('SETTINGS', 'RELEVANCE_THRESHOLD', args.relevance_threshold)
    override_config('SETTINGS', 'SIMILARITY_THRESHOLD', args.similarity_threshold)
    override_config('SETTINGS', 'ARXIV_SEARCH_START_DATE', args.arxiv_start_date)
    override_config('SETTINGS', 'OPENREVIEW_SEARCH_YEARS', args.openreview_years)

    # [修改] API Key 检查现在由 api_server.py 在启动时完成
    
    # ... (会议信息解析和路径设置逻辑保持不变) ...
    print("\n--- 正在解析目标会议信息 ---")
    conference_name, conference_tier = get_conference_info(args.tier)
    if not conference_tier:
        print("❌ 无法确定有效的会议信息，流水线终止。")
        sys.exit(1)

    pdf_path = Path(args.pdf)
    paper_name = pdf_path.stem
    output_base_dir = Path(config['PATHS']['OUTPUT_BASE_DIR'])
    paper_output_dir = output_base_dir / f"{paper_name}_output"
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    
    destination_pdf = paper_output_dir / pdf_path.name
    if args.force or not destination_pdf.exists():
        shutil.copy(pdf_path, destination_pdf)

    print(f"🚀 开始为论文 '{paper_name}' 启动审稿流水线")
    print(f"🎯 目标会议: {conference_name} (等级: {conference_tier})")
    if args.force: print("⚠️  模式: 强制重跑所有步骤。")
    print(f"📂 所有输出文件将保存在: {paper_output_dir.resolve()}")

    md_path = paper_output_dir / f"{paper_name}.md"
    analysis_json_path = paper_output_dir / f"{paper_name}_comprehensive_analysis.json"
    frontier_report_path = paper_output_dir / f"{paper_name}_frontier_report.json"
    openreview_csv_path = paper_output_dir / "final_relevant_papers.csv"
    reviewer1_report_path = paper_output_dir / f"{paper_name}_review_QualityInspector.md"
    reviewer2_report_path = paper_output_dir / f"{paper_name}_review_NoveltyAssessor.md"
    meta_review_path = paper_output_dir / f"{paper_name}_meta_review.md"
    step4_python_executable = config['PATHS']['STEP4_PYTHON_EXECUTABLE']

    # --- 阶段 1: 串行执行 ---
    try:
        # 我们需要将临时覆盖后的 config 传递给子进程
        # 最简单的方法是将会话特定的 config 写入一个临时文件
        session_config_path = paper_output_dir / 'session_config.ini'
        with open(session_config_path, 'w') as configfile:
            config.write(configfile)
        
        # 让所有子进程都读取这个会话特定的配置文件
        base_command_args = ["--config", str(session_config_path)]

        run_step(["python", "step1_preprocess.py", "--pdf_path", str(pdf_path), "--output_dir", str(paper_output_dir)] + base_command_args, "Step 1: PDF 预处理", md_path, args.force)
        run_step(["python", "step2_summarize.py", "--markdown_path", str(md_path), "--output_path", str(analysis_json_path)] + base_command_args, "Step 2: 论文核心分析", analysis_json_path, args.force)
    except RuntimeError as e:
        print(f"预处理步骤失败，无法继续执行。错误: {e}")
        sys.exit(1)

    # --- 阶段 2: 并行执行 ---
    print(f"\n{'='*25} 🚀 开始并行执行审稿分支 {'='*25}")
    with ThreadPoolExecutor(max_workers=3) as executor:
        def run_branch_b():
            # [修改] 传递会话配置文件
            run_step([step4_python_executable, "step4_frontier_analysis.py", "--analysis_json_path", str(analysis_json_path), "--output_path", str(frontier_report_path)] + base_command_args, "分支B (Step 4)", frontier_report_path, args.force)
            run_step(["python", "reviewer_2.py", "--summary_json_path", str(analysis_json_path), "--frontier_report_path", str(frontier_report_path), "--output_path", str(reviewer2_report_path)] + base_command_args, "分支B (Reviewer 2)", reviewer2_report_path, args.force, report_type="reviewer2")
            return "分支B 完成"

        futures = {
            executor.submit(run_branch_b): "分支B (新颖性路径)",
            executor.submit(run_step, ["python", "reviewer_1.py", "--markdown_path", str(md_path), "--json_path", str(analysis_json_path), "--output_path", str(reviewer1_report_path)] + base_command_args, "分支A (质量审查)", reviewer1_report_path, args.force, report_type="reviewer1"): "分支A (质量审查)",
            executor.submit(run_step, [step4_python_executable, "step5_analysis.py", "--analysis_json_path", str(analysis_json_path), "--output_csv_path", str(openreview_csv_path), "--target_tier", conference_tier] + base_command_args, "分支C (相似论文)", openreview_csv_path, args.force): "分支C (相似论文)"
        }
        
        try:
            for future in as_completed(futures):
                task_name = futures[future]
                result = future.result()
                print(f"✅ 并行任务 '{task_name}' 已成功完成。")
        except Exception:
            print(f"\n❌❌❌ 并行执行阶段出现致命错误，流水线终止。 ❌❌❌")
            sys.exit(1)

    print(f"\n{'='*25} ✅ 所有并行审稿分支执行完毕 {'='*25}")

    # --- 阶段 3: 最终汇合 ---
    try:
        run_step([
            "python", "meta_reviewer.py",
            "--base_path", str(paper_output_dir),
            "--paper_name", paper_name,
            "--conference_tier", conference_tier,
            "--conference_name", conference_name,
            "--output_path", str(meta_review_path)
        ] + base_command_args, "Meta Reviewer: 最终决策", meta_review_path, args.force)
    except RuntimeError:
        sys.exit(1)
    
    print("\n🎉🎉🎉 所有步骤执行完毕，流水线成功结束！ 🎉🎉🎉")
    print(f"最终的 Meta-Review 报告位于: {meta_review_path.resolve()}")