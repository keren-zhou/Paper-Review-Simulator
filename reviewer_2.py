# -*- coding: utf-8 -*-

import openai
import os
import json
from pathlib import Path
import time
from typing import Dict, Any, List
from thefuzz import fuzz  # 用于模糊字符串匹配, 确保已运行: pip install thefuzz
import argparse # 导入 argparse 库用于处理命令行参数

# ==============================================================================
# --- 1. 全局配置与客户端初始化 ---
# ==============================================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    # 这是一个兜底检查，主控脚本会确保API Key已设置
    raise ValueError("错误：请在您的环境中设置 DASHSCOPE_API_KEY 环境变量！")

try:
    qwen_client = openai.OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print("--- 千问 Qwen API 客户端初始化成功 ---")
except Exception as e:
    print(f"初始化 OpenAI 客户端（用于千问）时出错: {e}")
    # 在流水线中，如果初始化失败，最好直接退出
    exit(1)

# ==============================================================================
# --- 2. 核心功能函数 (与原版保持一致) ---
# ==============================================================================

def load_required_data(summary_path: Path, frontier_path: Path) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]] | None]:
    """加载审稿人所需的知识文件。"""
    print(f"\n--- 步骤 1/5: 正在加载知识文件 ---")
    summary_data, frontier_data = None, None
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        print(f"  ✅ 成功加载论文结构化摘要: {summary_path.name}")
    except Exception as e:
        print(f"  ❌ 错误: 加载或解析论文摘要文件 '{summary_path}' 失败: {e}")

    try:
        with open(frontier_path, 'r', encoding='utf-8') as f:
            frontier_data = json.load(f)
        print(f"  ✅ 成功加载领域前沿报告: {frontier_path.name}")
    except Exception as e:
        print(f"  ❌ 错误: 加载或解析前沿报告 '{frontier_path}' 失败: {e}")

    return summary_data, frontier_data

def filter_self_from_frontier(summary_data: Dict[str, Any], frontier_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从前沿报告中过滤掉用户自己的论文，避免自我比较。"""
    print("--- 步骤 2/5: 正在检查并从前沿报告中排除用户论文自身 ---")
    try:
        user_title = summary_data.get("paper_summary", {}).get("supporting_evidence", {}).get("title", "Unknown Title")
        if user_title == "Unknown Title":
             user_title = next((v for k, v in summary_data.items() if isinstance(v, str) and 'title' in k.lower()), "Unknown Title")
        
        filtered_list = []
        removed_count = 0
        for paper in frontier_data:
            frontier_title = paper.get('title', '')
            if fuzz.ratio(user_title.lower(), frontier_title.lower()) > 95:
                print(f"  ℹ️  检测到并已移除用户论文自身: '{frontier_title}'")
                removed_count += 1
            else:
                filtered_list.append(paper)
        
        if removed_count == 0:
            print("  ✅ 未在前沿报告中发现用户论文自身。")
            
        return filtered_list
    except Exception as e:
        print(f"  ⚠️ 警告: 在过滤自身论文时发生错误: {e}. 将使用未经过滤的列表。")
        return frontier_data

def prepare_llm_input_context(summary_data: Dict[str, Any], filtered_frontier_data: List[Dict[str, Any]]) -> str:
    """将加载的数据整合成一个结构清晰、适合LLM分析的文本上下文。"""
    print("--- 步骤 3/5: 正在为AI审稿顾问准备分析上下文 ---")
    
    paper_summary = summary_data.get("paper_summary", {})
    innovations = paper_summary.get("key_innovations", [])
    problem = paper_summary.get("problem_statement", "Not explicitly stated")

    innovations_text = "\n".join([f"- **{item.get('innovation_name', 'Unnamed Innovation')}**: {item.get('innovation_description', '')}" for item in innovations])

    user_paper_context = f"""
### Core Information of User's Paper
- **Problem Addressed**: {problem}
- **Claimed Key Innovations**:
{innovations_text}
"""

    frontier_papers_context = "\n### Background: State-of-the-Art Research in the Field\n"
    for i, paper in enumerate(filtered_frontier_data, 1):
        frontier_papers_context += f"""
---
**Frontier Paper {i}**:
- **Title**: {paper.get('title', 'N/A')}
- **Core Idea Summary**: {paper.get('frontier_summary', 'N/A')}
---
"""
    print("  ✅ 上下文准备完成。")
    return user_paper_context + frontier_papers_context

def generate_novelty_review_md_with_qwen(context: str) -> str | None:
    """调用千问大模型，扮演一个富有洞察力的审稿顾问，生成一份建设性的英文Markdown报告。"""
    # ... (此函数内部的 Prompt 和 API 调用逻辑保持不变) ...
    prompt = f"""
You are a top-tier AI research advisor with extensive experience and a sharp eye for detail. Your task is not merely to criticize a paper, but to act as a senior mentor. Based on the provided "Core Information of User's Paper" and the "State-of-the-Art Research Background," you must provide the author with a profound, forward-looking novelty assessment report. Your goal is to help them anticipate and prepare for the tough questions they might face during the rebuttal phase.

Your output MUST be the complete content for a **Markdown (.md)** file, written in professional, academic English.

Please strictly follow this Markdown structure for your analysis and report:

# Novelty and Contribution Assessment Report (Reviewer #2 Perspective)

## 1. Overall Originality Assessment
*   Provide a concise, insightful overall evaluation here. Where does the originality of this paper lie? Is it a completely new idea, a clever combination of existing techniques, or an incremental improvement upon prior work? Clearly position it within the current academic landscape.

## 2. Detailed Innovation Analysis
*   Analyze each of the user's "Claimed Key Innovations" one by one.
*   Compare each innovation against the "State-of-the-Art Research Background." Is there conceptual overlap? Or does it address a specific aspect overlooked by recent work? Your analysis must be specific and well-supported by the provided context.

## 3. Problem Timeliness & Research Motivation
*   Evaluate whether the problem this paper addresses is still an open and significant challenge in the field.
*   Based on the frontier research, are there new technological trends or paradigms that have shifted how this problem is typically approached? Is the paper's motivation sufficiently strong and well-argued?

## 4. Potential Discussion Points for Rebuttal
*   **Crucial Note**: Your goal is NOT to directly criticize the author for a "lack of baseline comparisons," as code for many state-of-the-art works is often unavailable. Instead, you must identify the **1-3 most relevant papers** from the frontier research that a human reviewer is most likely to bring up.
*   For each identified paper, simulate a reviewer's tone and pose a specific, pointed question. For example: "A reviewer might ask: How does your method fundamentally differ from, and what are its advantages over, [Frontier Paper X] in terms of [a specific aspect]?"
*   Then, provide the author with a direction for their thinking or a suggestion to help them formulate a response that emphasizes the uniqueness of their work. This section is about preparation and strategy, not criticism.

---
[Analysis Materials]
{context}
---

Now, please begin writing your Markdown assessment report. Your output must be the complete Markdown text, starting with `# Novelty and Contribution Assessment Report`.
"""
    print("--- 步骤 4/5: 正在调用千问API生成英文新颖性评估MD报告... ---")
    start_time = time.time()
    try:
        response = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.2,
        )
        end_time = time.time()
        print(f"  ✅ 成功收到API响应，耗时 {end_time - start_time:.2f} 秒。")
        return response.choices[0].message.content
    except Exception as e:
        print(f"  ❌ 错误：调用千问API时失败: {e}")
        return None

def save_report_md(report_content: str, output_path: Path):
    """将生成的Markdown报告内容保存为.md文件。"""
    print(f"--- 步骤 5/5: 正在保存Markdown评估报告 ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"  🎉 评估报告已成功保存到: {output_path.resolve()}")
    except Exception as e:
        print(f"  ❌ 错误: 保存报告失败: {e}")

# ==============================================================================
# --- 3. 主流程控制器 (已修改以适应流水线) ---
# ==============================================================================

def run_novelty_assessment(summary_json_path: str, frontier_report_path: str, output_md_path: str):
    """
    协调整个新颖性评估流程的主函数。
    此函数现在接收精确的输入和输出文件路径，而不是目录。
    """
    summary_file = Path(summary_json_path)
    frontier_file = Path(frontier_report_path)
    output_file = Path(output_md_path)
    
    # 确保输出文件的父目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data, frontier_data = load_required_data(summary_file, frontier_file)
    if not summary_data or not frontier_data:
        print("\n❌ 流程终止：缺少一个或多个必要的知识文件。")
        return

    filtered_frontier = filter_self_from_frontier(summary_data, frontier_data)
    
    llm_context = prepare_llm_input_context(summary_data, filtered_frontier)
    
    md_report = generate_novelty_review_md_with_qwen(llm_context)
    
    if md_report:
        # 直接将报告保存到由主控脚本指定的路径
        save_report_md(md_report, output_file)
    else:
        print("\n❌ 流程终止：未能生成新颖性评估报告。")

# ==============================================================================
# --- 4. 脚本入口 (已修改为接收命令行参数) ---
# ==============================================================================
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Reviewer 2: 基于前沿研究进行论文新颖性评估。")
    parser.add_argument("--summary_json_path", type=str, required=True, help="来自 Step 2 的综合分析 JSON 文件路径。")
    parser.add_argument("--frontier_report_path", type=str, required=True, help="来自 Step 4 的前沿分析报告 JSON 文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="用于保存新颖性审稿报告的 Markdown 文件路径。")
    parser.add_argument("--config", type=str, help="Path to the configuration file (accepted but not used).")
    # 解析从命令行传入的参数
    args = parser.parse_args()

    # 检查所有必要的输入文件是否存在，提供更清晰的错误提示
    input_paths = [args.summary_json_path, args.frontier_report_path]
    if not all(Path(p).exists() for p in input_paths):
         print("="*60)
         print("错误：一个或多个输入文件不存在。请检查 main.py 传递的路径是否正确。")
         print(f"  - 论文摘要路径: '{args.summary_json_path}' (存在: {Path(args.summary_json_path).exists()})")
         print(f"  - 前沿报告路径: '{args.frontier_report_path}' (存在: {Path(args.frontier_report_path).exists()})")
         print("="*60)
    else:
        # 调用主流程函数，传入解析后的参数
        run_novelty_assessment(
            summary_json_path=args.summary_json_path,
            frontier_report_path=args.frontier_report_path,
            output_md_path=args.output_path
        )