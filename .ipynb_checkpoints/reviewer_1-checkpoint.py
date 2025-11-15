# -*- coding: utf-8 -*-
# reviewer_1.py

"""
## 审稿智能体 1: 质量审查员 (Quality Inspector) ##

功能:
- 模拟一位严谨的审稿人，专注于论文的内部质量。
- 从论文的摘要、引言、方法、实验、结论等所有章节，评估其技术合理性、论证严谨性、写作清晰度和实验完整性。
- 该智能体不关心论文的新颖性或与外部工作的比较，只根据论文本身提供的内容进行严格的内部审查。

输入:
- 论文的 Markdown 全文文件 (由 step1_analysis.py 生成)。
- 论文的结构化分析 JSON 文件 (由 step2_analysis.py 生成)。

输出:
- 一份详细的 Markdown 格式的审稿报告，评估论文的质量、清晰度、重要性和实验设计，并提出尖锐问题。
"""

# ==============================================================================
# 1. 导入所需模块
# ==============================================================================
import openai
import os
import json
from pathlib import Path
import time
import argparse  # 引入 argparse 用于解析命令行参数

# ==============================================================================
# 2. 全局配置与客户端初始化
# ==============================================================================

# --- 千问 Qwen API 配置 ---
# 确保您的 DASHSCOPE_API_KEY 已设置为环境变量。
# 主控脚本 main.py 会自动注入这个环境变量。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：环境变量 DASHSCOPE_API_KEY 未设置！")

try:
    # 初始化 OpenAI 客户端以连接到千问服务
    qwen_client = openai.OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print("✅ [Reviewer_1] 千问客户端初始化成功。")
except Exception as e:
    print(f"❌ [Reviewer_1] 初始化千问客户端时出错: {e}")
    exit()

# ==============================================================================
# 3. 核心功能函数 (这部分函数逻辑保持不变)
# ==============================================================================

def load_required_files(md_path: Path, json_path: Path) -> tuple[str | None, dict | None]:
    """
    加载论文的 Markdown 全文和结构化的 JSON 分析文件。

    Args:
        md_path: 指向论文 Markdown 文件的 Path 对象。
        json_path: 指向综合分析 JSON 文件的 Path 对象。

    Returns:
        一个元组，包含 Markdown 内容和加载的 JSON 数据。如果文件未找到，则返回 (None, None)。
    """
    print(f"--- [Reviewer_1] 正在加载所需文件: {md_path.name} 和 {json_path.name} ---")

    if not md_path.exists():
        print(f"❌ 错误: Markdown 文件未找到于 '{md_path}'")
        return None, None
    markdown_content = md_path.read_text(encoding='utf-8')

    if not json_path.exists():
        print(f"❌ 错误: JSON 分析文件未找到于 '{json_path}'")
        return None, None
    with open(json_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    print("--- ✅ [Reviewer_1] 文件加载成功。 ---\n")
    return markdown_content, analysis_data

def generate_review_prompt(paper_summary: dict, paper_full_text: str) -> str:
    """
    构建一个高度详细和结构化的 Prompt，以指导大语言模型的审稿过程。

    Args:
        paper_summary: 包含论文结构化摘要的字典。
        paper_full_text: Markdown 格式的论文全文内容。

    Returns:
        一个包含完整 Prompt 的字符串。
    """
    summary_text = json.dumps(paper_summary, indent=2, ensure_ascii=False)

    # 这个精心设计的Prompt是该脚本的核心，它定义了AI审稿人的角色、原则和输出格式
    prompt = f"""
You are a senior reviewer for a top-tier AI conference (e.g., NeurIPS, CVPR, ICML). Your task is to write a profound, rigorous, and critical review based on the provided paper summary (in JSON) and the full paper text (in Markdown).

**Your Reviewing Principles:**
1.  **Internal Scrutiny:** All your assessments MUST be based strictly on the content of the paper itself. Do not introduce any external knowledge.
2.  **Critical Thinking:** Your primary mission is to identify weaknesses. Approach every claim with skepticism and seek out flaws in its argumentation and evidence.
3.  **Constructive Feedback:** While critical, your feedback must be constructive. Provide specific, actionable suggestions for improvement.

**Your review report MUST strictly follow this Markdown format:**

# Review Report: [Insert Paper Title Here]

## 1. Summary
*   In 2-3 objective sentences, summarize the paper's core objective, proposed method, and key findings.

## 2. Overall Assessment & Critical Insight
*   In a single paragraph, provide a high-level assessment. What is the most impressive strength of this paper, and what is its most critical flaw? This should be your sharpest, most insightful take after reading the entire paper.

## 3. Detailed Review

### Quality
*   **Technical Soundness:** Is the technical description rigorous? Are there logical fallacies or unreasonable assumptions in the methodology?
*   **Evidential Support:** Are the main claims supported by sufficient and solid evidence (e.g., theoretical proofs, experimental results)? Pinpoint specific claims that lack adequate support.
*   **Completeness:** Does this work represent a finished, mature study, or is it a preliminary exploration? Have the authors honestly and rigorously assessed the limitations of their own work?

### Clarity
*   **Writing & Structure:** Is the paper well-organized and logically structured? Are there ambiguous or confusing statements?
*   **Reproducibility:** Does the paper provide enough technical detail (e.g., pseudocode, model architecture, hyperparameter settings, data preprocessing steps) to allow an expert in the field to reproduce the results? If not, what key information is missing?
*   **Actionable Suggestions:** Provide 1-2 concrete suggestions to improve clarity (e.g., "The authors should add a transition paragraph at the beginning of Section 3 to clarify the relationship between Method A and Method B.").

### Significance
*   **Contribution Assessment:** What is the most significant contribution of this research? Is it a novel problem, a groundbreaking method, a unique dataset, or a profound theoretical insight?
*   **Potential Impact:** Are the results important enough that they are likely to be used by other researchers or inspire new research directions? Does this work solve a recognized problem in a significantly better way?

### Experimental Evaluation
*   **Experimental Design:** Is the experimental setup fair and reasonable? Are the evaluation metrics comprehensive and convincing?
*   **Comparative Analysis:**
    *   **Quantitative:** Is the comparison against baselines sufficient? Is there any suspicion of "cherry-picking" results? Is the performance improvement statistically significant?
    *   **Qualitative:** Do the qualitative results (e.g., visualizations, case studies) clearly demonstrate the advantages of the proposed method?
*   **Ablation Studies:** Has the paper thoroughly validated the necessity and effectiveness of each key component of its method through comprehensive ablation studies?
*   **Missing Experiments:** What crucial experiments are missing that, if included, would make the paper's claims much more convincing?

## 4. Critical Questions for the Authors
*   **This is the most important section of your report.** Formulate 2-3 of the most pointed and central questions you have. The answers to these questions should directly influence your final assessment of the paper. They should be designed to force the authors to address the weakest points of their work during the rebuttal phase.
*   **Question 1:** [State your challenging question here]
*   **Question 2:** [State another question targeting a core assumption or experimental design]
*   ...

---
**Begin your review now based on the following paper information.**

**Paper's Structured Summary (JSON):**
```json
{summary_text}
```
**Paper's Full Text (Markdown):**`
{paper_full_text}
"""
    return prompt

def generate_review(analysis_data: dict, markdown_content: str) -> str | None:
    """
    调用千问 API 来生成审稿报告。
    
    Args:
        analysis_data: 加载后的论文结构化分析字典。
        markdown_content: 论文的全文内容。
    
    Returns:
        生成的审稿报告字符串，如果 API 调用失败则返回 None。
    """
    paper_summary = analysis_data.get("paper_summary", {})
    # 如果在摘要中找不到标题，则使用一个通用标题作为后备
    paper_title = paper_summary.get("supporting_evidence", {}).get("title", "Untitled Paper")
    
    print(f"--- [Reviewer_1] 正在为论文生成审稿意见: '{paper_title}' ---")
    print("--- [Reviewer_1] 此过程可能需要 1-2 分钟，请稍候。 ---")
    
    # 步骤 1: 构建详细的 Prompt
    prompt = generate_review_prompt(paper_summary, markdown_content)
    
    # 步骤 2: 调用大语言模型 API
    start_time = time.time()
    try:
        response = qwen_client.chat.completions.create(
            model="qwen-plus",      # 使用一个强大的模型来完成这个复杂的任务
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,        # 分配足够的 token 以生成长篇审稿意见
            temperature=0.2,        # 使用较低的 temperature 以确保输出严谨、符合事实
        )
        end_time = time.time()
        review_content = response.choices[0].message.content
    
        # 将报告中的标题占位符替换为真实的论文标题
        review_content = review_content.replace("[Insert Paper Title Here]", paper_title, 1)
    
        print(f"--- ✅ [Reviewer_1] API 响应成功，耗时 {end_time - start_time:.2f} 秒。 ---\n")
        return review_content
    
    except Exception as e:
        print(f"❌ 错误: 调用千问 API 失败: {e}")
        return None

# ==============================================================================
# 4. 主流程控制器 (修改后)
# ==============================================================================

def run_internal_review_process(markdown_file_path: str, json_file_path: str, output_md_path: str):
    """
    协调单个论文的整个内部审稿流程。

    Args:
        markdown_file_path: 输入的 Markdown 文件的完整路径。
        json_file_path: 输入的 JSON 文件的完整路径。
        output_md_path: 输出的审稿报告 .md 文件的完整路径。
    """
    md_path = Path(markdown_file_path)
    json_path = Path(json_file_path)
    output_path = Path(output_md_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 加载必要的输入文件
    markdown_content, analysis_data = load_required_files(md_path, json_path)
    if not markdown_content or not analysis_data:
        print("❌ [Reviewer_1] 因文件加载失败，审稿流程中止。")
        return
    
    # 步骤 2: 使用大语言模型生成审稿报告
    review_report = generate_review(analysis_data, markdown_content)
    if not review_report:
        print("❌ [Reviewer_1] 因 API 调用失败，审稿流程中止。")
        return
    
    # 步骤 3: 将最终报告保存到指定的 Markdown 文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(review_report)
        print(f"🎉 [Reviewer_1] 审稿流程成功完成！")
        print(f"报告已保存至: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ 错误: 保存审稿报告失败: {e}")

# ==============================================================================
# 5. 脚本入口 (修改后)
# ==============================================================================
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="审稿智能体 1: 对论文进行内部质量审查。")
    parser.add_argument("--markdown_path", type=str, required=True, help="输入的论文 Markdown 文件路径。")
    parser.add_argument("--json_path", type=str, required=True, help="输入的综合分析 JSON 文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="输出的审稿报告 Markdown 文件路径。")
    parser.add_argument("--config", type=str, help="Path to the configuration file (accepted but not used).")
    # 解析从命令行传入的参数
    args = parser.parse_args()
    
    # 检查 API Key 是否已设置 (虽然客户端初始化时已检查，这里多一层保障)
    if not DASHSCOPE_API_KEY:
         print("="*60)
         print("错误：DASHSCOPE_API_KEY 环境变量未设置。")
         print("请确保在运行主控脚本前已在 config.ini 中配置好 API Key。")
         print("="*60)
    else:
        # 使用从命令行获取的参数来运行主流程
        run_internal_review_process(
            markdown_file_path=args.markdown_path,
            json_file_path=args.json_path,
            output_md_path=args.output_path
        )