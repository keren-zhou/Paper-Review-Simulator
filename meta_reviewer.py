import openai
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import argparse

# ==============================================================================
# --- 1. 全局配置与客户端初始化 ---
# ==============================================================================

def initialize_qwen_client() -> openai.OpenAI | None:
    """
    根据环境变量初始化并返回千问客户端。
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：环境变量 DASHSCOPE_API_KEY 未设置！")
        return None
    
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print("✅ 千问客户端初始化成功。")
        return client
    except Exception as e:
        print(f"❌ 初始化千问客户端时出错: {e}")
        return None

# 将客户端初始化推迟到函数中，使其成为全局变量
qwen_client = initialize_qwen_client()

# ==============================================================================
# --- 2. 数据加载函数 ---
# ==============================================================================

def load_json_data(file_path: Path) -> Dict[str, Any] | None:
    """加载 JSON 文件 (通常来自步骤2的论文核心分析)。"""
    print(f"   -> 正在加载论文核心分析文件: {file_path.name}")
    if not file_path.exists():
        print(f"   ❌ 错误：文件不存在 {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"   ❌ 错误：加载或解析 JSON 文件失败: {e}")
        return None

def load_markdown_report(file_path: Path) -> str | None:
    """加载 Markdown 格式的单个审稿人报告。"""
    print(f"   -> 正在加载审稿人报告: {file_path.name}")
    if not file_path.exists():
        print(f"   ❌ 错误：文件不存在 {file_path}")
        return None
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"   ❌ 错误：读取 Markdown 文件失败: {e}")
        return None

def load_and_sample_csv_reviews(file_path: Path, n_samples: int = 2) -> str | None:
    """加载 step5 输出的 CSV 文件，并采样正面和负面审稿意见作为参考。"""
    print(f"   -> 正在加载并采样参考审稿意见: {file_path.name}")
    if not file_path.exists():
        print(f"   ❌ 错误：参考审稿意见文件不存在 {file_path}")
        return "Reference review file not found."
    try:
        df = pd.read_csv(file_path)
        required_cols = ['review_rating', 'review_strengths', 'review_weaknesses', 'title']
        if not all(col in df.columns for col in required_cols):
            print(f"   ❌ 错误：CSV 文件缺少必需的列。需要: {required_cols}")
            return "Reference CSV is missing required columns."
        
        # 将评分转换为可排序的数值
        df['review_rating_num'] = pd.to_numeric(df['review_rating'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        df.dropna(subset=['review_rating_num'], inplace=True)
        df['review_rating_num'] = df['review_rating_num'].astype(int)
        
        df_sorted = df.sort_values(by='review_rating_num', ascending=False)
        
        # 确保有足够的样本
        if len(df_sorted) < n_samples * 2:
            n_samples = max(1, len(df_sorted) // 2) # 如果样本不足，则取一半
            if n_samples == 0:
                 print("   ⚠️ 警告：CSV中的有效评论太少，无法采样。")
                 return "Not enough valid reviews in the reference file to sample from."

        high_rated = df_sorted.head(n_samples)
        low_rated = df_sorted.tail(n_samples)

        reference_text = "--- 高分评价示例 (学习其风格和看重的优点) ---\n\n"
        for _, row in high_rated.iterrows():
            reference_text += f"**论文标题:** {row['title']}\n**评分:** {row['review_rating']}\n**优点:**\n{row.get('review_strengths', 'N/A')}\n**缺点:**\n{row.get('review_weaknesses', 'N/A')}\n\n---\n"
        
        reference_text += "\n--- 低分评价示例 (学习其批评角度和常见的拒稿原因) ---\n\n"
        for _, row in low_rated.iterrows():
            reference_text += f"**论文标题:** {row['title']}\n**评分:** {row['review_rating']}\n**优点:**\n{row.get('review_strengths', 'N/A')}\n**缺点:**\n{row.get('review_weaknesses', 'N/A')}\n\n---\n"
        
        return reference_text
    except Exception as e:
        print(f"   ❌ 错误：处理 CSV 参考文件失败: {e}")
        return "Error processing reference review file."

# ==============================================================================
# --- 3. 核心功能：主席 AI (Meta-Reviewer) ---
# ==============================================================================

def get_conference_standards(tier: str, name: str) -> str:
    """根据会议等级和具体名称返回更具针对性的评审标准描述。"""
    # 如果用户直接输入了CCF等级，则会议名和等级相同，避免冗余显示
    conf_display_name = f"{name} ({tier})" if name.upper() != tier.upper() else tier
    
    standards = {
        'CCF-A': f"This is a top-tier {conf_display_name} conference (e.g., NeurIPS, CVPR). Submissions are expected to be groundbreaking, with significant novelty, high impact, and technically flawless execution. Experimental validation must be comprehensive and rigorous.",
        'CCF-B': f"This is a reputable {conf_display_name} conference. Submissions should present solid, novel contributions to the field. Strong, complete experimental validation is crucial. The work should be a clear advancement over existing literature, but does not need to be revolutionary.",
        'CCF-C': f"This is a CCF-C conference ({name}). Submissions are expected to be correct, clear, and useful. Incremental contributions are acceptable if they are well-executed and properly evaluated. The focus is on technical correctness and clarity."
    }
    return standards.get(tier, standards['CCF-B'])

# --- [核心修改 2] --- 更新函数以接收和使用 conference_name
def generate_final_review_with_qwen(
    paper_analysis: Dict[str, Any],
    quality_review: str,
    novelty_review: str,
    reference_reviews: str,
    conference_tier: str,
    conference_name: str
) -> str | None:
    """调用大模型扮演主席AI角色，根据指定的会议标准，生成最终审稿意见。"""
    
    conference_standard_description = get_conference_standards(conference_tier, conference_name)
    conf_display_name = f"{conference_name} ({conference_tier})" if conference_name.upper() != conference_tier.upper() else conference_tier

    # --- [核心美化修改] 使用更丰富的 Markdown 格式 ---
    prompt = f"""
You are a highly experienced Area Chair for a {conference_tier} AI conference. Your judgment standard is:
> {conference_standard_description}

**Task**: Synthesize the provided analysis and reviews into a final meta-review. Your goal is to be decisive, insightful, and constructive.

---
### **Input 1: Paper's Core Analysis**
```json
{json.dumps(paper_analysis, indent=2)}
```
***Input 2: Reviewer R1 - Technical Quality Report
{quality_review}
Input 3: Reviewer R2 - Novelty Assessment Report
{novelty_review}
Input 4: Reference Reviews from Similar Venues
{reference_reviews}
YOUR META-REVIEW & OUTPUT FORMAT
Generate a comprehensive Meta-Review. You MUST strictly follow the Markdown format below. Use headings, bold text, and lists to structure your report for clarity.
Meta-Review: Final Decision & Rebuttal Strategy
1. Final Verdict
Overall Recommendation
[Choose ONE: Strong Accept, Weak Accept, Borderline, Reject]
Justification: A concise, high-level justification for your decision, directly linking to the {conference_tier} standards. Explain the most critical factor that led to this verdict.
Executive Summary
A 2-3 sentence summary of the paper's core contribution and the key factors (both positive and negative) that influenced the final decision.
2. Detailed Analysis
Strengths
Primary Strength: (Synthesize and elaborate on the most significant strength).
Secondary Strength: (List other notable positive aspects).
Weaknesses
(Ranked by severity. Be critical and specific.)
[Critical] Weakness 1: (Describe the most severe flaw, e.g., a fundamental issue with the core claim, methodology, or experimental validation).
[Major] Weakness 2: (Describe a major issue, e.g., missing key comparisons to state-of-the-art, insufficient ablation studies).
[Minor] Weakness 3: (Describe a minor issue, e.g., presentation issues, unclear sections).
3. Guidance for Author Rebuttal
Predicted Reviewer Questions
Based on the identified weaknesses, anticipate the most challenging questions the authors will face.
Regarding Weakness #1: A reviewer will likely ask: "[Your pointed question here]"
Regarding Weakness #2: It is crucial to address: "[Another challenging question]"
Strategic Advice
Provide tactical advice for the authors. For instance: "To address the concerns about baseline comparisons, the authors should not just state that code is unavailable. Instead, they should create a detailed table comparing their reported metrics against the metrics reported in the original papers of the SOTA methods, and discuss the potential reasons for any discrepancies. This would demonstrate a higher level of academic rigor."
"""
    print("\n[主席AI] 正在调用千问大模型进行最终分析与决策...")
    try:
        response = qwen_client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2, # 使用较低的温度以保证决策的稳定性
        )
        print("[主席AI] ✅ 成功收到千问的 Meta-Review 报告。")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[主席AI] ❌ 调用千问 API 失败: {e}")
        return None
def run_meta_review(
    base_path_str: str,
    paper_name: str,
    conference_tier: str,
    conference_name: str, # <-- [新增] 接收 name
    output_md_path: str
    ):
    """协调整个 Meta-Review 流程的主函数。"""
    print(f"\n--- 开始为论文 '{paper_name}' 生成 Meta-Review ---")
    base_dir = Path(base_path_str)
    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 根据传入的参数构建所有必需文件的完整路径
    analysis_json_path = base_dir / f"{paper_name}_comprehensive_analysis.json"
    quality_review_path = base_dir / f"{paper_name}_review_QualityInspector.md" # 审稿人1的文件名
    novelty_review_path = base_dir / f"{paper_name}_review_NoveltyAssessor.md"   # 审稿人2的文件名
    reference_csv_path = base_dir / "final_relevant_papers.csv"
    
    print("\n[步骤 1/3] 正在加载所有输入数据...")
    paper_analysis = load_json_data(analysis_json_path)
    quality_review = load_markdown_report(quality_review_path)
    novelty_review = load_markdown_report(novelty_review_path)
    reference_reviews = load_and_sample_csv_reviews(reference_csv_path)
    
    # 检查所有文件是否都已成功加载
    if not all([paper_analysis, quality_review, novelty_review, reference_reviews]):
        print("\n❌ 关键文件加载失败，无法继续生成 Meta-Review。请检查文件路径和内容。")
        return
        
    print("\n[步骤 2/3] 所有数据加载成功，准备提交给主席 AI...")
    final_report = generate_final_review_with_qwen(
        paper_analysis,
        quality_review,
        novelty_review,
        reference_reviews,
        conference_tier,
        conference_name
    )
    
    if final_report:
        print("\n[步骤 3/3] 正在保存最终的 Meta-Review 报告...")
        try:
            output_path.write_text(final_report, encoding='utf-8')
            print("\n🎉 Meta Reviewer 全部流程成功！")
            print(f"最终的 Meta-Review 报告已保存到: {output_path.resolve()}")
        except Exception as e:
            print(f"❌ 保存最终报告时出错: {e}")
    else:
        print("\n❌ 未能生成最终报告，流程终止。")
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Meta Reviewer: 综合所有评审信息并给出最终决策。")
    parser.add_argument("--base_path", type=str, required=True, help="包含所有论文相关输入文件的基础目录。")
    parser.add_argument("--paper_name", type=str, required=True, help="论文的文件名 (不含扩展名)。")
    parser.add_argument("--conference_tier", type=str, required=True, help="目标会议等级 (例如: CCF-A, CCF-B, CCF-C)。")
    # --- [新增] ---
    parser.add_argument("--conference_name", type=str, required=True, help="目标会议的具体名称 (例如: CVPR)。")
    parser.add_argument("--output_path", type=str, required=True, help="保存最终 meta-review 报告的完整路径。")
    parser.add_argument("--config", type=str, help="Path to the configuration file (accepted but not used).")
    args = parser.parse_args()
    # 检查客户端是否成功初始化
    if qwen_client:
        run_meta_review(
            base_path_str=args.base_path,
            paper_name=args.paper_name,
            conference_tier=args.conference_tier,
            conference_name=args.conference_name, # <-- [新增] 传递了 name
            output_md_path=args.output_path
        )
    else:
        print("程序因千问客户端初始化失败而终止。")