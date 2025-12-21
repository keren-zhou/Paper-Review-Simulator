# step2_analysis.py
# -*- coding: utf-8 -*-

import openai
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import time
from pathlib import Path
import re
import argparse # 导入 argparse 用于处理命令行参数

# ==============================================================================
# --- 1. 全局配置与客户端初始化 ---
# ==============================================================================
# API Key 将由主控脚本通过环境变量注入
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    # 这是一个安全检查，如果脚本被独立运行且未设置环境变量，则会报错
    raise ValueError("错误：请在您的环境中设置 DASHSCOPE_API_KEY 环境变量！")

try:
    qwen_client = openai.OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print("--- 千问 Qwen API 客户端初始化成功 ---")
except Exception as e:
    print(f"初始化 OpenAI 客户端（用于千问）时出错: {e}")
    exit()


# ==============================================================================
# --- 2. 核心功能函数 (与您提供的版本保持一致) ---
# ==============================================================================

def extract_key_sections_improved(markdown_text: str) -> str:
    """从 Markdown 中提取摘要、引言和结论。"""
    print("--- 步骤 1/7: 提取论文关键章节 (摘要、引言、结论) ---")
    INTRODUCTION_KEYWORDS = {'introduction'}
    CONCLUSION_KEYWORDS = {'conclusion', 'conclusions', 'summary', 'discussion', 'future work'}
    extracted_sections = {}
    
    abstract_pattern = re.compile(r"##?\s*Abstract\n(.*?)(?=\n##?\s)", re.IGNORECASE | re.DOTALL)
    abstract_match = abstract_pattern.search(markdown_text)
    if abstract_match:
        extracted_sections['abstract'] = abstract_match.group(0).strip()

    headings = list(re.finditer(r"^(##?)\s+(.*)", markdown_text, re.MULTILINE))
    sections = []
    for i, match in enumerate(headings):
        title_text = match.group(2).strip()
        start_pos = match.start()
        end_pos = headings[i+1].start() if i + 1 < len(headings) else len(markdown_text)
        content = markdown_text[start_pos:end_pos].strip()
        cleaned_title = re.sub(r"^\d+\.?\s*", "", title_text).lower()
        sections.append({"cleaned_title": cleaned_title, "content": content})

    for section in sections:
        if 'introduction' not in extracted_sections and section['cleaned_title'] in INTRODUCTION_KEYWORDS:
            extracted_sections['introduction'] = section['content']
        if 'conclusion' not in extracted_sections and section['cleaned_title'] in CONCLUSION_KEYWORDS:
            extracted_sections['conclusion'] = section['content']

    final_text_parts = [extracted_sections.get(key) for key in ['abstract', 'introduction', 'conclusion'] if extracted_sections.get(key)]
    if not final_text_parts:
        print("警告：未能提取任何关键章节。将使用全文进行分析。")
        return markdown_text
    
    print("--- 关键章节提取完成 ---\n")
    return "\n\n---\n\n".join(final_text_parts)

def analyze_paper_summary_with_qwen(paper_content: str) -> dict | None:
    """调用千问进行论文的“基础分析”。"""
    prompt = f"""
You are a world-class AI research assistant. Analyze the provided Abstract, Introduction, and Conclusion of a research paper and extract key information into a structured JSON format. Your output MUST be a single, valid JSON object with no other text. The JSON object must conform to the following structure: {{ "problem_statement": "...", "limitations_of_prior_work": ["..."], "key_innovations": [{{ "innovation_name": "...", "innovation_description": "..." }}], "supporting_evidence": {{ "theoretical_summary": "...", "experimental_summary": {{ "datasets": ["..."], "baselines": ["..."], "key_results": "..." }} }}, "keywords": ["..."] }}
---
{paper_content}
"""
    print("--- 步骤 2/7: 正在调用千问 API 进行论文基础分析... ---")
    start_time = time.time()
    try:
        response = qwen_client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.1)
        end_time = time.time()
        print("--- 成功收到 API 响应，耗时 {:.2f} 秒 ---\n".format(end_time - start_time))
        response_content = response.choices[0].message.content
        cleaned_content = re.sub(r'```json\n(.*?)\n```', r'\1', response_content, flags=re.DOTALL)
        return json.loads(cleaned_content)
    except Exception as e:
        print("错误：论文基础分析 API 调用或 JSON 解析失败: {}".format(e))
        return None

def extract_experimental_sections(markdown_text: str) -> str:
    """从 Markdown 中提取方法和实验部分。"""
    print("--- 步骤 3/7: 提取论文实验章节 (方法、实验) ---")
    METHOD_KEYWORDS = {'method', 'methods', 'methodology', 'our method', 'proposed method', 'approach'}
    EXPERIMENT_KEYWORDS = {'experiment', 'experiments', 'experimental setup', 'implementation details', 'evaluation', 'results', 'comparisons', 'ablation study'}
    headings = list(re.finditer(r"^(##?)\s+(.*)", markdown_text, re.MULTILINE))
    sections = []
    for i, match in enumerate(headings):
        title_text = match.group(2).strip()
        start_pos = match.start()
        end_pos = headings[i+1].start() if i + 1 < len(headings) else len(markdown_text)
        content = markdown_text[start_pos:end_pos].strip()
        cleaned_title = re.sub(r"^\d+\.?\s*", "", title_text).lower()
        sections.append({"cleaned_title": cleaned_title, "content": content})

    extracted_content = [section['content'] for section in sections if section['cleaned_title'] in METHOD_KEYWORDS or section['cleaned_title'] in EXPERIMENT_KEYWORDS]
    if not extracted_content:
        print("警告：未能找到方法或实验章节，将使用全文进行细节分析。")
        return markdown_text
    
    print("--- 实验章节提取完成 ---\n")
    return "\n\n---\n\n".join(extracted_content)

def analyze_paper_details_with_qwen(paper_content: str) -> dict | None:
    """调用千问进行论文的“细节分析”。"""
    prompt = f"""
You are an expert AI assistant. Analyze the provided Methodology and Experimental sections of a research paper. Your goal is to extract two key pieces of information into a structured JSON format: 1. A single, de-duplicated list of all methods the paper compares itself against. 2. A list of the core technological components used to build the paper's own method. Your output MUST be a single, valid JSON object with no other text. The JSON object must follow this exact structure: {{ "comparison_methods": ["..."], "methodological_components": [{{ "component_name": "...", "component_usage": "..." }}] }}
---
{paper_content}
"""
    print("--- 步骤 4/7: 正在调用千问 API 进行论文细节分析... ---")
    start_time = time.time()
    try:
        response = qwen_client.chat.completions.create(model="qwen-plus", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.1)
        end_time = time.time()
        print("--- 成功收到 API 响应，耗时 {:.2f} 秒 ---\n".format(end_time - start_time))
        response_content = response.choices[0].message.content
        cleaned_content = re.sub(r'```json\n(.*?)\n```', r'\1', response_content, flags=re.DOTALL)
        detail_result = json.loads(cleaned_content)
        # 确保去重
        if 'comparison_methods' in detail_result and isinstance(detail_result['comparison_methods'], list):
            seen = set()
            unique_list = [item for item in detail_result['comparison_methods'] if item and item.lower() not in seen and not seen.add(item.lower())]
            detail_result['comparison_methods'] = sorted(unique_list, key=str.lower)
        return detail_result
    except Exception as e:
        print("错误：论文细节分析 API 调用或 JSON 解析失败: {}".format(e))
        return None

def extract_references_from_markdown(markdown_text: str) -> list[str]:
    """使用 findall 策略提取完整的参考文献列表。"""
    print("--- 步骤 5/7: 提取完整的参考文献列表 ---")
    match = re.search(r'^##?\s+References\s*$', markdown_text, re.MULTILINE | re.IGNORECASE)
    if not match:
        print("警告：在 Markdown 文件中未找到 'References' 章节。")
        return []
    references_section = markdown_text[match.start():]
    references_section = re.sub(r'^##?\s+References\s*', '', references_section, count=1, flags=re.IGNORECASE).strip()
    pattern = re.compile(r'(\[\d+\].*?)(?=\s*\[\d+\]|$)', re.DOTALL)
    raw_refs = pattern.findall(references_section)
    extracted_refs = [re.sub(r'\s+', ' ', ref).strip() for ref in raw_refs if ref.strip()]
    if not extracted_refs:
        print("警告：未能提取出任何文献条目。")
    else:
        print("--- 成功提取 {} 条完整的参考文献 ---\n".format(len(extracted_refs)))
    return extracted_refs

def link_method_to_reference_by_citation(method_name: str, full_text: str, references_list: list[str]) -> tuple[str | None, str | None]:
    """通过引用编号将方法链接到参考文献。"""
    pattern = re.compile(r'\b' + re.escape(method_name) + r'\b[^\]]*?\[(\d+)', re.IGNORECASE)
    match = pattern.search(full_text)
    if not match:
        return None, None
    citation_number = match.group(1)
    ref_pattern = re.compile(r'^\s*\[\s*' + citation_number + r'\s*\]')
    for ref in references_list:
        if ref_pattern.match(ref):
            return citation_number, ref
    return citation_number, None


# ==============================================================================
# --- 3. 统一的主流程控制器 (已修改为接收参数) ---
# ==============================================================================

def run_comprehensive_analysis(markdown_path: str, output_json_path: str):
    """
    从指定的 Markdown 文件生成综合分析 JSON。
    :param markdown_path: 输入的 Markdown 文件路径。
    :param output_json_path: 输出的 JSON 文件路径。
    """
    md_file = Path(markdown_path)
    output_file = Path(output_json_path)

    if not md_file.exists():
        print(f"错误：输入文件不存在: {markdown_path}")
        return

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- 正在读取 Markdown 文件: {md_file.name} ---\n")
    full_text = md_file.read_text(encoding='utf-8')

    # --- Part 1: 执行基础分析 ---
    key_sections_text = extract_key_sections_improved(full_text)
    base_analysis_data = analyze_paper_summary_with_qwen(key_sections_text)
    if not base_analysis_data:
        print("❌ 论文基础分析失败，流程终止。")
        return

    # --- Part 2: 执行细节分析 ---
    experimental_sections_text = extract_experimental_sections(full_text)
    detail_analysis_data = analyze_paper_details_with_qwen(experimental_sections_text)
    if not detail_analysis_data:
        print("❌ 论文细节分析失败，流程终止。")
        return

    # --- Part 3: 链接参考文献 ---
    references = extract_references_from_markdown(full_text)
    print("--- 步骤 6/7: 开始链接参考文献 ---")
    comparison_methods = detail_analysis_data.get("comparison_methods", [])
    linked_methods = []
    for method in comparison_methods:
        citation_num, found_ref = link_method_to_reference_by_citation(method, full_text, references)
        reference_text = "Citation not found in text"
        if citation_num and found_ref:
            reference_text = found_ref
            print(f"  - ✅ 成功匹配 '{method}' -> 引用 [{citation_num}]")
        elif citation_num and not found_ref:
            reference_text = f"Found citation [{citation_num}] in text, but no matching entry in reference list."
            print(f"  - ⚠️  '{method}' 找到引用 [{citation_num}]，但未在文献列表中匹配。")
        else:
            print(f"  - ❌ 未能为 '{method}' 在整个文档中找到引用编号。")
        linked_methods.append({"method_name": method, "reference": reference_text})
    print("--- 参考文献链接完成 ---\n")

    # --- Part 4: 合并所有分析结果到一个 JSON 对象 ---
    print("--- 步骤 7/7: 正在合并所有分析结果... ---")
    
    comprehensive_json = {
        "paper_summary": base_analysis_data,
        "experimental_details": {
            "methodological_components": detail_analysis_data.get("methodological_components", []),
            "comparison_methods_with_references": linked_methods
        }
    }
    
    # --- Part 5: 保存最终的综合文件 ---
    with open(str(output_file), 'w', encoding='utf-8') as f:
        json.dump(comprehensive_json, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Step 2 成功！最终的综合分析结果已保存到: {output_file.resolve()}")


# ==============================================================================
# --- 4. 脚本入口 (已修改为接收命令行参数) ---
# ==============================================================================
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Step 2: Create a comprehensive JSON analysis from a paper's Markdown file.")
    parser.add_argument("--markdown_path", type=str, required=True, help="Path to the input Markdown file generated by step1.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output comprehensive_analysis.json file.")
    parser.add_argument("--config", type=str, help="Path to the configuration file (accepted but not used in this script).")
    args = parser.parse_args()
    
    # 检查输入文件是否存在 (虽然主函数里也检查了，但在这里提前检查可以提供更快的反馈)
    if not Path(args.markdown_path).exists():
         print("="*60)
         print(f"错误：输入的 Markdown 文件 '{args.markdown_path}' 不存在。")
         print("请确保 Step 1 已成功运行，并提供了正确的路径。")
         print("="*60)
    else:
        # 使用从命令行获取的路径调用主流程函数
        run_comprehensive_analysis(
            markdown_path=args.markdown_path,
            output_json_path=args.output_path
        )