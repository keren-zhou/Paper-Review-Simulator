# step3_frontier_analysis.py (migrated from step4_frontier_analysis.py)
# -*- coding: utf-8 -*-

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import openai
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import time
import datetime
import calendar
import arxiv
import threading
import queue
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import argparse
import configparser

from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# --- 全局变量 ---
qwen_client = None
async_qwen_client = None
embedding_model = None

# ==============================================================================
# --- 2. 核心功能函数 (已修复) ---
# ==============================================================================

def load_analysis_data(analysis_json_path: Path) -> Dict[str, Any] | None:
    if not analysis_json_path.exists():
        print(f"错误：分析文件不存在于 '{analysis_json_path}'")
        return None
    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 加载或解析JSON文件失败: {e}")
        return None

def generate_search_query_with_qwen(analysis_data: Dict[str, Any]) -> str | None:
    print("   [QueryGen] 正在尝试使用千问大模型生成智能搜索查询...")
    summary = analysis_data.get("paper_summary", {})
    problem = summary.get("problem_statement", "")
    innovations = [f"- {item.get('innovation_name', '')}: {item.get('innovation_description', '')}" for item in summary.get("key_innovations", [])]
    keywords_text = ', '.join(summary.get("keywords", []))
    innovations_text = '\n'.join(innovations)
    prompt = f"""
You are an expert research assistant. Based on the following information from a research paper, generate a single, effective search query for the arXiv database to find the latest related research.
The query's goal is to be **BROAD** to maximize discovery. It should:
1. Identify 3-5 of the most important and distinct conceptual groups from the paper's information.
2. For each group, you can include alternative terms using `OR` (e.g., `"3D Gaussian Splatting" OR "3DGS"`).
3. Crucially, you MUST link these main conceptual groups together using the `OR` operator, not `AND`. This will find papers related to any of the core concepts.
4. Your output MUST be only the query string itself, without any explanation, labels, or quotation marks around the whole string.
--- Paper Information ---
Problem Statement: {problem}
Key Innovations:
{innovations_text}
Keywords: {keywords_text}
---
Example of a good BROAD query: `("3D Gaussian Splatting" OR "3DGS") OR ("compositional scene generation") OR ("physics-aware layout")`
Now, generate the broad query based on the paper information provided.
"""
    try:
        response = qwen_client.chat.completions.create(
            model="qwen-plus", messages=[{"role": "user", "content": prompt}], max_tokens=200, temperature=0.2,
        )
        query = response.choices[0].message.content.strip()
        if query.startswith('"') and query.endswith('"'): query = query[1:-1]
        print(f"   [QueryGen] ✅ 成功生成智能查询。")
        return query
    except Exception as e:
        print(f"   [QueryGen] ❌ 调用千问生成查询失败: {e}")
        return None

# ==========================================================================
# --- [核心修复] ---
# 修正了 construct_search_query 函数中的 UnboundLocalError
# ==========================================================================
def construct_search_query(analysis_data: Dict[str, Any]) -> tuple[str, str]:
    """
    根据分析数据构建 arXiv 搜索查询和用于语义比较的核心文本。
    """
    summary = analysis_data.get("paper_summary", {})
    
    # --- 修复开始 ---
    # 无论后续操作如何，都先无条件地从摘要中提取所需的所有信息
    problem = summary.get("problem_statement", "")
    innovations = [item.get("innovation_name", "") for item in summary.get("key_innovations", [])]
    keywords = summary.get("keywords", [])
    # --- 修复结束 ---

    arxiv_query = generate_search_query_with_qwen(analysis_data)
    
    if not arxiv_query:
        print("   [QueryGen] 启动备用查询生成逻辑...")
        # 现在 'keywords' 变量在此处必定存在
        cleaned_keywords = sorted(list(set(kw.strip() for kw in keywords if kw)), key=len)
        core_keywords = cleaned_keywords[:5]
        arxiv_query = " OR ".join(f'"{kw}"' for kw in core_keywords)
        print(f"   [QueryGen] ✅ 已生成备用查询。")

    # 现在 'keywords' 变量在这一行也必定存在
    semantic_text = f"Problem: {problem}. Innovations: {'. '.join(innovations)}. Keywords: {', '.join(keywords)}"
    
    return arxiv_query, semantic_text

async def summarize_paper_with_qwen_async(title: str, abstract: str) -> Dict[str, Any] | None:
    prompt = f"""
You are a highly skilled AI assistant specializing in scientific literature. Your task is to summarize the following research paper based on its title and abstract.
Your summary MUST strictly follow this format:
"In the domain of [domain], to solve the problem of [problem], a method of [method A + method B] was proposed."
Do not add any other text, explanation, or introductory phrases. Your entire output should be a single JSON object containing one key "summary".
---
Title: {title}
Abstract: {abstract}
---
Output the result as a single JSON object like this: {{"summary": "In the domain of..."}}
"""
    try:
        response = await async_qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"错误：异步总结失败 (论文: '{title[:30]}...'): {e}")
        return None

# ==============================================================================
# --- 3. 流水线任务 (多线程) ---
# ==============================================================================

def search_arxiv_task(query: str, start_date_str: str, raw_papers_queue: queue.Queue):
    print(f"--- [生产者线程启动] 使用查询: '{query}' ---")
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m")
    end_date = datetime.datetime.now()
    client = arxiv.Client()
    seen_ids = set()
    for year in range(end_date.year, start_date.year - 1, -1):
        start_month = 1 if year > start_date.year else start_date.month
        end_month = end_date.month if year == end_date.year else 12
        for month in range(end_month, start_month - 1, -1):
            _, last_day = calendar.monthrange(year, month)
            start_of_month, end_of_month = f"{year}{month:02d}01", f"{year}{month:02d}{last_day}"
            print(f"--- [生产者] 正在搜索: {year}-{month:02d} ---")
            query_with_date = f'({query}) AND submittedDate:[{start_of_month} TO {end_of_month}]'
            search = arxiv.Search(query=query_with_date, max_results=2000, sort_by=arxiv.SortCriterion.SubmittedDate)
            new_papers_this_month = []
            try:
                for result in client.results(search):
                    if result.entry_id not in seen_ids:
                        new_papers_this_month.append({
                            'title': result.title.replace('\n', ' ').strip(),
                            'summary': result.summary.replace('\n', ' ').strip(),
                            'id': result.entry_id, 'pdf_url': result.pdf_url
                        })
                        seen_ids.add(result.entry_id)
            except Exception as e:
                print(f"   [生产者] 搜索 {year}-{month:02d} 时发生错误: {e}")
            if new_papers_this_month:
                print(f"   [生产者] -> 发现 {len(new_papers_this_month)} 篇新论文，放入过滤队列。")
                raw_papers_queue.put(new_papers_this_month)
            time.sleep(3)
    print("--- [生产者线程结束] 所有月份搜索完毕。 ---")
    raw_papers_queue.put(None)

def filter_papers_task(raw_queue: queue.Queue, filtered_queue: queue.Queue, base_semantic_text: str, model, relevance_threshold: float):
    print(f"--- [过滤器线程启动] 使用阈值 {relevance_threshold} 等待原始数据... ---")
    try:
        base_embedding = model.embed_query(base_semantic_text)
    except Exception as e:
        print(f"   [过滤器] 严重错误：无法创建基准嵌入！错误: {e}. 过滤器将无法工作。")
        raw_queue.get(); filtered_queue.put(None)
        return
    while True:
        paper_batch = raw_queue.get()
        if paper_batch is None:
            print("--- [过滤器线程结束] 收到生产者结束信号。---")
            filtered_queue.put(None); break
        print(f"   [过滤器] <- 收到 {len(paper_batch)} 篇论文，开始语义筛选...")
        try:
            texts_to_embed = [f"Title: {p['title']}. Abstract: {p['summary']}" for p in paper_batch]
            paper_embeddings = model.embed_documents(texts_to_embed)
            similarities = cosine_similarity([base_embedding], paper_embeddings)[0]
        except Exception as e:
            print(f"   [过滤器] 严重错误：无法为批次创建文档嵌入！错误: {e}. 跳过此批次。")
            continue
        
        relevant_papers = []
        for i, paper in enumerate(paper_batch):
            if similarities[i] >= relevance_threshold:
                paper['relevance_score'] = float(similarities[i])
                relevant_papers.append(paper)

        if relevant_papers:
            print(f"   [过滤器] -> 筛选出 {len(relevant_papers)}/{len(paper_batch)} 篇强相关论文。")
            filtered_queue.put(relevant_papers)

def summarize_papers_task(filtered_queue: queue.Queue, final_results_list: list):
    print("--- [总结器线程启动] 等待过滤后的论文... ---")
    async def process_batch_async(batch_to_process: List[Dict]):
        tasks = [summarize_paper_with_qwen_async(p['title'], p['summary']) for p in batch_to_process]
        summary_results = await asyncio.gather(*tasks, return_exceptions=True)
        for paper, result in zip(batch_to_process, summary_results):
            if isinstance(result, Exception) or not result or 'summary' not in result:
                print(f"       ❌ 总结失败: '{paper['title'][:50]}...'")
            else:
                final_results_list.append({
                    "title": paper['title'], "arxiv_id": paper['id'], "pdf_url": paper['pdf_url'],
                    "relevance_score": paper.get('relevance_score', 0.0), "frontier_summary": result['summary']
                })
                print(f"       ✅ 总结成功: '{paper['title'][:50]}...'")

    while True:
        paper_batch = filtered_queue.get()
        if paper_batch is None:
            print("--- [总结器线程结束] 收到过滤器结束信号。---"); break
        print(f"   [总结器] <- 收到 {len(paper_batch)} 篇相关论文，开始并发总结...")
        start_time = time.time()
        asyncio.run(process_batch_async(paper_batch))
        end_time = time.time()
        print(f"   [总结器] 完成 {len(paper_batch)} 篇论文的并发总结，耗时 {end_time - start_time:.2f} 秒。")

# ==============================================================================
# --- 4. 主流程控制器 ---
# ==============================================================================
def run_frontier_analysis(
    analysis_json_path: str, output_json_path: str, max_papers: int,
    model_name: str, relevance_threshold: float, search_start_date: str
):
    global qwen_client, async_qwen_client, embedding_model
    try:
        qwen_client = openai.OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        async_qwen_client = openai.AsyncOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("--- 千问 Qwen API 客户端初始化成功 ---")
    except Exception as e: print(f"初始化千问客户端时出错: {e}"); return
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        print(f"--- 语义嵌入模型加载完成 ---\n")
    except Exception as e: print(f"错误: 无法加载嵌入模型 '{model_name}': {e}"); return

    analysis_file, output_file = Path(analysis_json_path), Path(output_json_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("--- 步骤 1/4: 加载基础分析文件 ---")
    analysis_data = load_analysis_data(analysis_file)
    if not analysis_data: return
    
    print("\n--- 步骤 2/4: 构建搜索查询 ---")
    arxiv_query, semantic_text = construct_search_query(analysis_data)
    if not arxiv_query: print("错误：无法构建有效的搜索查询。"); return
    print(f"  ▶️ [最终使用的 arXiv 查询]: {arxiv_query}\n")
    
    raw_papers_queue, filtered_papers_queue, final_results = queue.Queue(10), queue.Queue(10), []
    
    threads = [
        threading.Thread(target=search_arxiv_task, args=(arxiv_query, search_start_date, raw_papers_queue)),
        threading.Thread(target=filter_papers_task, args=(raw_papers_queue, filtered_papers_queue, semantic_text, embedding_model, relevance_threshold)),
        threading.Thread(target=summarize_papers_task, args=(filtered_papers_queue, final_results))
    ]
    print("--- 步骤 3/4: 启动三阶段分析流水线 ---\n")
    for t in threads: t.start()
    for t in threads: t.join()
    
    print("\n--- 所有流水线任务已完成 ---\n")
    print(f"--- 步骤 4/4: 排序、筛选并保存最终的前沿分析报告 ---")
    if not final_results:
        print("警告：未能找到并总结任何相关的领域前沿论文。"); return
    final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results[:max_papers], f, ensure_ascii=False, indent=4)
    print(f"\n🎉 Step 3 成功！共总结了 {len(final_results)} 篇前沿论文。")
    print(f"报告中已保存相关度最高的 {len(final_results[:max_papers])} 篇。")
    print(f"分析报告已保存到: {output_file.resolve()}")

# ==============================================================================
# --- 5. 脚本入口 ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Find and summarize frontier research from arXiv.")
    parser.add_argument("--analysis_json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, default='config.ini', help="Path to the configuration file.")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    if not Path(args.config).exists():
        print(f"错误: 配置文件 '{args.config}' 未找到！"); exit()
    config.read(args.config)
    try:
        max_papers = int(config['SETTINGS']['MAX_PAPERS_FRONTIER'])
        model_name = config['PATHS']['EMBEDDING_MODEL_PATH']
        relevance_threshold = float(config.get('SETTINGS', 'RELEVANCE_THRESHOLD', fallback=0.8))
        search_start_date = config.get('SETTINGS', 'ARXIV_SEARCH_START_DATE', fallback='2025-01')
    except (KeyError, ValueError) as e:
        print(f"错误: 配置文件 'config.ini' 格式不正确或缺少必要的键: {e}"); exit()
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：环境变量 DASHSCOPE_API_KEY 未设置。"); exit()
    run_frontier_analysis(
        analysis_json_path=args.analysis_json_path, output_json_path=args.output_path,
        max_papers=max_papers, model_name=model_name,
        relevance_threshold=relevance_threshold, search_start_date=search_start_date
    )