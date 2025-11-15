# step5_analysis.py
# -*- coding: utf-8 -*-

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import openreview
import json
import os
import dill
import pandas as pd
from typing import List, Dict, Any
import argparse
import configparser
import re  # <--- 新增: 导入正则表达式模块

# For semantic filtering functionality
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================================================================
# 2. 核心功能代码 (来自您的项目)
# ==============================================================================

# --- Utility Functions ---
def get_client(email: str, password: str):
    """根据传入的凭证初始化 OpenReview 客户端。"""
    try:
        client_v1 = openreview.Client(baseurl='https://api.openreview.net', username=email, password=password)
        client_v2 = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net', username=email, password=password)
        print("✅ OpenReview 客户端初始化成功。")
        return client_v1, client_v2
    except Exception as e:
        print(f"❌ 错误: 初始化 OpenReview 客户端失败: {e}")
        print("   请检查您在 config.ini 文件中配置的 EMAIL 和 PASSWORD。")
        raise SystemExit

def papers_to_list(papers_dict: Dict) -> List:
    all_papers = []
    for group in papers_dict.values():
        for venue_papers in group.values():
            all_papers.extend(venue_papers)
    return all_papers

def to_csv_pandas(papers_list: List[Dict], fpath: str):
    if not papers_list:
        print("⚠️ 警告: 没有论文可以保存到 CSV。")
        return
    try:
        df = pd.DataFrame(papers_list)
        df.to_csv(fpath, index=False, encoding='utf-8-sig')
        print(f"✅ 成功保存 {len(df)} 篇论文到 '{fpath}'")
    except Exception as e:
        print(f"❌ 错误: 保存 CSV 文件失败: {e}")

def save_papers(papers: Any, fpath: str):
    with open(fpath, 'wb') as fp:
        dill.dump(papers, fp)
    print(f"   -> 原始论文数据已缓存至: {fpath}")

def load_papers(fpath: str) -> Any:
    with open(fpath, 'rb') as fp:
        papers = dill.load(fp)
    print(f"   -> 从缓存加载原始论文数据: {fpath}")
    return papers

# --- Venue Functions ---
def get_venues(clients, confs, years):
    """
    从OpenReview获取所有会议ID，并根据提供的会议缩写列表进行精确匹配。
    """
    client_v1, client_v2 = clients
    all_venues = set()
    try:
        venues_v1 = client_v1.get_group(id='venues').members
        all_venues.update(venues_v1)
    except Exception: pass
    try:
        venues_v2 = client_v2.get_group(id='venues').members
        all_venues.update(venues_v2)
    except Exception: pass

    # ==========================================================================
    # --- [优化核心] START ---
    # 使用正则表达式和单词边界(\b)来确保精确匹配，避免子字符串误匹配。
    # ==========================================================================
    reqd_venues = []
    for venue in all_venues:
        for conf in confs:
            # 创建一个正则表达式，确保conf是一个独立的单词或被非字母数字字符包围
            # 例如，查找 "RE" 时，它会匹配 "RE/2024"，但不会匹配 "NeurIPS" 或 "Conference"
            # re.escape() 会处理特殊字符，例如 CODES+ISSS 中的 '+'
            pattern = r'\b' + re.escape(conf) + r'\b'
            
            # 使用 re.search 进行不区分大小写的匹配
            if re.search(pattern, venue, re.IGNORECASE):
                # 如果匹配成功，再检查年份
                for year in years:
                    if year in venue:
                        reqd_venues.append(venue)
                        break # 找到年份后，无需再检查此会议的其他年份
                break # 找到会议后，无需再检查此venue是否匹配其他conf缩写
    # ==========================================================================
    # --- [优化核心] END ---
    # ==========================================================================
            
    return list(set(reqd_venues))


def group_venues(venues, bins):
    bins_dict = {bin_name: [] for bin_name in bins}
    for venue in venues:
        for bin_name in bins:
            # 这里的匹配逻辑也同样可以优化，以提高分组的准确性
            pattern = r'\b' + re.escape(bin_name) + r'\b'
            if re.search(pattern, venue, re.IGNORECASE):
                bins_dict[bin_name].append(venue)
                break
    return bins_dict

# --- Paper Fetching ---
def get_papers(clients, grouped_venues):
    _, client_v2 = clients
    papers = {}
    for group, venues in grouped_venues.items():
        if not venues: # 如果分组后某个会议没有找到任何venue，则跳过
            continue
        papers[group] = {}
        for venue in venues:
            print(f"   -> 正在查询会议: {venue}...")
            try:
                submissions = client_v2.get_all_notes(content={'venueid': venue}, details='directReplies')
                papers[group][venue] = submissions
                print(f"      找到 {len(submissions)} 篇已接收的论文。")
            except Exception as e:
                print(f"      ⚠️ 警告: 无法获取 {venue} 的论文。错误: {e}")
                papers[group][venue] = []
    return papers

# --- Filtering Logic ---
def check_keywords_with_text(keywords, text):
    if not text or not keywords: return None, False
    text_lower = str(text).lower()
    for keyword in keywords:
        if str(keyword).lower() in text_lower:
            return keyword, True
    return None, False

def title_filter(paper, keywords):
    title = paper.content.get('title', {}).get('value', '')
    return check_keywords_with_text(keywords, title)

def abstract_filter(paper, keywords):
    abstract = paper.content.get('abstract', {}).get('value', '')
    return check_keywords_with_text(keywords, abstract)

def satisfies_any_filters(paper, keywords, filters):
    for filter_func in filters:
        matched_keyword, matched = filter_func(paper, keywords)
        if matched:
            return matched_keyword, filter_func.__name__, True
    return None, None, False

# --- Data Extractor ---
class Extractor:
    def __init__(self, fields, subfields, details_subfields):
        self.fields = fields
        self.subfields = subfields
        self.details_subfields = details_subfields
  
    def __call__(self, paper):
        trimmed = {}
        for field in self.fields:
            trimmed[field] = getattr(paper, field, None)
        
        content = getattr(paper, 'content', {})
        for field in self.subfields.get('content', []):
            value_dict = content.get(field, {})
            trimmed[field] = value_dict.get('value') if isinstance(value_dict, dict) else value_dict
            
        if self.details_subfields and hasattr(paper, 'details'):
            for field in self.details_subfields: trimmed[f"review_{field}"] = []
            
            for reply in paper.details.get('directReplies', []):
                is_review = any('Official_Review' in inv for inv in reply.get('invitations', []))
                if is_review:
                    for field in self.details_subfields:
                        value_dict = reply.get('content', {}).get(field, {})
                        value = value_dict.get('value') if isinstance(value_dict, dict) else value_dict
                        if value: trimmed[f"review_{field}"].append(str(value))

            for field in self.details_subfields:
                trimmed[f"review_{field}"] = " ||| ".join(trimmed[f"review_{field}"])
        
        return trimmed

# --- Scraper Class (修改后) ---
class Scraper:
    def __init__(self, conferences, years, keywords, extractor, email, password, fns=[]):
        self.confs = conferences
        self.years = years
        self.keywords = keywords
        self.extractor = extractor
        self.fns = fns
        self.filters = []
        self.clients = get_client(email, password)
        self.papers = None

    def add_filter(self, filter_func):
        self.filters.append(filter_func)
  
    def __call__(self):
        print("\n[步骤 2] 正在查找匹配的 OpenReview 会议...")
        venues = get_venues(self.clients, self.confs, self.years)
        if not venues:
            print("❌ 未在 OpenReview 上找到与给定会议和年份匹配的条目。")
            self.papers = {}
            return
        print(f"   找到 {len(venues)} 个相关的会议 ID。")

        print("\n[步骤 3] 正在从会议中获取论文数据 (这可能需要一些时间)...")
        grouped = group_venues(venues, self.confs)
        papers_data = get_papers(self.clients, grouped)
        
        print("\n[步骤 4] 正在应用初始关键词过滤器并提取数据...")
        extracted_papers = {}
        count = 0
        for group, venues_data in papers_data.items():
            extracted_papers[group] = {}
            for venue, paper_list in venues_data.items():
                extracted_papers[group][venue] = []
                for paper in paper_list:
                    match_kw, match_type, satisfies = satisfies_any_filters(paper, self.keywords, self.filters)
                    if satisfies:
                        for fn in self.fns: paper = fn(paper)
                        
                        extracted = self.extractor(paper)
                        extracted['conference'] = group
                        extracted['match_keyword'] = match_kw
                        extracted['match_type'] = match_type
                        
                        extracted_papers[group][venue].append(extracted)
                        count += 1
        
        self.papers = extracted_papers
        print(f"   初始过滤完成。找到 {count} 篇可能相关的论文。")

# ==============================================================================
# 3. 流水线集成功能 (新增与修改)
# ==============================================================================

def generate_search_topic_and_keywords_from_json(analysis_json_path: str) -> (str, List[str]):
    """从 step2 的分析文件中动态生成高质量的搜索主题和关键词列表。"""
    print("\n[步骤 0] 正在从分析文件中动态生成搜索主题和关键词...")
    try:
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data.get("paper_summary", {})
        problem = summary.get("problem_statement", "a novel problem")
        innovations = [item.get("innovation_name", "unnamed innovation") for item in summary.get("key_innovations", [])]
        keywords = summary.get("keywords", [])
        
        search_topic = f"A research paper addressing the problem of '{problem}'. Key innovations include {', '.join(innovations)}. Core concepts are {', '.join(keywords)}."
        
        print(f"   -> 已生成搜索主题: \"{search_topic}\"")
        print(f"   -> 将使用关键词进行预过滤: {keywords}")
        return search_topic, keywords
    except Exception as e:
        print(f"   ❌ 错误: 无法从 JSON 文件生成搜索主题: {e}")
        return "", []

def load_conferences_for_tier(filepath: str, target_tier: str) -> List[str]:
    """从 JSON 知识库中为特定等级加载会议缩写。"""
    print(f"\n[步骤 1] 正在从 '{filepath}' 加载 '{target_tier}' 等级的会议...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_confs = json.load(f)
        
        target_confs = [
            conf['venue_abbr'] for conf in all_confs if conf.get('tier') == target_tier
        ]
        
        if not target_confs:
            print(f"   ⚠️ 警告: 未找到 '{target_tier}' 等级的会议。")
        else:
            print(f"   -> 找到 {len(target_confs)} 个会议: {', '.join(target_confs)}")
        return target_confs
    except FileNotFoundError:
        print(f"   ❌ 错误: 知识库文件未在 '{filepath}' 找到。")
        return []
    except json.JSONDecodeError:
        print(f"   ❌ 错误: 文件 '{filepath}' 不是有效的 JSON 文件。")
        return []

def filter_by_semantic_similarity(papers_list: List[Dict], topic: str, threshold: float, model_name: str) -> List[Dict]:
    """根据与给定主题的语义相似度过滤论文列表。"""
    if not papers_list:
        return []

    print("\n[步骤 5] 正在执行精确的语义过滤...")
    print(f"   -> 核心主题: '{topic}'")
    print(f"   -> 相似度阈值: {threshold}")

    try:
        print(f"   -> 正在加载嵌入模型 ({model_name})...")
        model = HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"   ❌ 错误: 无法加载嵌入模型。请检查网络连接。错误: {e}")
        return []

    print(f"   -> 正在为 {len(papers_list)} 篇论文创建文本嵌入...")
    paper_texts = [f"{p.get('title', '')}\n{p.get('abstract', '')}" for p in papers_list]
    
    query_embedding = model.embed_query(topic)
    paper_embeddings = model.embed_documents(paper_texts)
    
    print("   -> 正在计算余弦相似度...")
    similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
    
    highly_relevant_papers = []
    for i, paper in enumerate(papers_list):
        score = similarities[i]
        if score >= threshold:
            paper['similarity_score'] = round(score, 4)
            highly_relevant_papers.append(paper)
    
    highly_relevant_papers.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"   ✅ 语义过滤完成。找到 {len(highly_relevant_papers)} 篇高度相关的论文。")
    return highly_relevant_papers

def filter_papers_with_reviews(papers_list: List[Dict], review_fields: List[str]) -> List[Dict]:
    """最终过滤步骤，确保只保留具有实际、非空审稿内容的论文。"""
    print("\n[步骤 6] 最终检查: 过滤包含审稿内容的论文...")
    papers_with_actual_reviews = []
    for paper in papers_list:
        has_any_review_content = False
        for field in review_fields:
            review_key = f"review_{field}"
            if paper.get(review_key, "").strip():
                has_any_review_content = True
                break
        
        if has_any_review_content:
            papers_with_actual_reviews.append(paper)

    print(f"   -> 找到 {len(papers_with_actual_reviews)} 篇具有可访问审稿意见的论文。")
    return papers_with_actual_reviews

# ==============================================================================
# 4. 主执行流程 (完全重构以适应流水线)
# ==============================================================================

def run_openreview_scraper(
    analysis_json_path: str,
    output_csv_path: str,
    target_tier: str,
    email: str,
    password: str,
    similarity_threshold: float,
    max_papers: int,
    embedding_model_name: str,
    search_years: List[str]
):
    """
    协调整个 OpenReview 抓取和过滤流程的主函数。
    """
    output_dir = os.path.dirname(output_csv_path)
    papers_cache_path = os.path.join(output_dir, 'raw_papers_cache.pkl')

    search_topic, pre_filter_keywords = generate_search_topic_and_keywords_from_json(analysis_json_path)
    if not search_topic or not pre_filter_keywords:
        print("❌ 无法从分析文件生成搜索参数，流程终止。")
        return

    target_conferences = load_conferences_for_tier('venue_knowledge_base_ccf_auto.json', target_tier)
    
    if not target_conferences:
        print("❌ 未找到目标会议，流程终止。")
        return

    review_fields_to_extract = ['rating', 'confidence', 'summary', 'strengths', 'weaknesses']
    extractor = Extractor(
        fields=['forum'], 
        subfields={'content': ['title', 'keywords', 'abstract', 'pdf']},
        details_subfields=review_fields_to_extract
    )

    def modify_paper_links(paper):
        """辅助函数，创建完整的 URL。"""
        paper.forum = f"https://openreview.net/forum?id={paper.forum}"
        pdf_val = paper.content.get('pdf', {}).get('value')
        if pdf_val: paper.content['pdf']['value'] = f"https://openreview.net{pdf_val}"
        return paper

    if os.path.exists(papers_cache_path):
         print("\n[缓存] 发现已存在的原始论文缓存。从文件加载。")
         all_papers_raw = load_papers(papers_cache_path)
    else:
        scraper = Scraper(
            conferences=target_conferences, 
            years=search_years, 
            keywords=pre_filter_keywords, 
            extractor=extractor, 
            email=email,
            password=password,
            fns=[modify_paper_links]
        )
        scraper.add_filter(title_filter)
        scraper.add_filter(abstract_filter)
        
        scraper()
        
        all_papers_raw = scraper.papers
        if all_papers_raw:
            save_papers(all_papers_raw, papers_cache_path)

    initial_paper_list = papers_to_list(all_papers_raw)
    
    semantically_relevant_papers = filter_by_semantic_similarity(
        papers_list=initial_paper_list,
        topic=search_topic,
        threshold=similarity_threshold,
        model_name=embedding_model_name
    )
    
    final_papers_with_reviews = filter_papers_with_reviews(
        papers_list=semantically_relevant_papers,
        review_fields=review_fields_to_extract
    )
    
    print(f"\n[步骤 7] 正在为最终报告选择前 {max_papers} 篇最相关的论文...")
    top_papers = final_papers_with_reviews[:max_papers]
    print(f"   -> 所有过滤器后总论文数: {len(final_papers_with_reviews)}")
    print(f"   -> 为最终报告选择的论文数: {len(top_papers)}")

    print("\n[步骤 8] 正在将最终结果保存到 CSV...")
    to_csv_pandas(top_papers, output_csv_path)
    
    print("\n🎉 Step 5 流程成功完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: 从 OpenReview 抓取相关的论文和审稿意见。")
    parser.add_argument("--analysis_json_path", type=str, required=True, help="指向 comprehensive_analysis.json 文件的路径。")
    parser.add_argument("--output_csv_path", type=str, required=True, help="保存输出的 final_relevant_papers.csv 文件的路径。")
    parser.add_argument("--target_tier", type=str, required=True, help="目标会议等级 (例如: CCF-A, CCF-B, CCF-C)。")
    parser.add_argument("--config", type=str, default='config.ini', help="配置文件路径。")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    EMAIL = os.getenv('OPENREVIEW_EMAIL')
    PASSWORD = os.getenv('OPENREVIEW_PASSWORD')
    SIMILARITY_THRESHOLD = float(config['SETTINGS']['SIMILARITY_THRESHOLD'])
    MAX_PAPERS_IN_REPORT = int(config['SETTINGS']['MAX_PAPERS_OPENREVIEW'])
    EMBEDDING_MODEL_NAME = config['PATHS']['EMBEDDING_MODEL_PATH']
    
    SEARCH_YEARS = [year.strip() for year in config['SETTINGS']['OPENREVIEW_SEARCH_YEARS'].split(',')]

    if not EMAIL or not PASSWORD:
        print("❌ 关键错误: 请在运行前，在 config.ini 文件中设置您的 OpenReview EMAIL 和 PASSWORD。")
    else:
        run_openreview_scraper(
            analysis_json_path=args.analysis_json_path,
            output_csv_path=args.output_csv_path,
            target_tier=args.target_tier,
            email=EMAIL,
            password=PASSWORD,
            similarity_threshold=SIMILARITY_THRESHOLD,
            max_papers=MAX_PAPERS_IN_REPORT,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            search_years=SEARCH_YEARS
        )