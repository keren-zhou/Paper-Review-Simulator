# Auto-Reviewer: 自动化论文审稿流水线

本项目是一个端到端的自动化科研论文审稿解决方案。用户只需输入一篇PDF格式的论文和期望投递的会议等级，系统即可生成一份全面、深入的审稿报告，内容涵盖论文的质量评估、新颖性分析，并能预测在Rebuttal（作者回应）阶段可能遇到的尖锐问题。

## 目录
- [项目概览](#项目概览)
- [工作流程](#工作流程)
- [文件功能详解](#文件功能详解)
- [快速开始](#快速开始)
- [如何使用](#如何使用)

## 项目概览

Auto-Reviewer旨在通过模拟真实的同行评审流程，为研究人员提供一个强大的自动化审稿工具。该系统深度整合了大型语言模型（LLM）的能力，并结合了来自arXiv和OpenReview等学术知识库的外部信息。

整个复杂的审稿任务被分解为一系列模块化的步骤，每个步骤由专门的脚本负责。最终，系统会输出一份“主席审稿报告”（Meta-Review），该报告综合了两位专职AI审稿人的意见：一位专注于论文的**内部质量**，另一位则负责评估其在最新研究背景下的**新颖性**。

## 工作流程

整个流水线由 `main.py` 脚本统一调度，严格按照预设顺序执行以下步骤：

1.  **预处理 (Preprocessing)**：将输入的PDF论文转换为结构化的Markdown文件，并提取其中所有的图片。
2.  **核心分析 (Core Analysis)**：解析Markdown文本，提取论文的**问题陈述、关键创新点、实验结果**等核心信息，并存为一个结构化的JSON文件。
3.  **知识获取 (Knowledge Acquisition)**：
    *   **前沿分析**：根据论文关键词，在arXiv上搜索最新的相关研究，以构建该领域的前沿技术图景。
    *   **同行评审分析**：在OpenReview上检索与目标会议同级别的相似论文及其完整的同行评审意见。
4.  **多智能体审稿 (Multi-Agent Review)**：
    *   **审稿人1 (质量审查员)**：该智能体严格审查论文的内部逻辑、技术合理性、实验完整性和表述清晰度。
    *   **审稿人2 (新颖性评估员)**：该智能体将论文的创新点与从arXiv获取的前沿研究进行对比，评估其原创性和贡献度。
5.  **最终决策 (Meta-Review)**：最后，一位“主席审稿人”将综合两位AI审稿人的报告、论文的核心分析以及从OpenReview获取的真实审稿意见，生成一份最终的、全面的审稿报告，并为作者点明Rebuttal阶段的潜在质询点。

## 文件功能详解

以下是项目中每个脚本的详细功能说明。

### 调度与控制

-   `main.py`: **总控脚本**
    -   **功能**: 流水线的入口。负责解析命令行参数（如PDF路径、会议等级），读取`config.ini`配置文件，并按顺序调用其他所有步骤的脚本。
    -   **核心逻辑**: 通过`subprocess`安全地执行每一步，并管理环境变量（如API密钥）的传递。支持`--force`参数强制重新运行所有步骤。

### 核心处理步骤

-   `step1_preprocess.py`: **PDF预处理**
    -   **功能**: 1. 使用Nougat OCR模型将PDF论文转换为保留了公式和表格的结构化Markdown文件。2. 从PDF中提取所有图片，并保存到独立的`images`文件夹。
    -   **输入**: 待处理的PDF文件路径。
    -   **输出**: 一份`.md`格式的论文文本文件和一个包含所有图片的`images`文件夹。

-   `step2_summarize.py`: **论文核心分析**
    -   **功能**: 解析上一步生成的Markdown文件，提取摘要、引言、结论等关键章节。随后调用大语言模型，将这些信息提炼成一个结构化的JSON文件，其中包含论文的核心论点、创新、实验设置和结果等。
    -   **输入**: `step1`生成的Markdown文件。
    -   **输出**: 一份内容详尽的分析文件 (`_comprehensive_analysis.json`)。

-   `step3_frontier_analysis.py`: **前沿技术分析 (arXiv)**
    -   **功能**: 基于论文的关键词和创新点,智能生成arXiv搜索查询,以查找最新、最相关的研究。之后,通过多线程流水线(搜索、过滤、总结)对这些前沿论文进行处理和总结。
    -   **输入**: `step2`生成的核心分析JSON文件。
    -   **输出**: 一份前沿分析报告 (`_frontier_report.json`),其中包含了相关前沿论文的标题、链接和核心思想总结。

-   `step4_analysis.py`: **相似论文与评审意见检索 (OpenReview)**
    -   **功能**: 根据用户设定的会议等级（如CCF-A），在OpenReview上检索主题相似的已发表论文及其官方评审意见。这为模仿真实审稿人的口吻和关注点提供了宝贵的参考。
    -   **输入**: `step2`生成的核心分析JSON文件和用户指定的目标会议等级。
    -   **输出**: 一个CSV文件 (`final_relevant_papers.csv`)，包含相似论文的标题、摘要以及审稿人给出的优缺点总结。

**流水线架构:**
1.  **串行阶段 (准备工作)**:
    - 流程从 `Step 1` 开始，其输出是 `Step 2` 的输入。
    - `Step 2` 完成后，生成了所有后续步骤所依赖的核心分析文件 (`_comprehensive_analysis.json`)。
    - 这两个步骤必须按顺序执行。

2.  **并行阶段 (多分支审稿)**:
    - `Step 2` 完成后，`main.py` 使用Python的 `concurrent.futures.ThreadPoolExecutor` 启动一个线程池，同时执行三个独立的审稿分支：
        - **分支A (质量评审)**: 运行 `reviewer_1.py`。
        - **分支B (新颖性评审)**: 先运行 `step3_frontier_analysis.py`,完成后再运行 `reviewer_2.py`。
        - **分支C (相似论文分析)**: 运行 `step4_analysis.py`。
    - 这三个分支之间没有依赖关系，并行执行可以显著缩短总耗时，尤其是当网络请求和LLM调用成为瓶颈时。

3.  **同步与汇总阶段 (最终决策)**:
    - `main.py` 会等待所有三个并行分支都成功完成后，再启动最后一步 `meta_reviewer.py`。
    - `meta_reviewer.py` 收集所有分支的产出物，进行最终的综合评审。

**后端并发模型 (`api_server.py`):**
- **非阻塞式请求处理**: 服务器采用 `asyncio` 异步框架。当用户通过 `index.html` 发起审稿请求时，`api_server.py` 会立即接收请求，在后台创建一个异步任务来执行 `main.py` 脚本，并迅速向前端返回“任务已启动”的响应。这避免了因长时间运行脚本而导致HTTP请求超时。
- **实时日志流 (WebSocket)**:
    - 每个连接到服务器的前端都会被分配一个唯一的会话ID (SID)。
    - 后端启动 `main.py` 子进程时，会实时捕获其标准输出（stdout）。
    - 每当子进程打印一行日志，`api_server.py` 就会通过Socket.IO将这行日志定向发送给发起任务的那个特定前端客户端（通过SID）。
    - 这种设计实现了前端日志的实时更新，让用户能够清晰地看到后台任务的每一步进展。
### AI审稿智能体

-   `reviewer_1.py`: **审稿人1: 技术质量审查员**
    -   **功能**: 扮演一位严谨的审稿人，只专注于论文的**内部质量**。它会仔细阅读论文全文，评估其技术描述是否严谨、论证是否充分、写作是否清晰、实验设计是否合理。
    -   **输入**: 论文的Markdown全文和核心分析JSON文件。
    -   **输出**: 一份Markdown格式的审稿报告 (`_review_QualityInspector.md`)，从质量角度详细剖析论文的优缺点。

-   `reviewer_2.py`: **审稿人2: 新颖性评估员**
    -   **功能**: 扮演一位关注前沿的审稿人，专注于评估论文的**新颖性和贡献**。它会将论文的创新点与`step4`生成的前沿研究报告进行对比，判断该工作是突破性的、增量式的，还是与现有工作存在较大重叠。
    -   **输入**: 核心分析JSON文件和前沿分析报告JSON文件。
    -   **输出**: 一份Markdown格式的审稿报告 (`_review_NoveltyAssessor.md`)，深入评估论文的原创性和潜在影响力。

-   `meta_reviewer.py`: **主席审稿人 (Meta-Reviewer)**
    -   **功能**: 模拟会议主席或领域主席的角色。它会综合审稿人1（质量）和审稿人2（新颖性）的报告，并参考`step5`中从OpenReview获取的真实审稿人意见，最终形成一份**权威、全面、平衡**的最终审稿意见。最关键的是，它还会预测作者在Rebuttal阶段最可能被攻击的问题，并给出应对建议。
    -   **输入**: 核心分析JSON、两位审稿人的报告以及OpenReview的CSV参考文件。
    -   **输出**: 最终的Meta-Review报告 (`_meta_review.md`)。

## 快速开始
项目需要两个环境完成
1.  **创建环境**:
    ```bash
        conda create -n paper_review python==3.10
        conda activate paper_review
    ```
2.  **安装依赖**:
    建议在虚拟环境中安装。
    ```bash
    pip install -r requirements_main.txt
    ```
3.  **创建环境**:
    ```bash
        conda create -n langchain python==3.10
        conda activate langchain
    ```
4.  **安装依赖**:
    建议在虚拟环境中安装。
    ```bash
    pip install -r requirements_langchain.txt
    which python
    ```
    这个环境request包是不对的，可以先把这个request包从requirements里删了，然后装完之后再安装request，最后虽然报错，但是能运行代码。
    然后把路径copy到config.ini的STEP4_PYTHON_EXECUTABLE
5.  **配置环境**:
    在项目根目录填写一个 `config.ini` 文件。该文件用于存放API密钥和本地模型路径。您需要配置以下内容：
    -   **DashScope (通义千问) API Key**: 用于驱动所有大语言模型分析。
    -   **OpenReview 账号信息**: 用于从OpenReview检索论文。
    -   **Nougat 和 Embedding 模型的本地路径**。
    这个模型不知道可以直接问AI，让他帮你找下载地址。
Nougat:https://github.com/facebookresearch/nougat/releases
这个下载0.1.0-small到文件夹中。
4.  **准备论文**:
    将您需要审稿的PDF论文放置在任意位置。

## 如何使用

通过命令行启动整个审稿流水线。
首先配置环境变量:
export DASHSCOPE_API_KEY="" OPENREVIEW_EMAIL="" OPENREVIEW_PASSWORD=""
后面那个是OpenReview网站的个人账号和密码，用于调用API访问OPENREVIEW。前面那个是千问的API。
**使用示例**:
```bash
本地测试：
python main.py --pdf /path/to/your/paper.pdf --tier CCF-A
其中每个step都是能单独运行的，可以直接单个调试。

上线后直接运行：
python api_server.py 