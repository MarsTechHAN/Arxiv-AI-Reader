# Backend package

# Default configuration
DEFAULT_CONFIG = {
    "filter_keywords": [
        "video diffusion",
        "multimodal generation",
        "unified generation understanding",
        "efficient LLM",
        "efficient diffusion model",
        "diffusion language model",
        "autoregressive diffusion model",
    ],
    "negative_keywords": [
        "medical",
        "healthcare",
        "clinical",
        "protein",
        "molecule",
    ],
    "preset_questions": [
        "这篇论文的核心创新点是什么，他想解决什么问题，怎么解决的？",
        "基于他的前作，梳理这个方向的整个发展脉络，每一步相比于之前的工作都改进了什么，着重于几个不同的发展方向。",
        "他的前作有哪些？使用表格仔细讲讲他的每篇前作，他和前作的区别是什么，主要改善是什么？着重于具体相比于之前文章的改动",
        "论文提出了哪些关键技术方法，请列表格具体详细说明技术细节，需要包含具体的数学原理推导，以及具体参数。",
        "他使用了哪些评价指标与数据集，列表格具体讲讲他的评价指标的细节与数据集的细节",
        "论文在哪些数据集上进行了实验？主要的评估指标和性能提升是多少？",
        "论文的主要局限性有哪些？未来可能的改进方向是什么？",
    ],
    "system_prompt": "你是一个专业的学术论文分析助手请用中文回答所有问题。先仔细阅读下面文章，分析文章的要点，回答要准确、简洁、有深度。使用 Markdown 格式，包括：标题（##）、要点列表（-）、代码块（```）、加粗（**）等，让回答更易读。重点关注论文的技术创新和实际价值。",
    "fetch_interval": 300,
    "max_papers_per_fetch": 100,
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 2000,
    "concurrent_papers": 10,
    "min_relevance_score_for_stage2": 6,  # Minimum score to proceed to Stage 2 deep analysis
    "star_categories": [
        "高效视频生成", "LLM稀疏注意力", "注意力机制", "Roll-out方法"
    ],  # AI classification categories for starred papers (narrowest first)
    "mcp_search_url": None,  # Optional: external MCP search API for AI search candidates
}
