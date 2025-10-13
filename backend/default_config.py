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
    "preset_questions": [
        "这篇论文的核心创新点是什么？用3-5个要点总结。",
        "论文提出了哪些关键技术方法？请详细说明技术细节。",
        "论文在哪些数据集上进行了实验？主要的评估指标和性能提升是多少？",
        "论文的主要局限性有哪些？未来可能的改进方向是什么？",
        "这篇工作对实际应用的意义是什么？有哪些潜在的应用场景？"
    ],
    "system_prompt": "你是一个专业的学术论文分析助手。请用中文回答所有问题。回答要准确、简洁、有深度。使用 Markdown 格式，包括：标题（##）、要点列表（-）、代码块（```）、加粗（**）等，让回答更易读。重点关注论文的技术创新和实际价值。",
    "fetch_interval": 300,
    "max_papers_per_fetch": 50,
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 2000,
    "concurrent_papers": 5
}
