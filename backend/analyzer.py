"""
DeepSeek analyzer - two-stage filtering with KV cache optimization.

Stage 1: Quick filter using preview (abstract + first 2000 chars)
Stage 2: Deep analysis with Q&A (reuses KV cache)
"""

import asyncio
from openai import AsyncOpenAI
from typing import List, Optional
import json
import os
from pathlib import Path
import time
import httpx

from models import Paper, QAPair, Config
from typing import Callable

# Prevent duplicate Stage 2 for same paper when triggered from multiple places (search, startup, etc)
_stage2_in_flight: set = set()
_stage2_lock: asyncio.Lock = asyncio.Lock()


class DeepSeekAnalyzer:
    """
    Two-stage analysis with KV cache optimization.
    Key insight: Keep "system_prompt + content" fixed,
    only change the question to maximize cache hits.
    """

    def __init__(self, api_key: str = None, save_paper: Optional[Callable[[Paper], None]] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        # Configure httpx client with connection pool for high concurrency (stage1+stage2)
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),  # 5min total, 30s connect
            limits=httpx.Limits(
                max_keepalive_connections=200,
                max_connections=500,  # Support stage1(256)+stage2(128) concurrent
            ),
            http2=True,
        )
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            http_client=http_client,
        )
        
        from storage import DEFAULT_DATA_DIR
        self.data_dir = Path(DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._save_paper_cb = save_paper

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Very conservative estimate to avoid exceeding model limits.
        Uses 1.5 chars per token (very conservative for mixed Chinese/English).
        """
        if not text:
            return 0
        # Use 1.5 chars per token for very conservative estimation
        # This accounts for Chinese characters (often 1 char = 1 token) and English (often 4 chars = 1 token)
        return int(len(text) / 1.5)
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        """
        Check if error is a token limit error.
        """
        error_str = str(error)
        return (
            "maximum context length" in error_str.lower() or
            "invalid_request_error" in error_str.lower() and "token" in error_str.lower()
        )
    
    def _truncate_cache_prefix(self, cache_prefix: str, truncate_ratio: float = 0.15) -> str:
        """
        Truncate cache_prefix by a certain ratio from the end.
        Handles multiple formats: single paper, multi-paper, etc.
        
        Args:
            cache_prefix: The content to truncate
            truncate_ratio: Ratio to truncate (0.15 = truncate 15% from end)
        
        Returns:
            Truncated cache_prefix
        """
        if not cache_prefix:
            return cache_prefix
        
        # Calculate truncation length
        truncate_length = int(len(cache_prefix) * truncate_ratio)
        if truncate_length <= 0:
            return cache_prefix
        
        # Try to truncate intelligently: find the last "Content:\n" section and truncate from there
        # This preserves structure better than just truncating from the end
        
        # Check for multi-paper format (=== REFERENCE PAPER ===)
        if "=== REFERENCE PAPER" in cache_prefix:
            # Multi-paper format: truncate from the last paper's content
            parts = cache_prefix.rsplit("Content:\n", 1)
            if len(parts) == 2:
                header = parts[0] + "Content:\n"
                last_content = parts[1]
                if len(last_content) > truncate_length:
                    truncated_content = last_content[:-truncate_length]
                    return header + truncated_content + "\n\n[Content truncated due to token limit]"
        
        # Check for single paper format (Paper Content:\n)
        content_marker = "Paper Content:\n"
        if content_marker in cache_prefix:
            parts = cache_prefix.split(content_marker, 1)
            if len(parts) == 2:
                header = parts[0] + content_marker
                content = parts[1]
                if len(content) > truncate_length:
                    truncated_content = content[:-truncate_length]
                    return header + truncated_content + "\n\n[Content truncated due to token limit]"
        
        # Check for generic "Content:\n" format
        if "Content:\n" in cache_prefix:
            parts = cache_prefix.rsplit("Content:\n", 1)
            if len(parts) == 2:
                header = parts[0] + "Content:\n"
                content = parts[1]
                if len(content) > truncate_length:
                    truncated_content = content[:-truncate_length]
                    return header + truncated_content + "\n\n[Content truncated due to token limit]"
        
        # Fallback: truncate from end
        if len(cache_prefix) > truncate_length:
            return cache_prefix[:-truncate_length] + "\n\n[Content truncated due to token limit]"
        return cache_prefix
    
    def _truncate_content_to_fit_tokens(
        self,
        content: str,
        max_tokens: int,
        reserved_tokens: int = 10000
    ) -> str:
        """
        Truncate content to fit within token limit.
        
        Args:
            content: Content to truncate
            max_tokens: Maximum total tokens (model limit, e.g., 131072)
            reserved_tokens: Tokens reserved for system prompt, question, response, etc.
        
        Returns:
            Truncated content with truncation notice
        """
        available_tokens = max_tokens - reserved_tokens
        if available_tokens <= 0:
            return "[Content too large, cannot fit]"
        
        current_tokens = self._estimate_tokens(content)
        if current_tokens <= available_tokens:
            return content
        
        # Truncate to fit (use 1.5 chars per token to match estimation)
        max_chars = int(available_tokens * 1.5)
        truncated = content[:max_chars]
        return truncated + "\n\n[Content truncated due to token limit]"

    async def ai_search_rerank(
        self,
        query: str,
        candidates: List[dict],
        limit: int = 50,
        config: Optional[Config] = None,
    ) -> List[dict]:
        """
        Use DeepSeek to semantically re-rank candidate papers by relevance to query.
        candidates: list of dicts with id, title, abstract, one_line_summary?, detailed_summary?
        Returns re-ordered list with updated search_score from AI.
        """
        if not candidates:
            return []
        if len(candidates) <= 1:
            return candidates[:limit]

        # Use only summarized content (no raw abstract)
        filtered = [p for p in candidates[:80] if (p.get("one_line_summary") or p.get("detailed_summary"))]
        if not filtered:
            return candidates[:limit]
        papers_text = []
        for i, p in enumerate(filtered):
            title = (p.get("title") or "")[:150]
            one = (p.get("one_line_summary") or "")[:200]
            detailed = (p.get("detailed_summary") or "")[:300]
            block = f"[{i+1}] id={p.get('id','')}\ntitle: {title}\none_line: {one}\nsummary: {detailed}"
            papers_text.append(block)

        prompt = f"""User search query: "{query}"

Below are candidate papers (metadata only). Rank them by semantic relevance to the query.
Return JSON: {{"ranked": [{{"idx": 1-based index, "score": 0-10, "reason": "brief reason"}}, ...]}}
Include ALL papers, score 0 for irrelevant. Order by score descending.

Candidates:
"""
        prompt += "\n\n".join(papers_text)

        model = config.model if config else "deepseek-chat"
        temp = config.temperature if config else 0.2

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a relevance ranker. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            ranked = result.get("ranked", [])
        except Exception as e:
            print(f"AI search rerank failed: {e}, falling back to original order")
            return candidates[:limit]

        score_map = {r["idx"]: float(r.get("score", 0)) for r in ranked}
        ordered = []
        for i, p in enumerate(filtered):
            idx = i + 1
            score = score_map.get(idx, 0.0)
            p_copy = dict(p)
            p_copy["search_score"] = score
            ordered.append((score, p_copy))
        ordered.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in ordered][:limit]

    async def ai_search_with_mcp_tools(
        self,
        query: str,
        tool_executor,
        config: Config,
        limit: int = 50,
        on_progress=None,
    ) -> List[dict]:
        """
        AI search using MCP tools. DeepSeek calls search tools, we execute via tool_executor.
        tool_executor(name, args) -> result. on_progress(msg) for streaming status.
        Returns ranked paper dicts (with full fields filled by caller).
        """
        tools_def = [
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": "Search by keyword in title, abstract, authors, tags, and AI summaries. Fast. Use for: author names, broad topics, arXiv ID lookup. limit 15-25. from_date/to_date: ISO date (2024-01-01) to filter. sort_by: relevance or latest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search terms, author name, or arXiv ID (e.g. 2401.12345)"},
                            "limit": {"type": "integer", "default": 20},
                            "skip": {"type": "integer", "default": 0, "description": "Skip first N for pagination (e.g. 30 for next page)"},
                            "from_date": {"type": "string", "description": "Filter: papers published on or after (YYYY-MM-DD)"},
                            "to_date": {"type": "string", "description": "Filter: papers published on or before (YYYY-MM-DD)"},
                            "sort_by": {"type": "string", "enum": ["relevance", "latest"], "default": "relevance"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_generated_content",
                    "description": "Search ONLY in AI-generated content. Best for semantic queries. from_date/to_date: ISO date. sort_by: relevance or latest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 20},
                            "skip": {"type": "integer", "default": 0, "description": "Skip N for pagination"},
                            "from_date": {"type": "string", "description": "Filter: YYYY-MM-DD"},
                            "to_date": {"type": "string", "description": "Filter: YYYY-MM-DD"},
                            "sort_by": {"type": "string", "enum": ["relevance", "latest"], "default": "relevance"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_full_text",
                    "description": "Search within FULL paper text. Slower. Use for specific techniques/formulas/datasets. from_date/to_date: YYYY-MM-DD. sort_by: relevance or latest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 15},
                            "skip": {"type": "integer", "default": 0},
                            "max_scan": {"type": "integer", "default": 1500},
                            "from_date": {"type": "string", "description": "Filter: YYYY-MM-DD"},
                            "to_date": {"type": "string", "description": "Filter: YYYY-MM-DD"},
                            "sort_by": {"type": "string", "enum": ["relevance", "latest"], "default": "relevance"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_paper_ids_by_query",
                    "description": "Get arXiv IDs only. Fast. from_date/to_date: YYYY-MM-DD. sort_by: relevance or latest.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 30},
                            "skip": {"type": "integer", "default": 0},
                            "from_date": {"type": "string"},
                            "to_date": {"type": "string"},
                            "sort_by": {"type": "string", "enum": ["relevance", "latest"], "default": "relevance"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_ranking",
                    "description": "REQUIRED. Call when done. Submit ranked paper_ids (best first). return_count: 5-20 papers to return. skip: for pagination, skip first N (e.g. 30 for page 2).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "Ranked arXiv IDs, best first"},
                            "return_count": {"type": "integer", "default": 10, "minimum": 5, "maximum": 20, "description": "Number of papers to return (5-20)"},
                            "skip": {"type": "integer", "default": 0, "minimum": 0, "description": "Skip first N for pagination (e.g. 30 for next page)"},
                        },
                        "required": ["paper_ids"],
                    },
                },
            },
        ]

        sys_msg = """You are a paper search assistant for an arXiv corpus. Your goal: return the most relevant papers for the user query.

## TURN LIMIT (CRITICAL)
- You have at most 3 rounds of function calls. Ideally complete in 1 round.
- Prefer calling multiple search tools in parallel (same round) rather than sequential rounds.
- If you need several queries, call them ALL in one batch, then submit_ranking.

## PARALLEL TOOL USE (STRONGLY PREFERRED)
- Call multiple search tools at once when query has multiple aspects (e.g. search_papers("video generation") AND search_generated_content("diffusion models") in one round).
- Call multiple queries in parallel: search_papers("X"), search_papers("Y") together. Merge results, deduplicate, rank, submit_ranking.
- Do NOT do: round1 search A → round2 search B → round3 submit. Instead: round1 search A + search B → round2 submit_ranking.

## TOOL STRATEGY

1. **Choose search type by query:**
   - Conceptual/semantic → search_generated_content
   - Author, arXiv ID, broad keyword → search_papers
   - Specific technique/formula/dataset → search_full_text
   - Use skip for pagination (e.g. skip=30 for next page). return_count 5-20 in submit_ranking.

2. **Date and sort:** from_date/to_date (YYYY-MM-DD), sort_by="relevance" or "latest".

3. **Output: MUST call submit_ranking(paper_ids, return_count=5-20, skip=0).** return_count: how many to return. skip: for pagination.

4. **Reply briefly in Chinese before tool calls.**
"""
        user_msg = f"""User query: {query}

Search the paper corpus. Call multiple search tools in parallel if needed. Merge, deduplicate, rank. Then call submit_ranking(paper_ids, return_count=5-20, skip=0). Complete in 1 round if possible."""

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        max_rounds = 3
        final_ids = None
        return_count = 10
        skip_count = 0
        fallback_ids = []  # IDs from search tools, used when submit_ranking is missing/empty
        SUBMIT_RETRY_MAX = 3

        def collect_ids(result) -> None:
            nonlocal fallback_ids
            if isinstance(result, list):
                for r in result:
                    pid = r.get("id") if isinstance(r, dict) else (r if isinstance(r, str) else None)
                    if pid and pid not in fallback_ids:
                        fallback_ids.append(pid)
            elif isinstance(result, dict) and "id" in result and "error" not in result:
                pid = result.get("id")
                if pid and pid not in fallback_ids:
                    fallback_ids.append(pid)

        async def do_round(tool_choice="auto"):
            nonlocal final_ids, return_count, skip_count
            resp = await self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=tools_def,
                tool_choice=tool_choice,
                temperature=0.2,
                max_tokens=4000,
            )
            choice = resp.choices[0]
            if choice.message.content and choice.message.content.strip() and on_progress:
                await on_progress({"type": "thinking", "text": choice.message.content.strip()})
            if choice.finish_reason == "stop" and not choice.message.tool_calls:
                return "stop"
            if not choice.message.tool_calls:
                return "stop"

            tool_calls = choice.message.tool_calls
            if on_progress and len(tool_calls) > 1:
                await on_progress({"type": "tool_batch", "count": len(tool_calls), "tools": [t.function.name for t in tool_calls]})

            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
                    for t in tool_calls
                ],
            })
            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name == "submit_ranking":
                    final_ids = args.get("paper_ids", []) or []
                    return_count = min(max(int(args.get("return_count", 10)), 5), 20)
                    skip_count = max(0, int(args.get("skip", 0)))
                    return "submit_ok"
                q = args.get("query", "")
                if on_progress:
                    await on_progress({"type": "tool_start", "tool": name, "query": q or args.get("arxiv_id", "")})
                result = await tool_executor(name, args)
                collect_ids(result)
                if on_progress:
                    cnt = len(result) if isinstance(result, list) else (1 if isinstance(result, dict) else 0)
                    await on_progress({"type": "tool_done", "tool": name, "query": (q or args.get("arxiv_id", ""))[:40], "count": cnt})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })
            return "continue"

        for round_idx in range(max_rounds):
            try:
                status = await do_round(tool_choice="auto")
                if status == "submit_ok":
                    break
                if status == "stop":
                    break
            except Exception as e:
                if on_progress:
                    await on_progress({"type": "error", "message": str(e)})
                raise

        # Retry: force submit_ranking when model ended without it (or submitted empty)
        if not final_ids and fallback_ids:
            force_prompts = [
                "You MUST call submit_ranking(paper_ids, return_count=10) now. Use IDs from search results. Non-empty list required.",
                "CRITICAL: Call submit_ranking(paper_ids, return_count=10) immediately. Use IDs from tool results.",
                "Final attempt: submit_ranking(paper_ids) mandatory. return_count 5-20.",
            ]
            for retry_i in range(SUBMIT_RETRY_MAX):
                if final_ids:
                    break
                messages.append({"role": "user", "content": force_prompts[retry_i]})
                if on_progress:
                    await on_progress({"type": "thinking", "text": f"强制提交结果 ({retry_i + 1}/{SUBMIT_RETRY_MAX})..."})
                try:
                    status = await do_round(tool_choice={"type": "function", "function": {"name": "submit_ranking"}})
                    if status == "submit_ok" and final_ids:
                        break
                except Exception as e:
                    if on_progress:
                        await on_progress({"type": "error", "message": str(e)})
                    raise

        if final_ids:
            sliced = final_ids[skip_count:skip_count + return_count]
            return (sliced, return_count, skip_count)
        if fallback_ids:
            return (fallback_ids[:return_count], return_count, 0)
        return ([], 10, 0)

    async def stage1_filter(self, paper: Paper, config: Config) -> Paper:
        """
        Stage 1: Quick filter.
        Determines if paper is relevant based on keywords.
        Negative keywords are passed to LLM as scoring reference only (no auto-blocking).
        Retry logic: Up to 3 attempts with exponential backoff.
        Failed records are NOT saved.
        """
        neg_hint = ""
        if config.negative_keywords:
            neg_hint = f"\n负面关键词（出现则倾向低分，但由你综合判断）：{', '.join(config.negative_keywords)}\n"
        prompt = f"""分析这篇论文预览，判断它与以下关键词的相关性：

关键词：{', '.join(config.filter_keywords)}
{neg_hint}
论文标题：{paper.title}
论文预览：
{paper.preview_text}

请用 JSON 格式回答：
{{
    "is_relevant": true/false,
    "relevance_score": 0-10的分数（0=完全不相关，10=高度相关），
    "extracted_keywords": ["关键词1", "关键词2", ...],
    "one_line_summary": "一句话总结（中文，仅概括论文内容本身，勿提及与关键词的相关性、评分等）"
}}

注意：one_line_summary 必须是对论文内容的客观概括，不得包含"与关键词相关性"、"相关性较弱/较强"等表述。"""
        
        # Retry logic: up to 3 attempts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.temperature,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                paper.is_relevant = result.get("is_relevant", False)
                paper.relevance_score = float(result.get("relevance_score", 0))
                paper.extracted_keywords = result.get("extracted_keywords", [])
                paper.one_line_summary = result.get("one_line_summary", "")
                
                # Save updated paper ONLY on success
                self._save_paper(paper)
                
                score_display = f"({paper.relevance_score}/10)" if paper.relevance_score > 0 else ""
                print(f"  Stage 1: {'✓ Relevant' if paper.is_relevant else '✗ Not relevant'} {score_display} - {paper.id}")
                
                return paper
            
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s
                    wait_time = 2 ** attempt
                    print(f"  Stage 1 retry {attempt + 1}/{max_retries} for {paper.id} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure - do NOT save
                    print(f"  Stage 1 FAILED after {max_retries} attempts for {paper.id}: {e}")
                    paper.is_relevant = None  # Mark as unprocessed
        
        return paper
    
    async def stage2_qa(self, paper: Paper, config: Config) -> Paper:
        """
        Stage 2: Deep Q&A analysis.
        First generates detailed summary, then answers preset questions.
        Uses KV cache by keeping system + content fixed.
        """
        if not paper.is_relevant:
            return paper

        async with _stage2_lock:
            if paper.id in _stage2_in_flight:
                return paper  # Skip: another task is processing this paper
            _stage2_in_flight.add(paper.id)
        try:
            return await self._stage2_qa_impl(paper, config)
        finally:
            _stage2_in_flight.discard(paper.id)

    async def _stage2_qa_impl(self, paper: Paper, config: Config) -> Paper:
        """Internal Stage 2 implementation (called by stage2_qa with in-flight guard)."""
        # Build cache prefix (system prompt + paper content)
        # This stays the same for all questions -> KV cache hit
        # Limit content to fit within model's token limit
        MODEL_MAX_TOKENS = 100000  # Very conservative limit (model max is 131072)
        
        # Calculate fixed parts token count (system prompt + title + format strings)
        # Format: "Paper Title: {title}\n\nPaper Content:\n{content}\n\nQuestion: {question}"
        system_tokens = self._estimate_tokens(config.system_prompt)
        title_format_tokens = self._estimate_tokens(f"Paper Title: {paper.title}\n\nPaper Content:\n")
        question_format_tokens = self._estimate_tokens("\n\nQuestion: ")  # Max question length estimate
        max_question_tokens = 1000  # Reserve for longest question
        response_tokens = config.max_tokens
        overhead_tokens = 1000  # Safety margin
        
        # Calculate available tokens for content
        fixed_tokens = system_tokens + title_format_tokens + question_format_tokens + max_question_tokens + response_tokens + overhead_tokens
        available_content_tokens = MODEL_MAX_TOKENS - fixed_tokens
        
        if available_content_tokens <= 0:
            print(f"  ⚠️  Cannot fit paper {paper.id} even without content (fixed parts: {fixed_tokens} tokens)")
            return paper
        
        # Truncate content to fit
        content = paper.html_content or paper.abstract
        content_tokens = self._estimate_tokens(content)
        
        if content_tokens > available_content_tokens:
            # Truncate to fit (use 1.5 chars per token to match estimation)
            max_chars = int(available_content_tokens * 1.5)
            truncated = content[:max_chars]
            content = truncated + "\n\n[Content truncated due to token limit]"
            print(f"  ⚠️  Content truncated for {paper.id} (from {content_tokens} to ~{available_content_tokens} tokens)")
        
        # 1. Generate detailed summary and tags first
        # Prepare existing tags (from extracted_keywords) for reference
        existing_tags = paper.extracted_keywords if paper.extracted_keywords else []
        existing_tags_str = ", ".join(existing_tags) if existing_tags else "无"
        
        detailed_summary_question = f"""请用中文生成这篇论文的详细摘要（约200-300字），包括：
1. 研究背景和动机
2. 核心方法和技术创新
3. 主要实验结果
4. 研究意义和价值

使用 Markdown 格式，让摘要清晰易读。
摘要必须严格基于论文原文内容进行准确概括，不得出现"与关键词相关性较弱/较强"、"与多数关键词相关性"等与筛选标准相关的表述。

同时，请为这篇论文生成合适的标签（tags）。如果论文中已经存在以下标签，请优先使用这些标签：{existing_tags_str}

请用 JSON 格式回答：
{{
    "summary": "详细摘要（Markdown格式）",
    "tags": ["标签1", "标签2", "标签3", ...]
}}

注意：tags应该包含3-8个关键词，涵盖论文的核心主题、方法和技术。如果给定的标签已经很好地描述了论文，请优先使用它们。"""
        
        # Final token check before asking question
        MODEL_MAX_TOKENS = 131072  # Actual model limit
        system_tokens = self._estimate_tokens(config.system_prompt)
        title_format = f"Paper Title: {paper.title}\n\nPaper Content:\n"
        title_tokens = self._estimate_tokens(title_format)
        content_tokens_final = self._estimate_tokens(content)
        question_tokens = self._estimate_tokens(detailed_summary_question)
        question_format_tokens = self._estimate_tokens("\n\nQuestion: ")
        response_tokens = config.max_tokens
        overhead_tokens = 500
        
        total_tokens = system_tokens + title_tokens + content_tokens_final + question_format_tokens + question_tokens + response_tokens + overhead_tokens
        
        if total_tokens > MODEL_MAX_TOKENS:
            # Further truncate content
            available_for_content = MODEL_MAX_TOKENS - (
                system_tokens + title_tokens + question_format_tokens + question_tokens + response_tokens + overhead_tokens
            )
            if available_for_content > 0:
                max_chars = int(available_for_content * 1.5)
                content = content[:max_chars] + "\n\n[Content truncated due to token limit]"
                # Re-verify after truncation
                content_tokens_final = self._estimate_tokens(content)
                total_tokens = system_tokens + title_tokens + content_tokens_final + question_format_tokens + question_tokens + response_tokens + overhead_tokens
                print(f"  ⚠️  Final truncation in stage2_qa: content reduced to ~{available_for_content} tokens (total: {total_tokens})")
                if total_tokens > MODEL_MAX_TOKENS:
                    # Still too large, truncate more aggressively
                    available_for_content = int((MODEL_MAX_TOKENS - (total_tokens - content_tokens_final)) * 0.9)
                    if available_for_content > 0:
                        max_chars = int(available_for_content * 1.5)
                        content = content[:max_chars] + "\n\n[Content truncated due to token limit]"
                        print(f"  ⚠️  Aggressive truncation in stage2_qa: content reduced to ~{available_for_content} tokens")
                    else:
                        print(f"  ⚠️  Cannot fit question even with empty content (fixed parts: {total_tokens - content_tokens_final} tokens)")
                        return paper
            else:
                print(f"  ⚠️  Cannot fit question even with empty content (fixed parts: {total_tokens - content_tokens_final} tokens)")
                return paper
        
        cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{content}
"""
        
        # 1. Generate summary only if not already present (resume support)
        if not paper.detailed_summary or not paper.detailed_summary.strip():
            summary_response = await self._ask_question_with_retry(
                cache_prefix=cache_prefix,
                question=detailed_summary_question,
                config=config,
                cache_id=paper.id
            )
            
            if summary_response is None:
                print(f"  Stage 2 FAILED to generate summary for {paper.id}")
                return paper
            
            try:
                summary_data = json.loads(summary_response)
                paper.detailed_summary = summary_data.get("summary", summary_response)
                generated_tags = summary_data.get("tags", [])
                if existing_tags:
                    merged_tags = list(existing_tags)
                    for tag in generated_tags:
                        if tag not in merged_tags:
                            merged_tags.append(tag)
                    paper.tags = merged_tags[:10]
                else:
                    paper.tags = generated_tags[:10] if generated_tags else []
                print(f"  Stage 2: Generated summary and {len(paper.tags)} tags for {paper.id}")
            except json.JSONDecodeError:
                paper.detailed_summary = summary_response
                paper.tags = list(existing_tags) if existing_tags else []
                print(f"  Stage 2: Generated summary (JSON parse failed) for {paper.id}")
            self._save_paper(paper)  # Incremental save
        else:
            print(f"  Stage 2: Resuming for {paper.id} (summary exists)")
        
        # 2. Preset questions: summary above was first (pre-fill KV cache); all preset Qs concurrent
        already_answered = {qa.question for qa in paper.qa_pairs}
        pending = [q for q in config.preset_questions if q not in already_answered]
        if not pending:
            return paper

        async def ask_one(q: str) -> tuple[str, Optional[str]]:
            ans = await self._ask_question_with_retry(
                cache_prefix=cache_prefix, question=q, config=config, cache_id=paper.id
            )
            return (q, ans)

        results = await asyncio.gather(*[ask_one(q) for q in pending])
        for q, ans in results:
            if ans is not None:
                paper.qa_pairs.append(QAPair(question=q, answer=ans))
                print(f"  Stage 2: Answered '{q[:40]}...' for {paper.id}")
        if any(ans is not None for _, ans in results):
            self._save_paper(paper)
        
        return paper
    
    async def ask_custom_question_stream(
        self,
        paper: Paper,
        question: str,
        config: Config,
        parent_qa_id: Optional[int] = None,
        fetcher=None,
    ):
        """
        Ask a custom question about a paper with streaming response.
        Yields chunks of the answer as they arrive.
        fetcher: optional, use shared fetcher so ref papers use same store (SQLite).
        """
        import re
        
        is_reasoning = False
        original_question = question
        if question.lower().startswith("think:"):
            is_reasoning = True
            question = question[6:].strip()
        
        arxiv_id_pattern = r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]'
        referenced_ids = re.findall(arxiv_id_pattern, question)
        referenced_papers = []
        id_to_title = {}
        
        if referenced_ids:
            print(f"🔗 Detected {len(referenced_ids)} referenced papers: {referenced_ids}")
            if fetcher is None:
                from fetcher import ArxivFetcher
                fetcher = ArxivFetcher()
            
            for ref_id in referenced_ids:
                try:
                    ref_paper = await fetcher.fetch_single_paper(ref_id)
                    
                    if ref_paper.is_relevant is None:
                        print(f"   📊 Analyzing {ref_id}...")
                        await self.stage1_filter(ref_paper, config)
                    
                    if ref_paper.is_relevant and not ref_paper.detailed_summary:
                        print(f"   📚 Deep analysis for {ref_id}...")
                        await self.stage2_qa(ref_paper, config)
                    
                    referenced_papers.append(ref_paper)
                    short_title = ref_paper.title[:60] + "..." if len(ref_paper.title) > 60 else ref_paper.title
                    id_to_title[ref_id] = short_title
                    print(f"   ✓ {ref_id}: {short_title}")
                
                except Exception as e:
                    print(f"   ✗ Failed to load {ref_id}: {e}")
        
        # Build conversation context for follow-ups
        conversation_history = []
        if parent_qa_id is not None and 0 <= parent_qa_id < len(paper.qa_pairs):
            # Build conversation history (exclude thinking to maintain KV cache consistency)
            current_id = parent_qa_id
            while current_id is not None:
                qa = paper.qa_pairs[current_id]
                # Prepend to history (oldest first)
                conversation_history.insert(0, {
                    "question": qa.question,
                    "answer": qa.answer
                    # Note: thinking is excluded for KV cache consistency
                })
                current_id = qa.parent_qa_id
        
        # Build enhanced context with token limit enforcement
        MODEL_MAX_TOKENS = 100000  # Very conservative limit (model max is 131072)
        
        # Calculate fixed parts token count accurately
        system_tokens = self._estimate_tokens(config.system_prompt)
        conversation_tokens = sum(
            self._estimate_tokens(f"Question: {c['question']}") + self._estimate_tokens(c['answer'])
            for c in conversation_history
        ) if conversation_history else 0
        question_format_tokens = self._estimate_tokens("\n\nQuestion: ")
        max_question_tokens = self._estimate_tokens(question)  # Actual question length
        response_tokens = config.max_tokens
        overhead_tokens = 1000  # Safety margin
        
        # Calculate available tokens for content
        fixed_tokens = system_tokens + conversation_tokens + question_format_tokens + max_question_tokens + response_tokens + overhead_tokens
        available_content_tokens = MODEL_MAX_TOKENS - fixed_tokens
        
        if available_content_tokens <= 0:
            print(f"  ⚠️  Cannot fit question even without content (fixed parts: {fixed_tokens} tokens)")
            return
        
        if referenced_papers:
            enhanced_question = question
            for ref_id, title in id_to_title.items():
                enhanced_question = enhanced_question.replace(f"[{ref_id}]", f'"{title}"')
            
            # Build context with all papers, but truncate each to fit
            # Calculate format tokens for context structure
            format_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
                "Content:\n",
                ""
            ]
            format_tokens = self._estimate_tokens("\n".join(format_parts))
            
            # For each referenced paper, add format tokens
            for idx in range(1, len(referenced_papers) + 1):
                ref_format = f"=== REFERENCE PAPER {idx} ===\nTitle: {referenced_papers[idx-1].title}\nContent:\n\n"
                format_tokens += self._estimate_tokens(ref_format)
            
            # Available tokens for actual content (distribute evenly)
            num_papers = 1 + len(referenced_papers)
            available_content_per_paper = (available_content_tokens - format_tokens) // num_papers
            
            if available_content_per_paper <= 0:
                print(f"  ⚠️  Cannot fit papers even with empty content (format: {format_tokens} tokens)")
                return
            
            context_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
            ]
            
            # Truncate current paper content
            current_content = paper.html_content or paper.abstract
            current_tokens = self._estimate_tokens(current_content)
            if current_tokens > available_content_per_paper:
                max_chars = int(available_content_per_paper * 1.5)
                current_content = current_content[:max_chars] + "\n\n[Content truncated due to token limit]"
                print(f"  ⚠️  Current paper content truncated (from {current_tokens} to ~{available_content_per_paper} tokens)")
            context_parts.append(f"Content:\n{current_content}")
            context_parts.append("")
            
            # Truncate each referenced paper
            for idx, ref_paper in enumerate(referenced_papers, 1):
                context_parts.append(f"=== REFERENCE PAPER {idx} ===")
                context_parts.append(f"Title: {ref_paper.title}")
                ref_content = ref_paper.html_content or ref_paper.abstract
                ref_tokens = self._estimate_tokens(ref_content)
                if ref_tokens > available_content_per_paper:
                    max_chars = int(available_content_per_paper * 1.5)
                    ref_content = ref_content[:max_chars] + "\n\n[Content truncated due to token limit]"
                    print(f"  ⚠️  Reference paper {idx} content truncated (from {ref_tokens} to ~{available_content_per_paper} tokens)")
                context_parts.append(f"Content:\n{ref_content}")
                context_parts.append("")
            
            cache_prefix = "\n".join(context_parts)
            final_question = enhanced_question
            cache_id = f"{paper.id}_with_refs"
        else:
            # Single paper: truncate content to fit
            # Format: "Paper Title: {title}\n\nPaper Content:\n{content}"
            title_format_tokens = self._estimate_tokens(f"Paper Title: {paper.title}\n\nPaper Content:\n")
            available_for_content = available_content_tokens - title_format_tokens
            
            if available_for_content <= 0:
                print(f"  ⚠️  Cannot fit paper even without content (title format: {title_format_tokens} tokens)")
                return
            
            content = paper.html_content or paper.abstract
            content_tokens = self._estimate_tokens(content)
            if content_tokens > available_for_content:
                max_chars = int(available_for_content * 1.5)
                content = content[:max_chars] + "\n\n[Content truncated due to token limit]"
                print(f"  ⚠️  Content truncated (from {content_tokens} to ~{available_for_content} tokens)")
            
            cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{content}
"""
            final_question = question
            cache_id = paper.id
        
        # Stream the answer (with reasoning support and retry)
        full_answer = ""
        full_thinking = ""
        
        # Choose model based on reasoning mode
        model = "deepseek-reasoner" if is_reasoning else config.model
        
        # Retry logic for streaming (up to 3 attempts) with dynamic token truncation
        max_retries = 3
        success = False
        current_cache_prefix = cache_prefix
        
        for attempt in range(max_retries):
            try:
                full_answer = ""
                full_thinking = ""
                
                async for chunk in self._ask_question_stream(
                    current_cache_prefix, 
                    final_question, 
                    config, 
                    cache_id,
                    model=model,
                    is_reasoning=is_reasoning,
                    conversation_history=conversation_history if parent_qa_id is not None else None
                ):
                    # All chunks are now dicts: {"thinking": ...} or {"content": ...}
                    if "thinking" in chunk:
                        full_thinking += chunk["thinking"]
                        yield {"type": "thinking", "chunk": chunk["thinking"]}
                    if "content" in chunk:
                        full_answer += chunk["content"]
                        yield {"type": "content", "chunk": chunk["content"]}
                
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a token limit error
                if self._is_token_limit_error(e):
                    # Truncate cache_prefix by 15% for each retry
                    truncate_ratio = 0.15 * (attempt + 1)  # Increase truncation with each attempt
                    if len(current_cache_prefix) > 1000:  # Only truncate if content is substantial
                        old_length = len(current_cache_prefix)
                        current_cache_prefix = self._truncate_cache_prefix(current_cache_prefix, truncate_ratio)
                        new_length = len(current_cache_prefix)
                        print(f"  ⚠️  Token limit error detected in stream, truncating cache_prefix: {old_length} -> {new_length} chars (attempt {attempt + 1}/{max_retries})")
                        yield {"type": "error", "chunk": f"⚠️ Token limit exceeded, truncating content and retrying (attempt {attempt + 1}/{max_retries})...\n"}
                        
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue  # Retry with truncated content
                        else:
                            print(f"  Stream FAILED after {max_retries} attempts with truncation: {e}")
                            yield {"type": "error", "chunk": f"❌ Failed after {max_retries} attempts with truncation: {str(e)}\n"}
                            return  # Don't save on failure
                    else:
                        # Content too small to truncate further
                        print(f"  Stream FAILED: Content too small to truncate further: {e}")
                        yield {"type": "error", "chunk": f"❌ Content too small to truncate further: {str(e)}\n"}
                        return  # Don't save on failure
                else:
                    # Not a token limit error, use normal retry logic
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  Stream retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                        yield {"type": "error", "chunk": f"⚠️ Connection error, retrying in {wait_time}s...\n"}
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  Stream FAILED after {max_retries} attempts: {e}")
                        yield {"type": "error", "chunk": f"❌ Failed after {max_retries} attempts: {str(e)}\n"}
                        return  # Don't save on failure
        
        # Save to paper ONLY if successful
        if success and (full_answer or full_thinking):
            paper.qa_pairs.append(QAPair(
                question=original_question,
                answer=full_answer,
                thinking=full_thinking if is_reasoning else None,
                is_reasoning=is_reasoning,
                parent_qa_id=parent_qa_id
            ))
            self._save_paper(paper)
    
    async def ask_custom_question(
        self,
        paper: Paper,
        question: str,
        config: Config,
        fetcher=None,
    ) -> str:
        """
        Ask a custom question about a paper. Supports cross-paper comparison.
        fetcher: optional, use shared fetcher so ref papers use same store.
        """
        import re
        
        arxiv_id_pattern = r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]'
        referenced_ids = re.findall(arxiv_id_pattern, question)
        referenced_papers = []
        id_to_title = {}
        
        if referenced_ids:
            print(f"🔗 Detected {len(referenced_ids)} referenced papers: {referenced_ids}")
            if fetcher is None:
                from fetcher import ArxivFetcher
                fetcher = ArxivFetcher()
            
            for ref_id in referenced_ids:
                try:
                    # Fetch paper (or load if exists)
                    ref_paper = await fetcher.fetch_single_paper(ref_id)
                    
                    # Ensure it's analyzed (Stage 1 + 2)
                    if ref_paper.is_relevant is None:
                        print(f"   📊 Analyzing {ref_id}...")
                        await self.stage1_filter(ref_paper, config)
                    
                    if ref_paper.is_relevant and not ref_paper.detailed_summary:
                        print(f"   📚 Deep analysis for {ref_id}...")
                        await self.stage2_qa(ref_paper, config)
                    
                    referenced_papers.append(ref_paper)
                    
                    # Create short title (first 60 chars)
                    short_title = ref_paper.title[:60] + "..." if len(ref_paper.title) > 60 else ref_paper.title
                    id_to_title[ref_id] = short_title
                    print(f"   ✓ {ref_id}: {short_title}")
                
                except Exception as e:
                    print(f"   ✗ Failed to load {ref_id}: {e}")
                    # Continue with available papers
        
        # Build enhanced context with token limit enforcement
        MODEL_MAX_TOKENS = 100000  # Very conservative limit (model max is 131072)
        
        # Calculate fixed parts token count
        system_tokens = self._estimate_tokens(config.system_prompt)
        question_format_tokens = self._estimate_tokens("\n\nQuestion: ")
        max_question_tokens = self._estimate_tokens(question)
        response_tokens = config.max_tokens
        overhead_tokens = 1000
        
        # Calculate available tokens for content
        fixed_tokens = system_tokens + question_format_tokens + max_question_tokens + response_tokens + overhead_tokens
        available_content_tokens = MODEL_MAX_TOKENS - fixed_tokens
        
        if available_content_tokens <= 0:
            raise ValueError(f"Cannot fit question even without content (fixed parts: {fixed_tokens} tokens)")
        
        if referenced_papers:
            # Replace IDs with titles in question
            enhanced_question = question
            for ref_id, title in id_to_title.items():
                enhanced_question = enhanced_question.replace(f"[{ref_id}]", f'"{title}"')
            
            # Calculate format tokens for context structure
            format_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
                "Content:\n",
                ""
            ]
            format_tokens = self._estimate_tokens("\n".join(format_parts))
            
            # For each referenced paper, add format tokens
            for idx in range(1, len(referenced_papers) + 1):
                ref_format = f"=== REFERENCE PAPER {idx} ===\nTitle: {referenced_papers[idx-1].title}\nContent:\n\n"
                format_tokens += self._estimate_tokens(ref_format)
            
            # Available tokens for actual content (distribute evenly)
            num_papers = 1 + len(referenced_papers)
            available_content_per_paper = (available_content_tokens - format_tokens) // num_papers
            
            if available_content_per_paper <= 0:
                raise ValueError(f"Cannot fit papers even with empty content (format: {format_tokens} tokens)")
            
            # Build context: current paper + referenced papers
            context_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
            ]
            
            # Truncate current paper content
            current_content = paper.html_content or paper.abstract
            current_tokens = self._estimate_tokens(current_content)
            if current_tokens > available_content_per_paper:
                max_chars = int(available_content_per_paper * 1.5)
                current_content = current_content[:max_chars] + "\n\n[Content truncated due to token limit]"
            context_parts.append(f"Content:\n{current_content}")
            context_parts.append("")
            
            # Truncate each referenced paper
            for idx, ref_paper in enumerate(referenced_papers, 1):
                context_parts.append(f"=== REFERENCE PAPER {idx} ===")
                context_parts.append(f"Title: {ref_paper.title}")
                ref_content = ref_paper.html_content or ref_paper.abstract
                ref_tokens = self._estimate_tokens(ref_content)
                if ref_tokens > available_content_per_paper:
                    max_chars = int(available_content_per_paper * 1.5)
                    ref_content = ref_content[:max_chars] + "\n\n[Content truncated due to token limit]"
                context_parts.append(f"Content:\n{ref_content}")
                context_parts.append("")
            
            cache_prefix = "\n".join(context_parts)
            final_question = enhanced_question
            # Use combined ID for cache (disable cache for multi-paper queries to avoid confusion)
            cache_id = f"{paper.id}_with_refs"
        else:
            # Standard single-paper question
            # Format: "Paper Title: {title}\n\nPaper Content:\n{content}"
            title_format_tokens = self._estimate_tokens(f"Paper Title: {paper.title}\n\nPaper Content:\n")
            available_for_content = available_content_tokens - title_format_tokens
            
            if available_for_content <= 0:
                raise ValueError(f"Cannot fit paper even without content (title format: {title_format_tokens} tokens)")
            
            content = paper.html_content or paper.abstract
            content_tokens = self._estimate_tokens(content)
            if content_tokens > available_for_content:
                max_chars = int(available_for_content * 1.5)
                content = content[:max_chars] + "\n\n[Content truncated due to token limit]"
            
            cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{content}
"""
            final_question = question
            cache_id = paper.id
        
        answer = await self._ask_question(
            cache_prefix=cache_prefix,
            question=final_question,
            config=config,
            cache_id=cache_id
        )
        
        # Save to paper (save original question)
        paper.qa_pairs.append(QAPair(
            question=question,
            answer=answer
        ))
        self._save_paper(paper)
        
        return answer
    
    async def _ask_question_stream(
        self,
        cache_prefix: str,
        question: str,
        config: Config,
        cache_id: str,
        model: Optional[str] = None,
        is_reasoning: bool = False,
        conversation_history: Optional[list] = None
    ):
        """
        Ask a question with streaming response.
        Yields chunks as they arrive from the API.
        
        For reasoning mode (deepseek-reasoner):
        - Yields {"thinking": chunk} for reasoning_content
        - Yields {"content": chunk} for final content
        
        For normal mode:
        - Yields text chunks directly
        """
        # Build messages with conversation history
        messages = [{"role": "system", "content": config.system_prompt}]
        
        # Add conversation history if provided (for follow-ups)
        if conversation_history:
            for conv in conversation_history:
                messages.append({"role": "user", "content": f"Question: {conv['question']}"})
                messages.append({"role": "assistant", "content": conv['answer']})
        
        # Build user message
        user_content = f"{cache_prefix}\n\nQuestion: {question}"
        
        # Final token check before sending request
        MODEL_MAX_TOKENS = 131072  # Actual model limit
        total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in messages)
        total_tokens += self._estimate_tokens(user_content)
        total_tokens += config.max_tokens  # Add response tokens
        overhead_tokens = 500  # Message formatting overhead
        
        if total_tokens + overhead_tokens > MODEL_MAX_TOKENS:
            # Need to truncate cache_prefix further
            available_for_prefix = MODEL_MAX_TOKENS - (
                sum(self._estimate_tokens(msg["content"]) for msg in messages) +
                self._estimate_tokens(f"\n\nQuestion: {question}") +
                config.max_tokens +
                overhead_tokens
            )
            
            if available_for_prefix > 0:
                max_chars = int(available_for_prefix * 1.5)
                if len(cache_prefix) > max_chars:
                    cache_prefix = cache_prefix[:max_chars] + "\n\n[Content truncated due to token limit]"
                    user_content = f"{cache_prefix}\n\nQuestion: {question}"
                    # Re-verify after truncation
                    total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in messages)
                    total_tokens += self._estimate_tokens(user_content)
                    total_tokens += config.max_tokens + overhead_tokens
                    print(f"  ⚠️  Final truncation: cache_prefix reduced to ~{available_for_prefix} tokens (total: {total_tokens})")
                    if total_tokens > MODEL_MAX_TOKENS:
                        # Still too large, truncate more aggressively
                        available_for_prefix = int((MODEL_MAX_TOKENS - total_tokens + self._estimate_tokens(cache_prefix)) * 0.9)
                        if available_for_prefix > 0:
                            max_chars = int(available_for_prefix * 1.5)
                            cache_prefix = cache_prefix[:max_chars] + "\n\n[Content truncated due to token limit]"
                            user_content = f"{cache_prefix}\n\nQuestion: {question}"
                            print(f"  ⚠️  Aggressive truncation: cache_prefix reduced to ~{available_for_prefix} tokens")
            else:
                raise ValueError(f"Request too large: {total_tokens + overhead_tokens} tokens exceeds limit {MODEL_MAX_TOKENS}")
        
        messages.append({"role": "user", "content": user_content})
        
        response = await self.client.chat.completions.create(
            model=model or config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        
        # Stream response
        async for chunk in response:
            # Check if chunk has choices and delta
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            
            delta = chunk.choices[0].delta
            if not delta:
                continue
            
            if is_reasoning:
                # Reasoning mode: handle both reasoning_content and content
                # Note: deepseek-reasoner may yield both in the same chunk or separately
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    yield {"thinking": delta.reasoning_content}
                if delta.content:
                    yield {"content": delta.content}
            else:
                # Normal mode: yield as dict for consistency
                if delta.content:
                    yield {"content": delta.content}
    
    async def _ask_question(
        self,
        cache_prefix: str,
        question: str,
        config: Config,
        cache_id: str
    ) -> str:
        """
        Ask a question with KV cache optimization.
        
        Key: cache_prefix stays the same, only question changes.
        """
        # Final token check before sending request
        MODEL_MAX_TOKENS = 131072  # Actual model limit
        system_tokens = self._estimate_tokens(config.system_prompt)
        user_content = f"{cache_prefix}\n\nQuestion: {question}"
        user_tokens = self._estimate_tokens(user_content)
        total_tokens = system_tokens + user_tokens + config.max_tokens + 500  # +500 for overhead
        
        if total_tokens > MODEL_MAX_TOKENS:
            # Truncate cache_prefix further
            available_for_prefix = MODEL_MAX_TOKENS - (
                system_tokens +
                self._estimate_tokens(f"\n\nQuestion: {question}") +
                config.max_tokens +
                500
            )
            
            if available_for_prefix > 0:
                max_chars = int(available_for_prefix * 1.5)
                if len(cache_prefix) > max_chars:
                    cache_prefix = cache_prefix[:max_chars] + "\n\n[Content truncated due to token limit]"
                    user_content = f"{cache_prefix}\n\nQuestion: {question}"
                    # Re-verify after truncation
                    user_tokens = self._estimate_tokens(user_content)
                    total_tokens = system_tokens + user_tokens + config.max_tokens + 500
                    print(f"  ⚠️  Final truncation in _ask_question: cache_prefix reduced to ~{available_for_prefix} tokens (total: {total_tokens})")
                    if total_tokens > MODEL_MAX_TOKENS:
                        # Still too large, truncate more aggressively
                        available_for_prefix = int((MODEL_MAX_TOKENS - (system_tokens + self._estimate_tokens(f"\n\nQuestion: {question}") + config.max_tokens + 500)) * 0.9)
                        if available_for_prefix > 0:
                            max_chars = int(available_for_prefix * 1.5)
                            cache_prefix = cache_prefix[:max_chars] + "\n\n[Content truncated due to token limit]"
                            user_content = f"{cache_prefix}\n\nQuestion: {question}"
                            print(f"  ⚠️  Aggressive truncation in _ask_question: cache_prefix reduced to ~{available_for_prefix} tokens")
            else:
                raise ValueError(f"Request too large: {total_tokens} tokens exceeds limit {MODEL_MAX_TOKENS}")
        
        # Check if question asks for JSON format
        use_json_format = "JSON" in question.upper() or "json" in question.lower()
        
        response_kwargs = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if use_json_format:
            response_kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.client.chat.completions.create(**response_kwargs)
        
        return response.choices[0].message.content
    
    async def _ask_question_with_retry(
        self,
        cache_prefix: str,
        question: str,
        config: Config,
        cache_id: str,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Ask a question with retry logic and dynamic token truncation.
        When token limit error occurs, truncate cache_prefix and retry.
        Returns None if all retries fail.
        """
        current_cache_prefix = cache_prefix
        
        for attempt in range(max_retries):
            try:
                return await self._ask_question(current_cache_prefix, question, config, cache_id)
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a token limit error
                if self._is_token_limit_error(e):
                    # Truncate cache_prefix by 15% for each retry
                    truncate_ratio = 0.15 * (attempt + 1)  # Increase truncation with each attempt
                    if len(current_cache_prefix) > 1000:  # Only truncate if content is substantial
                        old_length = len(current_cache_prefix)
                        current_cache_prefix = self._truncate_cache_prefix(current_cache_prefix, truncate_ratio)
                        new_length = len(current_cache_prefix)
                        print(f"  ⚠️  Token limit error detected, truncating cache_prefix: {old_length} -> {new_length} chars (attempt {attempt + 1}/{max_retries})")
                        
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue  # Retry with truncated content
                        else:
                            print(f"  FAILED after {max_retries} attempts with truncation: {e}")
                            return None
                    else:
                        # Content too small to truncate further
                        print(f"  FAILED: Content too small to truncate further: {e}")
                        return None
                else:
                    # Not a token limit error, use normal retry logic
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  FAILED after {max_retries} attempts: {e}")
                        return None
        
        return None
    
    async def classify_starred_paper(self, paper: Paper, config: Config) -> str:
        """
        Classify a starred paper into one of the configured categories using title + abstract.
        When multiple categories fit, choose the NARROWEST one.
        Returns the category name, or 'Other' if none fits.
        """
        categories = getattr(config, 'star_categories', None) or [
            "高效视频生成", "LLM稀疏注意力", "注意力机制", "Roll-out方法"
        ]
        categories = list(categories) + ["Other"]
        category_list_str = ", ".join(f'"{c}"' for c in categories[:-1])
        
        prompt = f"""Classify this paper into EXACTLY ONE category. Choose the NARROWEST/most specific match.

Valid categories (use EXACT string): {category_list_str}, "Other"

Rule: When a paper fits multiple categories (e.g. video sparse attention -> both "efficient video generation" and "attention mechanism"), choose the NARROWEST. Use "Other" only if it fits NONE of the above.

Paper Title: {paper.title}
Abstract: {paper.abstract[:1500]}

Respond with JSON only: {{"category": "exact_category_name"}}"""

        try:
            response = await self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Respond ONLY with valid JSON: {\"category\": \"category_name\"}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content.strip()
            data = json.loads(result)
            cat = data.get("category", "").strip()
            if cat in categories:
                print(f"  Classified {paper.id} -> {cat}")
                return cat
            # Fuzzy fallback: model might return slight variation
            for valid_cat in categories:
                if valid_cat in cat or cat in valid_cat:
                    return valid_cat
            print(f"  Classification fallback to Other for {paper.id}: model returned '{cat}'")
            return "Other"
        except json.JSONDecodeError as e:
            # Fallback: extract category from raw response
            raw = response.choices[0].message.content.strip()
            for cat in categories:
                if cat in raw:
                    print(f"  Classified {paper.id} -> {cat} (from raw parse)")
                    return cat
            print(f"  Classification JSON parse failed for {paper.id}: {e}, fallback to Other")
            return "Other"
        except Exception as e:
            print(f"  Classification error for {paper.id}: {e}")
            return "Other"
    
    def _save_paper(self, paper: Paper):
        """Save paper via store (SQLite/JSON). Must use fetcher.save_paper when using SQLite."""
        if self._save_paper_cb:
            self._save_paper_cb(paper)
        else:
            file_path = self.data_dir / f"{paper.id}.json"
            with open(file_path, 'w') as f:
                json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
    
    async def run_streaming_fetch_and_analyze(
        self, fetcher, config: Config, max_papers_per_category: int = 100
    ) -> int:
        """
        Decoupled pipeline: fetch streams papers to stage1 queue; stage1 passes relevant
        to stage2 queue. Stage1 can start before fetch completes; stage2 can start
        before stage1 completes. Uses separate concurrency: stage1 (default 256),
        stage2 (default 128).
        Returns total papers processed through stage1.
        """
        n1 = getattr(config, "stage1_concurrency", 256)
        n2 = getattr(config, "stage2_concurrency", 128)
        min_score = getattr(config, "min_relevance_score_for_stage2", 6.0)
        stage1_queue: asyncio.Queue = asyncio.Queue()
        stage2_queue: asyncio.Queue = asyncio.Queue()
        sem1 = asyncio.Semaphore(n1)
        sem2 = asyncio.Semaphore(n2)

        async def stage1_worker() -> None:
            while True:
                paper = await stage1_queue.get()
                if paper is None:
                    break
                try:
                    async with sem1:
                        await self.stage1_filter(paper, config)
                    if paper.is_relevant and paper.relevance_score >= min_score:
                        await stage2_queue.put(paper)
                except Exception as e:
                    print(f"  Stage 1 error for {paper.id}: {e}")
                finally:
                    stage1_queue.task_done()

        async def stage2_worker() -> None:
            while True:
                paper = await stage2_queue.get()
                if paper is None:
                    break
                try:
                    async with sem2:
                        await self.stage2_qa(paper, config)
                except Exception as e:
                    print(f"  Stage 2 error for {paper.id}: {e}")
                finally:
                    stage2_queue.task_done()

        n_workers1 = min(n1, 128)
        n_workers2 = min(n2, 128)
        stage1_tasks = [asyncio.create_task(stage1_worker()) for _ in range(n_workers1)]
        stage2_tasks = [asyncio.create_task(stage2_worker()) for _ in range(n_workers2)]
        print(f"\n🔍 Stage 1 workers: {len(stage1_tasks)}, Stage 2 workers: {len(stage2_tasks)}")

        async def run_fetch() -> int:
            async def on_paper(p: Paper) -> None:
                await stage1_queue.put(p)
            papers = await fetcher.fetch_latest(
                max_papers_per_category, on_new_paper=on_paper
            )
            return len(papers)

        fetch_task = asyncio.create_task(run_fetch())
        total_fetched = await fetch_task
        for _ in range(len(stage1_tasks)):
            await stage1_queue.put(None)
        await asyncio.gather(*stage1_tasks)
        for _ in range(len(stage2_tasks)):
            await stage2_queue.put(None)
        await asyncio.gather(*stage2_tasks)
        print(f"✓ Pipeline complete: {total_fetched} papers through stage1")
        return total_fetched

    async def process_papers(
        self,
        papers: List[Paper],
        config: Config,
        skip_stage1: bool = False
    ) -> List[Paper]:
        """
        Process papers with separate stage concurrency. Uses stage1_concurrency
        and stage2_concurrency from config (default 256/128).
        """
        if not papers:
            return papers
        n1 = getattr(config, "stage1_concurrency", 256)
        n2 = getattr(config, "stage2_concurrency", 128)
        min_score = getattr(config, "min_relevance_score_for_stage2", 6.0)
        sem1 = asyncio.Semaphore(n1)
        sem2 = asyncio.Semaphore(n2)

        if not skip_stage1:
            print(f"\n🔍 Stage 1: Filtering {len(papers)} papers (concurrent={n1})...")
            async def stage1_one(p: Paper) -> Paper:
                async with sem1:
                    return await self.stage1_filter(p, config)
            papers = list(await asyncio.gather(*[stage1_one(p) for p in papers]))
            relevant_papers = [p for p in papers if p.is_relevant and p.relevance_score >= min_score]
            low = len([p for p in papers if p.is_relevant and p.relevance_score < min_score])
            print(f"✓ Found {len(relevant_papers)} papers with score >= {min_score} for deep analysis")
            if low > 0:
                print(f"  Skipped {low} relevant papers with score < {min_score}")
        else:
            print(f"\n🔍 Skipping Stage 1, {len(papers)} papers for Stage 2...")
            relevant_papers = papers

        if relevant_papers:
            print(f"\n📚 Stage 2: Deep analysis of {len(relevant_papers)} papers (concurrent={n2})...")
            async def stage2_one(p: Paper) -> Paper:
                async with sem2:
                    return await self.stage2_qa(p, config)
            await asyncio.gather(*[stage2_one(p) for p in relevant_papers])
            print(f"✓ Completed Stage 2 analysis for {len(relevant_papers)} papers")
        return papers


async def analyze_new_papers():
    """
    Analyze any papers that haven't been analyzed yet.
    Run this periodically after the fetcher.
    """
    from fetcher import ArxivFetcher
    
    from storage import DATA_ROOT
    config = Config.load(str(DATA_ROOT / "config.json"))
    fetcher = ArxivFetcher()
    analyzer = DeepSeekAnalyzer(save_paper=fetcher.save_paper)
    
    # Find unanalyzed papers (is_relevant is None)
    all_papers = fetcher.list_papers(limit=1000)
    unanalyzed = [p for p in all_papers if p.is_relevant is None]
    
    if unanalyzed:
        print(f"📊 Analyzing {len(unanalyzed)} unanalyzed papers...")
        await analyzer.process_papers(unanalyzed, config)
    else:
        print("✓ All papers already analyzed")


if __name__ == "__main__":
    # Test analyzer
    asyncio.run(analyze_new_papers())

