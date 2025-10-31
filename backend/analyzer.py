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

from models import Paper, QAPair, Config


class DeepSeekAnalyzer:
    """
    Two-stage analysis with KV cache optimization.
    
    Key insight: Keep "system_prompt + content" fixed,
    only change the question to maximize cache hits.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        self.data_dir = Path("data/papers")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    async def stage1_filter(self, paper: Paper, config: Config) -> Paper:
        """
        Stage 1: Quick filter.
        Determines if paper is relevant based on keywords.
        First checks negative keywords - if matched, score=1 (irrelevant).
        
        Retry logic: Up to 3 attempts with exponential backoff.
        Failed records are NOT saved.
        """
        # Check negative keywords first (fast path for rejection)
        if config.negative_keywords:
            searchable_text = f"{paper.title} {paper.preview_text}".lower()
            for neg_kw in config.negative_keywords:
                if neg_kw.lower() in searchable_text:
                    paper.is_relevant = False
                    paper.relevance_score = 1.0
                    paper.extracted_keywords = [f"❌ {neg_kw}"]
                    paper.one_line_summary = f"论文包含负面关键词「{neg_kw}」，自动标记为不相关"
                    self._save_paper(paper)
                    print(f"  Stage 1: ✗ Negative keyword '{neg_kw}' matched - {paper.id}")
                    return paper
        
        # Normal analysis if no negative keywords matched
        prompt = f"""分析这篇论文预览，判断它与以下关键词的相关性：

关键词：{', '.join(config.filter_keywords)}

论文标题：{paper.title}
论文预览：
{paper.preview_text}

请用 JSON 格式回答：
{{
    "is_relevant": true/false,
    "relevance_score": 0-10的分数（0=完全不相关，10=高度相关），
    "extracted_keywords": ["关键词1", "关键词2", ...],
    "one_line_summary": "一句话总结（中文）"
}}
"""
        
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
        
        # Build cache prefix (system prompt + paper content)
        # This stays the same for all questions -> KV cache hit
        cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{paper.html_content or paper.abstract}
"""
        
        # 1. Generate detailed summary first
        detailed_summary_question = """请用中文生成这篇论文的详细摘要（约200-300字），包括：
1. 研究背景和动机
2. 核心方法和技术创新
3. 主要实验结果
4. 研究意义和价值

使用 Markdown 格式，让摘要清晰易读。"""
        
        detailed_summary = await self._ask_question_with_retry(
            cache_prefix=cache_prefix,
            question=detailed_summary_question,
            config=config,
            cache_id=paper.id
        )
        
        if detailed_summary is None:
            # Failed after retries - do NOT save
            print(f"  Stage 2 FAILED to generate summary for {paper.id}")
            return paper
        
        paper.detailed_summary = detailed_summary
        print(f"  Stage 2: Generated detailed summary for {paper.id}")
        
        # 2. Ask each preset question
        all_success = True
        for question in config.preset_questions:
            answer = await self._ask_question_with_retry(
                cache_prefix=cache_prefix,
                question=question,
                config=config,
                cache_id=paper.id
            )
            
            if answer is None:
                # Failed after retries
                print(f"  Stage 2 FAILED for {paper.id}, Q: {question[:40]}")
                all_success = False
                break
            
            paper.qa_pairs.append(QAPair(
                question=question,
                answer=answer
            ))
            
            print(f"  Stage 2: Answered '{question[:40]}...' for {paper.id}")
        
        # Save updated paper ONLY if all succeeded
        if all_success:
            self._save_paper(paper)
        else:
            print(f"  Stage 2: Skipping save for {paper.id} due to failures")
        
        return paper
    
    async def ask_custom_question_stream(
        self,
        paper: Paper,
        question: str,
        config: Config,
        parent_qa_id: Optional[int] = None
    ):
        """
        Ask a custom question about a paper with streaming response.
        Yields chunks of the answer as they arrive.
        
        Supports:
        - Reasoning mode: prefix question with "think:" to use deepseek-reasoner
        - Follow-up: provide parent_qa_id to build conversation context
        """
        import re
        
        # Check for reasoning mode (case-insensitive "think:" prefix)
        is_reasoning = False
        original_question = question
        if question.lower().startswith("think:"):
            is_reasoning = True
            question = question[6:].strip()  # Remove "think:" prefix
        
        # Extract arXiv IDs from question (format: [2510.09212] or [2510.09212v1])
        arxiv_id_pattern = r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]'
        referenced_ids = re.findall(arxiv_id_pattern, question)
        
        # If references found, fetch and analyze them
        referenced_papers = []
        id_to_title = {}
        
        if referenced_ids:
            print(f"🔗 Detected {len(referenced_ids)} referenced papers: {referenced_ids}")
            
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
        
        # Build enhanced context
        if referenced_papers:
            enhanced_question = question
            for ref_id, title in id_to_title.items():
                enhanced_question = enhanced_question.replace(f"[{ref_id}]", f'"{title}"')
            
            context_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
                f"Content:\n{paper.html_content or paper.abstract}",
                ""
            ]
            
            for idx, ref_paper in enumerate(referenced_papers, 1):
                context_parts.extend([
                    f"=== REFERENCE PAPER {idx} ===",
                    f"Title: {ref_paper.title}",
                    f"Content:\n{ref_paper.html_content or ref_paper.abstract}",
                    ""
                ])
            
            cache_prefix = "\n".join(context_parts)
            final_question = enhanced_question
            cache_id = f"{paper.id}_with_refs"
        else:
            cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{paper.html_content or paper.abstract}
"""
            final_question = question
            cache_id = paper.id
        
        # Stream the answer (with reasoning support and retry)
        full_answer = ""
        full_thinking = ""
        
        # Choose model based on reasoning mode
        model = "deepseek-reasoner" if is_reasoning else config.model
        
        # Retry logic for streaming (up to 3 attempts)
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                full_answer = ""
                full_thinking = ""
                
                async for chunk in self._ask_question_stream(
                    cache_prefix, 
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
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Stream retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    yield {"type": "error", "chunk": f"⚠️ Connection error, retrying in {wait_time}s...\n"}
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Stream FAILED after {max_retries} attempts: {e}")
                    yield {"type": "error", "chunk": f"❌ Failed after {max_retries} attempts: {str(e)}"}
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
        config: Config
    ) -> str:
        """
        Ask a custom question about a paper.
        Supports cross-paper comparison by detecting arXiv IDs in question (e.g., [2510.09212]).
        Referenced papers will be fetched and included in context.
        """
        import re
        
        # Extract arXiv IDs from question (format: [2510.09212] or [2510.09212v1])
        arxiv_id_pattern = r'\[(\d{4}\.\d{4,5}(?:v\d+)?)\]'
        referenced_ids = re.findall(arxiv_id_pattern, question)
        
        # If references found, fetch and analyze them
        referenced_papers = []
        id_to_title = {}  # Map ID to short title for replacement
        
        if referenced_ids:
            print(f"🔗 Detected {len(referenced_ids)} referenced papers: {referenced_ids}")
            
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
        
        # Build enhanced context
        if referenced_papers:
            # Replace IDs with titles in question
            enhanced_question = question
            for ref_id, title in id_to_title.items():
                enhanced_question = enhanced_question.replace(f"[{ref_id}]", f'"{title}"')
            
            # Build context: current paper + referenced papers
            context_parts = [
                "=== CURRENT PAPER ===",
                f"Title: {paper.title}",
                f"Content:\n{paper.html_content or paper.abstract}",
                ""
            ]
            
            for idx, ref_paper in enumerate(referenced_papers, 1):
                context_parts.extend([
                    f"=== REFERENCE PAPER {idx} ===",
                    f"Title: {ref_paper.title}",
                    f"Content:\n{ref_paper.html_content or ref_paper.abstract}",
                    ""
                ])
            
            cache_prefix = "\n".join(context_parts)
            final_question = enhanced_question
            # Use combined ID for cache (disable cache for multi-paper queries to avoid confusion)
            cache_id = f"{paper.id}_with_refs"
        else:
            # Standard single-paper question
            cache_prefix = f"""Paper Title: {paper.title}

Paper Content:
{paper.html_content or paper.abstract}
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
        
        # Add current question
        messages.append({"role": "user", "content": f"{cache_prefix}\n\nQuestion: {question}"})
        
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
        response = await self.client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": f"{cache_prefix}\n\nQuestion: {question}"}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        
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
        Ask a question with retry logic.
        Returns None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                return await self._ask_question(cache_prefix, question, config, cache_id)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  FAILED after {max_retries} attempts: {e}")
                    return None
    
    def _save_paper(self, paper: Paper):
        """Save paper to JSON file"""
        file_path = self.data_dir / f"{paper.id}.json"
        with open(file_path, 'w') as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
    
    async def process_papers(
        self,
        papers: List[Paper],
        config: Config,
        skip_stage1: bool = False
    ) -> List[Paper]:
        """
        Process multiple papers concurrently.
        
        Stage 1: Filter all papers (fast, all concurrent)
        Stage 2: Deep analysis only for relevant papers (batched by config.concurrent_papers)
        
        Args:
            papers: List of papers to process
            config: Configuration
            skip_stage1: If True, skip Stage 1 and go directly to Stage 2 for all papers
        """
        if not papers:
            return papers
        
        # Stage 1: Filter papers (unless skipped)
        if not skip_stage1:
            print(f"\n🔍 Stage 1: Filtering {len(papers)} papers...")
            stage1_tasks = [self.stage1_filter(paper, config) for paper in papers]
            papers = await asyncio.gather(*stage1_tasks)
            
            # Find relevant papers with score >= min_relevance_score_for_stage2
            min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
            relevant_papers = [
                p for p in papers 
                if p.is_relevant and p.relevance_score >= min_score
            ]
            
            low_score_count = len([p for p in papers if p.is_relevant and p.relevance_score < min_score])
            print(f"✓ Found {len(relevant_papers)} papers with score >= {min_score} for deep analysis")
            if low_score_count > 0:
                print(f"  Skipped {low_score_count} relevant papers with score < {min_score}")
        else:
            # Skip Stage 1, treat all papers as relevant for Stage 2
            print(f"\n🔍 Skipping Stage 1, directly processing {len(papers)} papers for Stage 2...")
            relevant_papers = papers
        
        # Stage 2: Deep analysis for relevant papers
        if relevant_papers:
            concurrent = config.concurrent_papers
            print(f"\n📚 Stage 2: Deep analysis of {len(relevant_papers)} papers (concurrent={concurrent})...")
            
            # Process in batches to control concurrency
            for i in range(0, len(relevant_papers), concurrent):
                batch = relevant_papers[i:i + concurrent]
                print(f"   Processing batch {i//concurrent + 1}/{(len(relevant_papers) + concurrent - 1)//concurrent} ({len(batch)} papers)")
                stage2_tasks = [self.stage2_qa(paper, config) for paper in batch]
                await asyncio.gather(*stage2_tasks)
        
        return papers


async def analyze_new_papers():
    """
    Analyze any papers that haven't been analyzed yet.
    Run this periodically after the fetcher.
    """
    from fetcher import ArxivFetcher
    
    config = Config.load("data/config.json")
    fetcher = ArxivFetcher()
    analyzer = DeepSeekAnalyzer()
    
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

