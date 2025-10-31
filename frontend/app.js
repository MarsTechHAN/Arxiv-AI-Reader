// API Base URL
const API_BASE = window.location.origin;

// State
let currentPage = 0;
let currentPaperId = null;
let searchTimeout = null;
let currentSortBy = 'relevance';
let currentKeyword = null;
let hasMorePapers = true;
let isLoadingMore = false;
let lastPaperCount = 0;
let lastAnalyzedCount = 0;

// DOM Elements
const timeline = document.getElementById('timeline');
const loading = document.getElementById('loading');
const loadMoreBtn = document.getElementById('loadMore');
const searchInput = document.getElementById('searchInput');
const sortSelect = document.getElementById('sortSelect');
const clearKeywordBtn = document.getElementById('clearKeywordBtn');
const configBtn = document.getElementById('configBtn');
const fetchBtn = document.getElementById('fetchBtn');
const configModal = document.getElementById('configModal');
const paperModal = document.getElementById('paperModal');
const statsEl = document.getElementById('stats');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Restore search state from URL or sessionStorage
    restoreSearchState();
    
    loadPapers();
    loadStats();
    setupEventListeners();
    setupInfiniteScroll();
    checkDeepLink();  // Check if URL has paper ID parameter
    
    // Auto-refresh stats every 30s
    setInterval(loadStats, 30000);
    
    // Check for updates every 30s
    setInterval(checkForUpdates, 30000);
});

// Event Listeners
function setupEventListeners() {
    // Search - input event (with debounce)
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            if (e.target.value.trim()) {
                searchPapers(e.target.value.trim());
            } else {
                currentPage = 0;
                loadPapers();
            }
        }, 500);
    });
    
    // Search - Enter key (immediate search)
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            clearTimeout(searchTimeout);  // Cancel debounced search
            const query = e.target.value.trim();
            if (query) {
                searchPapers(query);
            } else {
                currentPage = 0;
                loadPapers();
            }
        }
    });
    
    // Sort
    sortSelect.addEventListener('change', (e) => {
        currentSortBy = e.target.value;
        currentPage = 0;
        loadPapers();
    });
    
    // Clear keyword filter
    clearKeywordBtn.addEventListener('click', () => {
        currentKeyword = null;
        clearKeywordBtn.style.display = 'none';
        currentPage = 0;
        loadPapers();
    });
    
    // Config button (if exists)
    if (configBtn) {
        configBtn.addEventListener('click', () => openConfigModal());
    }
    
    // Fetch button (if exists)
    if (fetchBtn) {
        fetchBtn.addEventListener('click', () => triggerFetch());
    }
    
    // Config modal (if exists)
    const configModalClose = configModal?.querySelector('.close');
    if (configModalClose) {
        configModalClose.addEventListener('click', (e) => {
            e.stopPropagation();
            closeModal(configModal);
        });
    }
    
    const saveConfigBtn = document.getElementById('saveConfig');
    if (saveConfigBtn) {
        saveConfigBtn.addEventListener('click', () => saveConfig());
    }
    
    // Paper modal - Enhanced close button handling
    const paperModalClose = paperModal?.querySelector('.close');
    if (paperModalClose) {
        paperModalClose.addEventListener('click', (e) => {
            e.stopPropagation();
            closeModal(paperModal);
        });
    }
    
    // Ask question (main input)
    document.getElementById('askInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            askQuestion(currentPaperId, e.target.value.trim(), null);  // null = new question, not follow-up
        }
    });
    
    // Close modals on outside click or ESC key
    [configModal, paperModal].filter(Boolean).forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal(modal);
            }
        });
    });
    
    // ESC key to close modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (paperModal?.classList.contains('active')) {
                closeModal(paperModal);
            } else if (configModal?.classList.contains('active')) {
                closeModal(configModal);
            }
        }
    });
    
    // Share button for paper modal
    const shareBtn = document.getElementById('shareBtn');
    if (shareBtn && paperModal) {
        shareBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sharePaper(currentPaperId);
        });
    }
    
    // Fullscreen toggle for paper modal
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    if (fullscreenBtn && paperModal) {
        fullscreenBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            paperModal.classList.toggle('fullscreen');
        });
    }
    
    // Load more
    loadMoreBtn.addEventListener('click', () => {
        currentPage++;
        loadPapers(currentPage);
    });
    
    // Header title click - clear search and refresh
    const headerTitle = document.getElementById('headerTitle');
    if (headerTitle) {
        headerTitle.addEventListener('click', () => {
            // Clear search input
            searchInput.value = '';
            // Clear keyword filter
            currentKeyword = null;
            clearKeywordBtn.style.display = 'none';
            // Clear search state
            clearSearchState();
            // Reset to first page
            currentPage = 0;
            // Reload papers
            loadPapers();
        });
    }
}

// Infinite scroll
function setupInfiniteScroll() {
    window.addEventListener('scroll', async () => {
        // Check if near bottom
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        // Trigger when 200px from bottom
        const threshold = 200;
        const distanceFromBottom = documentHeight - (scrollTop + windowHeight);
        
        // Only load if: not already loading, has more papers, and near bottom
        if (distanceFromBottom < threshold && !isLoadingMore && hasMorePapers) {
            isLoadingMore = true;
            currentPage++;
            
            try {
                await loadPapers(currentPage);
            } finally {
                isLoadingMore = false;
            }
        }
    });
}

// Load Papers
async function loadPapers(page = 0, shouldScroll = true) {
    showLoading(true);
    
    // Clear search state when loading normal papers list
    if (page === 0) {
        clearSearchState();
    }
    
    try {
        let url = `${API_BASE}/papers?skip=${page * 20}&limit=20&sort_by=${currentSortBy}`;
        if (currentKeyword) {
            url += `&keyword=${encodeURIComponent(currentKeyword)}`;
        }
        
        const response = await fetch(url);
        const papers = await response.json();
        
        if (page === 0) {
            timeline.innerHTML = '';
            hasMorePapers = true;  // Reset state
            hideEndMarker();
            
            // Add starred papers section at the top
            await addStarredPapersSection();
            
            if (shouldScroll) {
                window.scrollTo(0, 0);  // Only scroll when explicitly requested
            }
        }
        
        // Check if we've reached the end
        if (papers.length === 0) {
            hasMorePapers = false;
            if (page > 0) {
                return; // No more papers to add
            }
            // Page 0 with no papers - show empty state
            timeline.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px;">暂无论文</p>';
            return;
        }
        
        if (papers.length < 20) {
            // Last page
            hasMorePapers = false;
        }
        
        // Add papers to timeline (API already excludes starred papers)
        papers.forEach(paper => {
            timeline.appendChild(createPaperCard(paper));
        });
        
        // Show end marker if no more papers
        if (!hasMorePapers && page > 0) {
            showEndMarker();
        }
        
        loadMoreBtn.style.display = 'none';
    } catch (error) {
        console.error('Error loading papers:', error);
        showError('Failed to load papers');
    } finally {
        showLoading(false);
    }
}

// Search Papers
async function searchPapers(query) {
    showLoading(true);
    currentPage = 0;  // Reset page
    hasMorePapers = false;  // Disable infinite scroll for search results
    hideEndMarker();
    
    // Save search state
    saveSearchState(query);
    
    try {
        const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&limit=50`);
        const results = await response.json();
        
        timeline.innerHTML = '';
        window.scrollTo(0, 0);
        
        if (results.length === 0) {
            timeline.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px;">未找到相关论文</p>';
        } else {
            results.forEach(paper => {
                timeline.appendChild(createPaperCard(paper));
            });
            // Show end marker for search results
            showEndMarker();
        }
        
        loadMoreBtn.style.display = 'none';
    } catch (error) {
        console.error('Error searching:', error);
        showError('Search failed');
    } finally {
        showLoading(false);
    }
}

// Create Paper Card
function createPaperCard(paper) {
    const card = document.createElement('div');
    card.className = `paper-card ${paper.is_relevant ? 'relevant' : paper.is_relevant === false ? 'not-relevant' : ''}`;
    card.setAttribute('data-paper-id', paper.id);  // Add paper ID for easy lookup
    
    // Add click event to entire card
    card.style.cursor = 'pointer';
    card.addEventListener('click', () => {
        openPaperModal(paper.id);
    });
    
    // Format date
    let dateStr = '';
    if (paper.published_date) {
        try {
            const date = new Date(paper.published_date);
            dateStr = date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' });
        } catch (e) {
            console.warn('Invalid date:', paper.published_date);
        }
    }
    
    // Relevance score badge
    let scoreBadge = '';
    if (paper.relevance_score > 0) {
        let scoreClass = 'low';
        if (paper.relevance_score >= 7) scoreClass = 'high';
        else if (paper.relevance_score >= 5) scoreClass = 'medium';
        scoreBadge = `<span class="relevance-badge ${scoreClass}">${paper.relevance_score}/10</span>`;
    }
    
    // Only show status for non-relevant or pending papers
    let statusBadge = '';
    if (paper.is_relevant === null) {
        statusBadge = '<span class="paper-status status-pending">⏳ 待分析</span>';
    } else if (paper.is_relevant === false) {
        statusBadge = '<span class="paper-status status-not-relevant">✗ 不相关</span>';
    }
    // Don't show "✓ 相关" for relevant papers
    
    // Stage 2 status (only show "pending" for incomplete analysis)
    const hasDeepAnalysis = paper.detailed_summary && paper.detailed_summary.trim() !== '';
    const stage2Badge = (paper.is_relevant && !hasDeepAnalysis) ? 
        '<span class="stage-badge stage-pending">⏳ 待深度分析</span>' : '';
    
    // Safe authors handling
    const authors = paper.authors || [];
    const authorsText = authors.length > 0 
        ? escapeHtml(authors.slice(0, 3).join(', ')) + (authors.length > 3 ? ' et al.' : '')
        : '作者信息缺失';
    
    card.innerHTML = `
        <div class="paper-header">
            <div style="flex: 1;">
                ${dateStr ? `<p class="paper-date">📅 ${dateStr}</p>` : ''}
                <h3 class="paper-title">${escapeHtml(paper.title || '无标题')}</h3>
                <p class="paper-authors">${authorsText}</p>
            </div>
            <div class="paper-badges" style="display: flex; flex-direction: column; gap: 8px; align-items: flex-end;">
                <span class="relevance-badge-wrapper">${scoreBadge}</span>
                ${statusBadge}
                ${stage2Badge}
            </div>
        </div>
        
        ${paper.one_line_summary ? `
            <div class="paper-summary markdown-content">${renderMarkdown(paper.one_line_summary)}</div>
        ` : `
            <p class="paper-abstract">${escapeHtml(paper.abstract || '摘要缺失')}</p>
        `}
        
        ${paper.extracted_keywords && paper.extracted_keywords.length > 0 ? `
            <div class="paper-keywords">
                ${paper.extracted_keywords.map(kw => 
                    `<span class="keyword" onclick="filterByKeyword('${escapeHtml(kw)}'); event.stopPropagation();">${escapeHtml(kw)}</span>`
                ).join('')}
            </div>
        ` : ''}
        
        <div class="paper-actions" onclick="event.stopPropagation();">
            <button onclick="toggleStar('${paper.id}')" class="${paper.is_starred ? 'starred' : ''}">
                ${paper.is_starred ? '★' : '☆'} ${paper.is_starred ? 'Stared' : 'Star'}
            </button>
            <button onclick="hidePaper('${paper.id}')">🚫 Hide</button>
        </div>
    `;
    
    return card;
}

// Open Paper Modal
async function openPaperModal(paperId) {
    currentPaperId = paperId;
    
    try {
        const response = await fetch(`${API_BASE}/papers/${paperId}`);
        const paper = await response.json();
        
        document.getElementById('paperTitle').textContent = paper.title;
        
        const detailsHtml = `
            <div class="detail-section">
                <h3>作者</h3>
                <p>${escapeHtml(paper.authors.join(', '))}</p>
            </div>
            
            ${paper.detailed_summary ? `
                <div class="detail-section">
                    <h3>AI 详细摘要</h3>
                    <div class="markdown-content">${renderMarkdown(paper.detailed_summary)}</div>
                </div>
            ` : paper.one_line_summary ? `
                <div class="detail-section">
                    <h3>AI 总结</h3>
                    <div class="markdown-content" style="font-size: 16px;">${renderMarkdown(paper.one_line_summary)}</div>
                </div>
            ` : `
                <div class="detail-section">
                    <h3>摘要</h3>
                    <p>${escapeHtml(paper.abstract)}</p>
                </div>
            `}
            
            <div class="detail-section">
                <h3>PDF</h3>
                <div class="paper-links">
                    <a href="${getPdfUrl(paper.url)}" target="_blank" class="pdf-download-link">
                        📄 ${getPdfUrl(paper.url)}
                    </a>
                    <button onclick="togglePdfViewer('${escapeHtml(paper.id)}')" class="btn btn-secondary btn-compact">
                        👁️ 在线预览
                    </button>
                </div>
                <div id="pdfViewerContainer" class="pdf-viewer-container" style="display: none;">
                    <iframe id="pdfViewer" src="" frameborder="0"></iframe>
                </div>
            </div>
            
            ${paper.extracted_keywords && paper.extracted_keywords.length > 0 ? `
                <div class="detail-section">
                    <h3>关键词</h3>
                    <div class="paper-keywords">
                        ${paper.extracted_keywords.map(kw => 
                            `<span class="keyword" onclick="filterByKeyword('${escapeHtml(kw)}'); closeModal(paperModal); event.stopPropagation();">${escapeHtml(kw)}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
        `;
        
        document.getElementById('paperDetails').innerHTML = detailsHtml;
        
        // Load Q&A (with Markdown rendering, thinking support, and follow-up buttons)
        const qaHtml = paper.qa_pairs && paper.qa_pairs.length > 0 ? 
            paper.qa_pairs.map((qa, index) => `
                <div class="qa-item">
                    <div class="qa-question">
                        Q: ${escapeHtml(qa.question)}
                        ${qa.parent_qa_id !== null && qa.parent_qa_id !== undefined ? '<span class="follow-up-badge">↩️ Follow-up</span>' : ''}
                    </div>
                    ${qa.thinking ? `
                        <details class="thinking-section">
                            <summary>🤔 Thinking process</summary>
                            <div class="thinking-content markdown-content">${renderMarkdown(qa.thinking)}</div>
                        </details>
                    ` : ''}
                    <div class="qa-answer markdown-content">${renderMarkdown(qa.answer)}</div>
                    <div class="qa-actions">
                        <button class="btn-follow-up" onclick="startFollowUp(event, ${index})">
                            ↩️ Follow-up
                        </button>
                    </div>
                </div>
            `).join('') : 
            '<p style="color: var(--text-muted);">暂无问答。请在下方输入问题！</p>';
        
        document.getElementById('qaList').innerHTML = qaHtml;
        document.getElementById('askInput').value = '';
        
        // Reset PDF viewer
        const pdfContainer = document.getElementById('pdfViewerContainer');
        const pdfViewer = document.getElementById('pdfViewer');
        if (pdfContainer && pdfViewer) {
            pdfContainer.style.display = 'none';
            pdfViewer.src = '';
        }
        
        // Show relevance editor for non-relevant papers
        const relevanceEditor = document.getElementById('relevanceEditor');
        const currentRelevanceScore = document.getElementById('currentRelevanceScore');
        const relevanceScoreInput = document.getElementById('relevanceScoreInput');
        
        if (paper.is_relevant === false) {
            relevanceEditor.style.display = 'block';
            currentRelevanceScore.textContent = paper.relevance_score || 0;
            relevanceScoreInput.value = paper.relevance_score || 5;
        } else {
            relevanceEditor.style.display = 'none';
        }
        
        paperModal.classList.add('active');
        document.body.classList.add('modal-open');
        
        // Reset scroll to top after modal is active
        setTimeout(() => {
            const modalBody = paperModal.querySelector('.modal-body');
            if (modalBody) {
                modalBody.scrollTop = 0;
            }
        }, 0);
    } catch (error) {
        console.error('Error loading paper:', error);
        showError('Failed to load paper details');
    }
}

// Ask Question (with streaming, reasoning, and follow-up support)
async function askQuestion(paperId, question, parentQaId = null) {
    const askInput = document.getElementById('askInput');
    const askLoading = document.getElementById('askLoading');
    const qaList = document.getElementById('qaList');
    
    askInput.disabled = true;
    askLoading.style.display = 'block';
    
    // Check if it's reasoning mode
    const isReasoning = question.toLowerCase().startsWith('think:');
    
    // Calculate the index for this new QA item (will be added at the end)
    const currentQaIndex = qaList.children.length;
    
    // Create placeholder Q&A item
    const qaItem = document.createElement('div');
    qaItem.className = 'qa-item';
    qaItem.innerHTML = `
        <div class="qa-question">
            Q: ${escapeHtml(question)}
            ${parentQaId !== null ? '<span class="follow-up-badge">↩️ Follow-up</span>' : ''}
        </div>
        ${isReasoning ? `
            <details class="thinking-section" open>
                <summary>🤔 Thinking process...</summary>
                <div class="thinking-content markdown-content streaming-answer"></div>
            </details>
        ` : ''}
        <div class="qa-answer markdown-content streaming-answer"></div>
        <div class="qa-actions">
            <button class="btn-follow-up" onclick="startFollowUp(event, ${currentQaIndex})">
                ↩️ Follow-up
            </button>
        </div>
    `;
    qaList.appendChild(qaItem);
    
    const thinkingDiv = qaItem.querySelector('.thinking-content');
    const answerDiv = qaItem.querySelector('.qa-answer');
    const thinkingSection = qaItem.querySelector('.thinking-section');
    
    let fullAnswer = '';
    let fullThinking = '';
    let pendingUpdate = null;
    let needsUpdate = false;
    
    // Throttled update function using requestAnimationFrame for smooth streaming
    const updateDisplay = (immediate = false) => {
        // Mark that we need an update
        needsUpdate = true;
        
        if (immediate) {
            // Force immediate update, cancel pending animation frame
            if (pendingUpdate !== null) {
                cancelAnimationFrame(pendingUpdate);
                pendingUpdate = null;
            }
            // Update immediately
            if (thinkingDiv && fullThinking) {
                thinkingDiv.innerHTML = renderMarkdown(fullThinking) + '<span class="cursor-blink">▊</span>';
            }
            if (fullAnswer) {
                answerDiv.innerHTML = renderMarkdown(fullAnswer) + '<span class="cursor-blink">▊</span>';
            }
            needsUpdate = false;
        } else if (pendingUpdate === null) {
            // Schedule update using requestAnimationFrame (smoother than setTimeout)
            pendingUpdate = requestAnimationFrame(() => {
                if (needsUpdate) {
                    // Update both thinking and content if they exist
                    if (thinkingDiv && fullThinking) {
                        thinkingDiv.innerHTML = renderMarkdown(fullThinking) + '<span class="cursor-blink">▊</span>';
                    }
                    if (fullAnswer) {
                        answerDiv.innerHTML = renderMarkdown(fullAnswer) + '<span class="cursor-blink">▊</span>';
                    }
                    needsUpdate = false;
                }
                pendingUpdate = null;
            });
        }
    };
    
    try {
        console.log(`[Stream] Starting request: ${API_BASE}/papers/${paperId}/ask_stream`);
        console.log(`[Stream] Question: ${question.substring(0, 50)}..., parentQaId: ${parentQaId}`);
        
        const response = await fetch(`${API_BASE}/papers/${paperId}/ask_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question,
                parent_qa_id: parentQaId
            })
        });
        
        console.log(`[Stream] Response status: ${response.status}, headers:`, response.headers);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`[Stream] Response error: ${response.status} - ${errorText}`);
            answerDiv.innerHTML = `<span style="color: var(--danger);">HTTP Error ${response.status}: ${escapeHtml(errorText)}</span>`;
            return;
        }
        
        if (!response.body) {
            console.error('[Stream] Response body is null!');
            answerDiv.innerHTML = `<span style="color: var(--danger);">No response body</span>`;
            return;
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let buffer = '';
        let chunkCount = 0;
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) {
                console.log(`[Stream] Stream done, processed ${chunkCount} chunks`);
                break;
            }
            
            // Decode chunk and append to buffer (handle partial SSE messages)
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete SSE messages (data: {...}\n\n)
            let newlineIndex;
            while ((newlineIndex = buffer.indexOf('\n\n')) !== -1) {
                const message = buffer.substring(0, newlineIndex);
                buffer = buffer.substring(newlineIndex + 2);
                
                // Skip empty messages
                if (!message.trim()) continue;
                
                // Parse SSE format: "data: {...}"
                if (message.startsWith('data: ')) {
                    try {
                        const jsonStr = message.slice(6);
                        const data = JSON.parse(jsonStr);
                        
                        chunkCount++;
                        if (chunkCount <= 5 || chunkCount % 20 == 0) {
                            console.log(`[Stream] Chunk ${chunkCount}:`, data.type, data.chunk?.substring(0, 30));
                        }
                        
                        if (data.type === 'thinking' && data.chunk) {
                            fullThinking += data.chunk;
                            updateDisplay();
                        } else if (data.type === 'content' && data.chunk) {
                            fullAnswer += data.chunk;
                            updateDisplay();
                        } else if (data.type === 'error' && data.chunk) {
                            // Display error/retry messages inline
                            fullAnswer += data.chunk;
                            updateDisplay(true);  // Force immediate update for errors
                        } else if (data.done) {
                            // Finalize - remove cursors
                            console.log('[Stream] Received done signal');
                            if (thinkingDiv && fullThinking) {
                                thinkingDiv.innerHTML = renderMarkdown(fullThinking);
                                thinkingDiv.classList.remove('streaming-answer');
                                // Auto-collapse thinking after completion
                                setTimeout(() => {
                                    if (thinkingSection) thinkingSection.open = false;
                                }, 500);
                            }
                            if (fullAnswer) {
                                answerDiv.innerHTML = renderMarkdown(fullAnswer);
                                answerDiv.classList.remove('streaming-answer');
                            }
                        } else if (data.error) {
                            // Legacy error format
                            console.error('[Stream] Error:', data.error);
                            answerDiv.innerHTML = `<span style="color: var(--danger);">Error: ${escapeHtml(data.error)}</span>`;
                        }
                    } catch (e) {
                        console.warn(`[Stream] JSON parse error:`, e, `Message:`, message.substring(0, 100));
                        // Continue processing - might be partial chunk
                    }
                } else {
                    console.warn(`[Stream] Unexpected SSE format:`, message.substring(0, 100));
                }
            }
        }
        
        // Final cleanup - force final update
        if (pendingUpdate !== null) {
            cancelAnimationFrame(pendingUpdate);
            pendingUpdate = null;
        }
        
        // Final render without cursor
        if (thinkingDiv && fullThinking) {
            thinkingDiv.innerHTML = renderMarkdown(fullThinking);
            thinkingDiv.classList.remove('streaming-answer');
        }
        answerDiv.innerHTML = renderMarkdown(fullAnswer);
        answerDiv.classList.remove('streaming-answer');
        askInput.value = '';
        
    } catch (error) {
        console.error('Error asking question:', error);
        answerDiv.innerHTML = `<span style="color: var(--danger);">Failed to get answer: ${escapeHtml(error.message)}</span>`;
    } finally {
        askInput.disabled = false;
        askLoading.style.display = 'none';
    }
}

// Config Modal
async function openConfigModal() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        
        // Keywords
        document.getElementById('filterKeywords').value = config.filter_keywords.join(', ');
        document.getElementById('negativeKeywords').value = (config.negative_keywords || []).join(', ');
        
        // Q&A
        document.getElementById('presetQuestions').value = config.preset_questions.join('\n');
        document.getElementById('systemPrompt').value = config.system_prompt;
        
        // Model settings
        document.getElementById('model').value = config.model || 'deepseek-chat';
        document.getElementById('temperature').value = config.temperature || 0.3;
        document.getElementById('maxTokens').value = config.max_tokens || 2000;
        
        // Fetch settings
        document.getElementById('fetchInterval').value = config.fetch_interval || 300;
        document.getElementById('maxPapersPerFetch').value = config.max_papers_per_fetch || 100;
        
        // Analysis settings
        document.getElementById('concurrentPapers').value = config.concurrent_papers || 10;
        document.getElementById('minRelevanceScoreForStage2').value = config.min_relevance_score_for_stage2 || 6;
        
        configModal.classList.add('active');
        document.body.classList.add('modal-open');
    } catch (error) {
        console.error('Error loading config:', error);
        showError('Failed to load configuration');
    }
}

async function saveConfig() {
    // Keywords
    const keywords = document.getElementById('filterKeywords').value
        .split(',')
        .map(k => k.trim())
        .filter(k => k);
    
    const negativeKeywords = document.getElementById('negativeKeywords').value
        .split(',')
        .map(k => k.trim())
        .filter(k => k);
    
    // Q&A
    const questions = document.getElementById('presetQuestions').value
        .split('\n')
        .map(q => q.trim())
        .filter(q => q);
    
    const systemPrompt = document.getElementById('systemPrompt').value.trim();
    
    // Model settings
    const model = document.getElementById('model').value.trim();
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxTokens = parseInt(document.getElementById('maxTokens').value);
    
    // Fetch settings
    const fetchInterval = parseInt(document.getElementById('fetchInterval').value);
    const maxPapersPerFetch = parseInt(document.getElementById('maxPapersPerFetch').value);
    
    // Analysis settings
    const concurrentPapers = parseInt(document.getElementById('concurrentPapers').value);
    const minRelevanceScoreForStage2 = parseFloat(document.getElementById('minRelevanceScoreForStage2').value);
    
    // Validation
    if (isNaN(temperature) || temperature < 0 || temperature > 2) {
        showError('Temperature must be between 0 and 2');
        return;
    }
    if (isNaN(maxTokens) || maxTokens < 100 || maxTokens > 8000) {
        showError('Max Tokens must be between 100 and 8000');
        return;
    }
    if (isNaN(fetchInterval) || fetchInterval < 60) {
        showError('Fetch Interval must be at least 60 seconds');
        return;
    }
    if (isNaN(maxPapersPerFetch) || maxPapersPerFetch < 1 || maxPapersPerFetch > 500) {
        showError('Max Papers Per Fetch must be between 1 and 500');
        return;
    }
    if (isNaN(concurrentPapers) || concurrentPapers < 1 || concurrentPapers > 50) {
        showError('Concurrent Papers must be between 1 and 50');
        return;
    }
    if (isNaN(minRelevanceScoreForStage2) || minRelevanceScoreForStage2 < 0 || minRelevanceScoreForStage2 > 10) {
        showError('Min Relevance Score must be between 0 and 10');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filter_keywords: keywords,
                negative_keywords: negativeKeywords,
                preset_questions: questions,
                system_prompt: systemPrompt,
                model: model,
                temperature: temperature,
                max_tokens: maxTokens,
                fetch_interval: fetchInterval,
                max_papers_per_fetch: maxPapersPerFetch,
                concurrent_papers: concurrentPapers,
                min_relevance_score_for_stage2: minRelevanceScoreForStage2
            })
        });
        
        const result = await response.json();
        
        closeModal(configModal);
        showSuccess(result.message || 'Configuration saved');
    } catch (error) {
        console.error('Error saving config:', error);
        showError('Failed to save configuration');
    }
}

// Trigger Fetch
async function triggerFetch() {
    fetchBtn.disabled = true;
    fetchBtn.textContent = '⏳ Fetching...';
    
    try {
        await fetch(`${API_BASE}/fetch`, { method: 'POST' });
        showSuccess('Fetch triggered! Papers will be updated shortly.');
        
        // Reload after 10 seconds
        setTimeout(() => {
            currentPage = 0;
            loadPapers();
            loadStats();
        }, 10000);
    } catch (error) {
        console.error('Error triggering fetch:', error);
        showError('Failed to trigger fetch');
    } finally {
        setTimeout(() => {
            fetchBtn.disabled = false;
            fetchBtn.textContent = '🔄 Fetch Now';
        }, 2000);
    }
}

// Load Stats
async function loadStats() {
    if (!statsEl) return; // Stats element doesn't exist in current UI
    
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        
        statsEl.innerHTML = `
            📊 总计: ${stats.total_papers} | 
            ✓ 相关: ${stats.relevant_papers} | 
            ⭐ 收藏: ${stats.starred_papers} | 
            ⏳ 待分析: ${stats.pending_analysis}
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Utilities
function closeModal(modal) {
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
}

function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

function showError(message) {
    alert('❌ ' + message);
}

function showSuccess(message) {
    alert('✅ ' + message);
}

// Filter by keyword
function filterByKeyword(keyword) {
    currentKeyword = keyword;
    currentPage = 0;
    clearKeywordBtn.style.display = 'block';
    clearKeywordBtn.textContent = `清除筛选: ${keyword}`;
    loadPapers();
}

// Toggle star
async function toggleStar(paperId) {
    try {
        const response = await fetch(`${API_BASE}/papers/${paperId}/star`, {
            method: 'POST'
        });
        const result = await response.json();
        const isStarred = result.is_starred;
        
        // Update UI: find card by data-paper-id attribute
        const card = document.querySelector(`.paper-card[data-paper-id="${paperId}"]`);
        if (card) {
            const starBtn = card.querySelector('button[onclick*="toggleStar"]');
            if (starBtn) {
                if (isStarred) {
                    starBtn.classList.add('starred');
                    starBtn.innerHTML = '★ Stared';
                } else {
                    starBtn.classList.remove('starred');
                    starBtn.innerHTML = '☆ Star';
                }
            }
            
            // If starred, remove from timeline (will appear in starred section)
            if (isStarred) {
                card.style.transition = 'opacity 0.3s ease-out';
                card.style.opacity = '0';
                setTimeout(() => {
                    card.remove();
                    // Refresh starred section
                    refreshStarredSection();
                }, 300);
            } else {
                // If unstarred, also refresh starred section to remove it
                refreshStarredSection();
            }
        }
        
        // Also update starred items in the starred section
        updateStarredItemButton(paperId, isStarred);
        
    } catch (error) {
        console.error('Error toggling star:', error);
    }
}

// Update star button in starred section (if viewing from there)
function updateStarredItemButton(paperId, isStarred) {
    // If we're in the starred section and unstar, remove the item
    if (!isStarred) {
        const starredItem = document.querySelector(`.starred-item[data-paper-id="${paperId}"]`);
        if (starredItem) {
            starredItem.style.transition = 'opacity 0.3s ease-out';
            starredItem.style.opacity = '0';
            setTimeout(() => starredItem.remove(), 300);
        }
    }
}

// Refresh only the starred section (without reloading timeline)
async function refreshStarredSection() {
    try {
        // Remove existing starred section
        const existingSection = document.querySelector('.starred-section');
        if (existingSection) {
            existingSection.remove();
        }
        
        // Re-add starred section at the top of timeline
        await addStarredPapersSection();
    } catch (error) {
        console.error('Error refreshing starred section:', error);
    }
}

// Hide paper
async function hidePaper(paperId) {
    try {
        await fetch(`${API_BASE}/papers/${paperId}/hide`, {
            method: 'POST'
        });
        
        // Remove from timeline with smooth fade out using data-paper-id
        const card = document.querySelector(`.paper-card[data-paper-id="${paperId}"]`);
        if (card) {
            card.style.transition = 'opacity 0.3s ease-out';
            card.style.opacity = '0';
            setTimeout(() => card.remove(), 300);
        }
        
        // Also remove from starred section if present
        const starredItem = document.querySelector(`.starred-item[data-paper-id="${paperId}"]`);
        if (starredItem) {
            starredItem.style.transition = 'opacity 0.3s ease-out';
            starredItem.style.opacity = '0';
            setTimeout(() => starredItem.remove(), 300);
        }
    } catch (error) {
        console.error('Error hiding paper:', error);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Safe markdown rendering with fallback
function renderMarkdown(text) {
    if (!text || text.trim() === '') {
        return '';
    }
    try {
        // Clean up markdown wrapper artifacts
        let cleanedText = text;
        
        // Remove wrapping ```markdown...``` blocks
        cleanedText = cleanedText.replace(/^```markdown\s*\n([\s\S]*?)\n```$/gm, '$1');
        cleanedText = cleanedText.replace(/^```\s*\n([\s\S]*?)\n```$/gm, '$1');
        
        // Step 1: Protect LaTeX formulas with unique base64-encoded placeholders
        const latexMap = new Map();
        let latexIndex = 0;
        
        // Protect display math ($$...$$)
        cleanedText = cleanedText.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
            const id = `LATEXDISPLAY${latexIndex}BASE64`;
            latexMap.set(id, match);
            latexIndex++;
            return id;
        });
        
        // Protect inline math ($...$)
        cleanedText = cleanedText.replace(/\$([^\$\n]+?)\$/g, (match) => {
            const id = `LATEXINLINE${latexIndex}BASE64`;
            latexMap.set(id, match);
            latexIndex++;
            return id;
        });
        
        // Parse markdown with protected LaTeX
        let html = marked.parse(cleanedText);
        
        // Step 2: Restore LaTeX (replace all occurrences)
        latexMap.forEach((latex, id) => {
            // Use split-join method which is more reliable than regex for this
            html = html.split(id).join(latex);
        });
        
        // Step 3: Create temporary div and render LaTeX
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        
        // Render LaTeX with KaTeX
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(tempDiv, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false,
                errorColor: '#cc0000',
                strict: false
            });
        }
        
        return tempDiv.innerHTML;
    } catch (error) {
        console.error('Markdown parsing error:', error);
        // Fallback: escape HTML and preserve line breaks
        return escapeHtml(text).replace(/\n/g, '<br>');
    }
}

// Update relevance
async function updateRelevance(paperId) {
    const scoreInput = document.getElementById('relevanceScoreInput');
    const score = parseFloat(scoreInput.value);
    
    if (isNaN(score) || score < 0 || score > 10) {
        alert('请输入0-10之间的评分');
        return;
    }
    
    try {
        await fetch(`${API_BASE}/papers/${paperId}/update_relevance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                is_relevant: true,
                relevance_score: score
            })
        });
        
        // Close modal and refresh list
        closeModal(paperModal);
        currentPage = 0;
        loadPapers(0, false);  // Don't scroll
    } catch (error) {
        console.error('Error updating relevance:', error);
    }
}

// End marker functions
function showEndMarker() {
    // Remove existing marker if any
    hideEndMarker();
    
    const marker = document.createElement('div');
    marker.id = 'endMarker';
    marker.className = 'end-marker';
    marker.innerHTML = `
        <div class="end-marker-line"></div>
        <div class="end-marker-text">🎉 已加载全部论文</div>
        <div class="end-marker-line"></div>
    `;
    timeline.appendChild(marker);
}

function hideEndMarker() {
    const existing = document.getElementById('endMarker');
    if (existing) {
        existing.remove();
    }
}

// Share paper - copy URL with paper ID
function sharePaper(paperId) {
    if (!paperId) return;
    
    const shareUrl = `${window.location.origin}${window.location.pathname}?paper=${paperId}`;
    
    // Copy to clipboard (with proper fallback)
    if (navigator.clipboard && navigator.clipboard.writeText) {
        // Modern browsers with clipboard API
        navigator.clipboard.writeText(shareUrl).then(() => {
            showSuccess('分享链接已复制到剪贴板！');
        }).catch((err) => {
            console.error('Clipboard API failed:', err);
            fallbackCopy(shareUrl);
        });
    } else {
        // Fallback for older browsers or non-HTTPS
        fallbackCopy(shareUrl);
    }
}

// Fallback copy method
function fallbackCopy(text) {
    const tempInput = document.createElement('input');
    tempInput.value = text;
    tempInput.style.position = 'fixed';
    tempInput.style.opacity = '0';
    document.body.appendChild(tempInput);
    tempInput.select();
    tempInput.setSelectionRange(0, 99999); // For mobile devices
    
    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showSuccess('分享链接已复制到剪贴板！');
        } else {
            showError('复制失败，请手动复制链接');
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
        showError('复制失败，请手动复制链接');
    }
    
    document.body.removeChild(tempInput);
}

// Check deep link - open paper if URL has ?paper=ID parameter
function checkDeepLink() {
    const urlParams = new URLSearchParams(window.location.search);
    const paperId = urlParams.get('paper');
    
    if (paperId) {
        // Open paper modal after a short delay to ensure page is ready
        setTimeout(() => {
            openPaperModal(paperId);
        }, 500);
    }
}

// Add starred papers collapsible section
async function addStarredPapersSection() {
    try {
        // Fetch only starred papers (optimized request)
        const response = await fetch(`${API_BASE}/papers?skip=0&limit=100&starred_only=true`);
        const starredPapers = await response.json();
        
        if (starredPapers.length === 0) return;
        
        // Create collapsible section (collapsed by default)
        const section = document.createElement('div');
        section.className = 'starred-section';
        section.innerHTML = `
            <div class="starred-header" onclick="toggleStarredSection()">
                <h3>⭐ 收藏的论文 (${starredPapers.length})</h3>
                <span class="toggle-icon" id="starredToggle">▶</span>
            </div>
            <div class="starred-content" id="starredContent" style="display: none;">
                ${starredPapers.map(paper => `
                    <div class="starred-item" data-paper-id="${paper.id}" onclick="openPaperModal('${paper.id}')">
                        <div class="starred-item-content">
                            <div class="starred-title">${escapeHtml(paper.title)}</div>
                            ${paper.one_line_summary ? `
                                <div class="starred-summary">${escapeHtml(paper.one_line_summary)}</div>
                            ` : ''}
                        </div>
                        <span class="starred-score">${paper.relevance_score}/10</span>
                    </div>
                `).join('')}
            </div>
        `;
        
        timeline.appendChild(section);
    } catch (error) {
        console.error('Error loading starred papers:', error);
    }
}

// Toggle starred section
function toggleStarredSection() {
    const content = document.getElementById('starredContent');
    const toggle = document.getElementById('starredToggle');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.textContent = '▼';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▶';
    }
}

// Check for updates (new papers or completed analysis)
async function checkForUpdates() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        
        // Initialize on first check
        if (lastPaperCount === 0) {
            lastPaperCount = stats.total_papers;
            lastAnalyzedCount = stats.analyzed_papers;
            return;
        }
        
        // Check if there are updates
        const hasNewPapers = stats.total_papers > lastPaperCount;
        const hasNewAnalysis = stats.analyzed_papers > lastAnalyzedCount;
        
        if (hasNewPapers || hasNewAnalysis) {
            showUpdateNotification();
            lastPaperCount = stats.total_papers;
            lastAnalyzedCount = stats.analyzed_papers;
        }
    } catch (error) {
        console.error('Error checking for updates:', error);
    }
}

// Show update notification
function showUpdateNotification() {
    const notification = document.getElementById('updateNotification');
    if (notification) {
        notification.style.display = 'flex';
    }
}

// Dismiss update notification
function dismissUpdate() {
    const notification = document.getElementById('updateNotification');
    if (notification) {
        notification.style.display = 'none';
    }
}

// Refresh papers (triggered by update notification)
function refreshPapers() {
    dismissUpdate();
    currentPage = 0;
    
    // Check if there's a search query
    const searchQuery = searchInput.value.trim();
    if (searchQuery) {
        searchPapers(searchQuery);
    } else {
        loadPapers();
    }
}

// Save search state to sessionStorage
function saveSearchState(query) {
    if (query) {
        sessionStorage.setItem('arxiv_search_query', query);
    } else {
        sessionStorage.removeItem('arxiv_search_query');
    }
}

// Clear search state
function clearSearchState() {
    sessionStorage.removeItem('arxiv_search_query');
}

// Restore search state on page load
function restoreSearchState() {
    const savedQuery = sessionStorage.getItem('arxiv_search_query');
    if (savedQuery && searchInput) {
        searchInput.value = savedQuery;
        // Don't auto-trigger search on restore, let user decide
    }
}

// Start follow-up question
function startFollowUp(event, qaIndex) {
    event.stopPropagation();
    
    const qaItem = event.target.closest('.qa-item');
    
    // Check if follow-up input already exists
    let followUpContainer = qaItem.querySelector('.follow-up-container');
    
    if (followUpContainer) {
        // Toggle visibility
        followUpContainer.style.display = followUpContainer.style.display === 'none' ? 'block' : 'none';
        if (followUpContainer.style.display === 'block') {
            followUpContainer.querySelector('input').focus();
        }
        return;
    }
    
    // Create follow-up input container
    followUpContainer = document.createElement('div');
    followUpContainer.className = 'follow-up-container';
    followUpContainer.innerHTML = `
        <div class="follow-up-input-wrapper">
            <input 
                type="text" 
                class="input follow-up-input" 
                placeholder="Ask a follow-up question... (Press Enter to send)"
            >
            <button class="btn-cancel" onclick="this.closest('.follow-up-container').remove()">×</button>
        </div>
        <p class="follow-up-hint">💡 Tip: Use "think:" prefix for reasoning mode</p>
    `;
    
    qaItem.appendChild(followUpContainer);
    
    const input = followUpContainer.querySelector('input');
    input.focus();
    
    // Handle Enter key
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            const question = input.value.trim();
            followUpContainer.remove();
            askQuestion(currentPaperId, question, qaIndex);
        }
    });
}

// Convert arXiv abstract URL to PDF URL
function getPdfUrl(url) {
    if (!url) return '';
    
    // Convert http://arxiv.org/abs/XXXX to http://arxiv.org/pdf/XXXX.pdf
    if (url.includes('arxiv.org/abs/')) {
        return url.replace('/abs/', '/pdf/') + '.pdf';
    }
    
    return url;
}

// Toggle PDF viewer
function togglePdfViewer(paperId) {
    const container = document.getElementById('pdfViewerContainer');
    const viewer = document.getElementById('pdfViewer');
    
    if (container.style.display === 'none') {
        // Get paper URL and convert to PDF
        fetch(`${API_BASE}/papers/${paperId}`)
            .then(res => res.json())
            .then(paper => {
                const pdfUrl = getPdfUrl(paper.url);
                viewer.src = pdfUrl;
                container.style.display = 'block';
            })
            .catch(err => {
                console.error('Error loading PDF:', err);
                showError('无法加载 PDF');
            });
    } else {
        container.style.display = 'none';
        viewer.src = ''; // Clear iframe to stop loading
    }
}

