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
    
    // Ask question
    document.getElementById('askInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            askQuestion(currentPaperId, e.target.value.trim());
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
                <h3 class="paper-title" onclick="openPaperModal('${paper.id}')" style="cursor: pointer;">${escapeHtml(paper.title || '无标题')}</h3>
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
        
        // Reset scroll to top when opening modal
        const modalBody = paperModal.querySelector('.modal-body');
        if (modalBody) {
            modalBody.scrollTop = 0;
        }
        
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
                <h3>链接</h3>
                <p><a href="${paper.url}" target="_blank" style="color: var(--primary);">${paper.url}</a></p>
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
        
        // Load Q&A (with Markdown rendering)
        const qaHtml = paper.qa_pairs && paper.qa_pairs.length > 0 ? 
            paper.qa_pairs.map(qa => `
                <div class="qa-item">
                    <div class="qa-question">Q: ${escapeHtml(qa.question)}</div>
                    <div class="qa-answer markdown-content">${renderMarkdown(qa.answer)}</div>
                </div>
            `).join('') : 
            '<p style="color: var(--text-muted);">暂无问答。请在下方输入问题！</p>';
        
        document.getElementById('qaList').innerHTML = qaHtml;
        document.getElementById('askInput').value = '';
        
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
    } catch (error) {
        console.error('Error loading paper:', error);
        showError('Failed to load paper details');
    }
}

// Ask Question (with streaming)
async function askQuestion(paperId, question) {
    const askInput = document.getElementById('askInput');
    const askLoading = document.getElementById('askLoading');
    const qaList = document.getElementById('qaList');
    
    askInput.disabled = true;
    askLoading.style.display = 'block';
    
    // Create placeholder Q&A item
    const qaItem = document.createElement('div');
    qaItem.className = 'qa-item';
    qaItem.innerHTML = `
        <div class="qa-question">Q: ${escapeHtml(question)}</div>
        <div class="qa-answer markdown-content streaming-answer"></div>
    `;
    qaList.appendChild(qaItem);
    
    const answerDiv = qaItem.querySelector('.qa-answer');
    let fullAnswer = '';
    
    try {
        const response = await fetch(`${API_BASE}/papers/${paperId}/ask_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            // Decode chunk
            const chunk = decoder.decode(value, { stream: true });
            
            // Parse SSE format (data: {...}\n\n)
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.chunk) {
                            fullAnswer += data.chunk;
                            // Update display with streaming effect
                            answerDiv.innerHTML = renderMarkdown(fullAnswer) + '<span class="cursor-blink">▊</span>';
                        } else if (data.done) {
                            // Remove cursor and finalize
                            answerDiv.innerHTML = renderMarkdown(fullAnswer);
                        } else if (data.error) {
                            answerDiv.innerHTML = `<span style="color: var(--danger);">Error: ${escapeHtml(data.error)}</span>`;
                        }
                    } catch (e) {
                        // Ignore JSON parse errors for partial chunks
                    }
                }
            }
        }
        
        // Finalize answer
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
        
        document.getElementById('filterKeywords').value = config.filter_keywords.join(', ');
        document.getElementById('negativeKeywords').value = (config.negative_keywords || []).join(', ');
        document.getElementById('presetQuestions').value = config.preset_questions.join('\n');
        document.getElementById('systemPrompt').value = config.system_prompt;
        
        configModal.classList.add('active');
    } catch (error) {
        console.error('Error loading config:', error);
        showError('Failed to load configuration');
    }
}

async function saveConfig() {
    const keywords = document.getElementById('filterKeywords').value
        .split(',')
        .map(k => k.trim())
        .filter(k => k);
    
    const negativeKeywords = document.getElementById('negativeKeywords').value
        .split(',')
        .map(k => k.trim())
        .filter(k => k);
    
    const questions = document.getElementById('presetQuestions').value
        .split('\n')
        .map(q => q.trim())
        .filter(q => q);
    
    const systemPrompt = document.getElementById('systemPrompt').value.trim();
    
    try {
        await fetch(`${API_BASE}/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filter_keywords: keywords,
                negative_keywords: negativeKeywords,
                preset_questions: questions,
                system_prompt: systemPrompt
            })
        });
        
        closeModal(configModal);
        showSuccess('Configuration saved');
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
        await fetch(`${API_BASE}/papers/${paperId}/star`, {
            method: 'POST'
        });
        
        // Reload papers to show updated state (without scrolling)
        currentPage = 0;
        loadPapers(0, false);  // false = don't scroll
    } catch (error) {
        console.error('Error toggling star:', error);
    }
}

// Hide paper
async function hidePaper(paperId) {
    try {
        await fetch(`${API_BASE}/papers/${paperId}/hide`, {
            method: 'POST'
        });
        
        // Remove from timeline with smooth fade out
        const cards = timeline.querySelectorAll('.paper-card');
        for (const card of cards) {
            const titleEl = card.querySelector('.paper-title');
            if (titleEl && titleEl.getAttribute('onclick')?.includes(paperId)) {
                card.style.transition = 'opacity 0.3s ease-out';
                card.style.opacity = '0';
                setTimeout(() => card.remove(), 300);
                break;
            }
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
        // Parse markdown
        const html = marked.parse(text);
        return html;
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
        // Fetch all starred papers
        const response = await fetch(`${API_BASE}/papers?skip=0&limit=1000&sort_by=starred`);
        const allPapers = await response.json();
        const starredPapers = allPapers.filter(p => p.is_starred);
        
        if (starredPapers.length === 0) return;
        
        // Create collapsible section
        const section = document.createElement('div');
        section.className = 'starred-section';
        section.innerHTML = `
            <div class="starred-header" onclick="toggleStarredSection()">
                <h3>⭐ 收藏的论文 (${starredPapers.length})</h3>
                <span class="toggle-icon" id="starredToggle">▼</span>
            </div>
            <div class="starred-content" id="starredContent" style="display: none;">
                ${starredPapers.map(paper => `
                    <div class="starred-item" onclick="openPaperModal('${paper.id}')">
                        <span class="starred-title">${escapeHtml(paper.title)}</span>
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
        toggle.textContent = '▲';
    } else {
        content.style.display = 'none';
        toggle.textContent = '▼';
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
    loadPapers();
}

