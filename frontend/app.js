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
let currentTab = 'all';  // 'all' or category name (e.g. '高效视频生成', 'Other')
let starCategories = ['高效视频生成', 'LLM稀疏注意力', '注意力机制', 'Roll-out方法'];
let currentPaperList = [];  // Store current paper list for navigation
let currentPaperIndex = -1;  // Current paper index in the list
let stage2PollInterval = null;
let configInitialState = null;  // Snapshot when config modal opened
let configCloseWarningShown = false;  // For "click again to discard"

// DOM Elements
const timeline = document.getElementById('timeline');
const loading = document.getElementById('loading');
const loadMoreBtn = document.getElementById('loadMore');
const searchInput = document.getElementById('searchInput');
const sortSelect = document.getElementById('sortSelect');
const clearKeywordBtn = document.getElementById('clearKeywordBtn');
const configBtn = document.getElementById('configBtn');
const configModal = document.getElementById('configModal');
const paperModal = document.getElementById('paperModal');
const tabAll = document.getElementById('tabAll');
const categoryTabsContainer = document.getElementById('categoryTabsContainer');
const searchBarWrapper = document.getElementById('searchBarWrapper');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    if (sortSelect) {
        currentSortBy = sortSelect.value;
    }
    const aiRestored = restoreSearchState();
    await loadConfigAndRenderTabs();
    setupEventListeners();
    setupInfiniteScroll();
    setupPullToRefresh();
    if (!aiRestored) loadPapers();
    checkDeepLink();

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            const cached = restoreSearchResults(currentTab);
            if (cached && searchInput?.value === cached.query && (!currentPaperList || currentPaperList.length === 0) && cached.results.length > 0) {
                renderSearchResults(cached.results);
            }
        }
    });
});

// Event Listeners
function setupEventListeners() {
    // Search - input event (with debounce). ai: queries only run on Enter.
    searchInput.addEventListener('input', (e) => {
        const val = e.target.value.trim();
        if (/^ai[:：]\s*/i.test(val)) return;  // ai: search only on Enter
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            if (val) {
                searchPapers(val);
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
    
    // Drag-and-drop PDF onto search bar
    if (searchBarWrapper) {
        searchBarWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer.types.includes('Files')) {
                searchBarWrapper.classList.add('drag-over');
            }
        });
        searchBarWrapper.addEventListener('dragleave', (e) => {
            if (!searchBarWrapper.contains(e.relatedTarget)) {
                searchBarWrapper.classList.remove('drag-over');
            }
        });
        searchBarWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            searchBarWrapper.classList.remove('drag-over');
            const files = e.dataTransfer?.files;
            if (!files || files.length === 0) return;
            const pdfFile = Array.from(files).find(f => f.name.toLowerCase().endsWith('.pdf'));
            if (pdfFile) {
                uploadAndParsePdf(pdfFile);
            } else {
                showError('请拖入 PDF 文件');
            }
        });
    }
    
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
    
    // Config modal close (with unsaved changes check)
    const configModalClose = document.getElementById('configModalClose');
    if (configModalClose) {
        configModalClose.addEventListener('click', (e) => {
            e.stopPropagation();
            handleConfigModalClose();
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
    
    // Close paper modal on outside click (config modal does NOT close on outside click)
    if (paperModal) {
        paperModal.addEventListener('click', (e) => {
            if (e.target === paperModal) {
                closeModal(paperModal);
            }
        });
    }
    
    // ESC key to close modals and PDF preview
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const fullscreenViewer = document.getElementById('fullscreenPdfViewer');
            if (fullscreenViewer && fullscreenViewer.style.display !== 'none') {
                closeFullscreenPdf();
            } else if (paperModal?.classList.contains('active')) {
                closeModal(paperModal);
            } else if (configModal?.classList.contains('active')) {
                handleConfigModalClose();
            }
        }
    });
    
    // Star button for paper modal
    const starModalBtn = document.getElementById('starModalBtn');
    if (starModalBtn && paperModal) {
        starModalBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleStarFromModal(currentPaperId);
        });
    }
    
    // Share button for paper modal
    const shareBtn = document.getElementById('shareBtn');
    if (shareBtn && paperModal) {
        shareBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sharePaper(currentPaperId);
        });
    }
    
    // Export button for paper modal
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn && paperModal) {
        exportBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            exportPaperToMarkdown(currentPaperId);
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
    
    // Keyboard navigation for paper modal (always enabled when modal is active)
    document.addEventListener('keydown', (e) => {
        if (paperModal?.classList.contains('active')) {
            // Check if input/textarea is focused (don't navigate when typing)
            const activeElement = document.activeElement;
            const isInputFocused = activeElement && (
                activeElement.tagName === 'INPUT' || 
                activeElement.tagName === 'TEXTAREA' ||
                activeElement.isContentEditable
            );
            
            if (!isInputFocused) {
                if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                    e.preventDefault();
                    navigateToPaper(-1);  // Previous paper
                } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    navigateToPaper(1);  // Next paper
                }
            }
        }
    });
    
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
    
    // Tab switching
    if (tabAll) {
        tabAll.addEventListener('click', () => switchTab('all'));
    }
}

async function loadConfigAndRenderTabs() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        starCategories = config.star_categories || ['高效视频生成', 'LLM稀疏注意力', '注意力机制', 'Roll-out方法'];
        renderCategoryTabs();
    } catch (e) {
        console.warn('Failed to load config for tabs:', e);
        renderCategoryTabs();
    }
}

function renderCategoryTabs() {
    if (!categoryTabsContainer) return;
    categoryTabsContainer.innerHTML = '';
    const categories = [...starCategories, 'Other'];
    categories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'tab-btn';
        btn.dataset.tab = cat;
        btn.textContent = cat;
        btn.title = `收藏 · ${cat}`;
        btn.addEventListener('click', () => switchTab(cat));
        categoryTabsContainer.appendChild(btn);
    });
}

// Pull-to-refresh (fix for category tabs)
function setupPullToRefresh() {
    const indicator = document.getElementById('pullRefreshIndicator');
    if (!indicator) return;
    
    let startY = 0;
    let pulling = false;
    
    const handleStart = (e) => {
        if (window.scrollY <= 10) {
            startY = e.touches ? e.touches[0].clientY : e.clientY;
            pulling = true;
        }
    };
    
    const handleMove = (e) => {
        if (!pulling || window.scrollY > 10) return;
        const y = e.touches ? e.touches[0].clientY : e.clientY;
        const pullDist = y - startY;
        if (pullDist > 60) {
            indicator.classList.add('visible');
        } else if (pullDist < 30) {
            indicator.classList.remove('visible');
        }
    };
    
    const doRefresh = () => {
        indicator.classList.add('loading');
        currentPage = 0;
        hasMorePapers = true;
        const searchQuery = searchInput.value.trim();
        (searchQuery ? searchPapers(searchQuery) : loadPapers(0, true))
            .finally(() => indicator.classList.remove('visible', 'loading'));
    };
    
    const handleEnd = () => {
        if (!pulling) return;
        pulling = false;
        if (indicator.classList.contains('visible')) doRefresh();
        else indicator.classList.remove('visible');
    };
    
    document.addEventListener('touchstart', handleStart, { passive: true });
    document.addEventListener('touchmove', handleMove, { passive: true });
    document.addEventListener('touchend', handleEnd);
    
    // Mouse support for desktop
    document.addEventListener('mousedown', (e) => {
        if (window.scrollY <= 10) {
            startY = e.clientY;
            pulling = true;
        }
    });
    document.addEventListener('mousemove', (e) => {
        if (!pulling || window.scrollY > 10) return;
        const pullDist = e.clientY - startY;
        if (pullDist > 60) indicator.classList.add('visible');
        else if (pullDist < 30) indicator.classList.remove('visible');
    });
    document.addEventListener('mouseup', () => {
        if (pulling && indicator.classList.contains('visible')) doRefresh();
        pulling = false;
        indicator.classList.remove('visible');
    });
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

// Switch Tab
function switchTab(tab) {
    if (currentTab === tab) return;

    currentTab = tab;
    currentPage = 0;
    hasMorePapers = true;

    if (tabAll) tabAll.classList.toggle('active', tab === 'all');
    categoryTabsContainer?.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    const cached = restoreSearchResults(tab);
    if (cached && cached.query) {
        searchInput.value = cached.query;
        currentKeyword = null;
        clearKeywordBtn.style.display = 'none';
        hasMorePapers = false;
        hideEndMarker();
        renderSearchResults(cached.results);
    } else {
        searchInput.value = '';
        currentKeyword = null;
        clearKeywordBtn.style.display = 'none';
        loadPapers(0, true);
    }
}

// Load Papers
async function loadPapers(page = 0, shouldScroll = true) {
    showLoading(true);
    
    // Clear search state when loading normal papers list
    if (page === 0) {
        clearSearchState();
    }
    
    try {
        const isCategoryTab = currentTab !== 'all';
        let url = `${API_BASE}/papers?skip=${page * 20}&limit=20&sort_by=${currentSortBy}&starred_only=${isCategoryTab ? 'true' : 'false'}`;
        if (isCategoryTab) {
            url += `&category=${encodeURIComponent(currentTab)}`;
        }
        if (currentKeyword) {
            url += `&keyword=${encodeURIComponent(currentKeyword)}`;
        }
        
        const response = await fetch(url);
        const papers = await response.json();
        
        if (page === 0) {
            timeline.innerHTML = '';
            currentPaperList = [];  // Reset paper list
            hasMorePapers = true;  // Reset state
            hideEndMarker();
            
            if (shouldScroll) {
                window.scrollTo(0, 0);  // Only scroll when explicitly requested
            }
        }
        
        // Check if we've reached the end
        if (papers.length === 0) {
            hasMorePapers = false;
            if (page > 0) {
                return;
            }
            const emptyMessage = isCategoryTab 
                ? `<p style="text-align: center; color: var(--text-muted); padding: 40px;">暂无「${escapeHtml(currentTab)}」分类的论文</p>`
                : '<p style="text-align: center; color: var(--text-muted); padding: 40px;">暂无论文</p>';
            timeline.innerHTML = emptyMessage;
            return;
        }
        
        if (papers.length < 20) {
            // Last page
            hasMorePapers = false;
        }
        
        // Add papers to timeline
        papers.forEach(paper => {
            timeline.appendChild(createPaperCard(paper));
        });
        
        // Update current paper list for navigation
        if (page === 0) {
            currentPaperList = papers.map(p => p.id);
        } else {
            // Append new papers to the list
            papers.forEach(paper => {
                if (!currentPaperList.includes(paper.id)) {
                    currentPaperList.push(paper.id);
                }
            });
        }
        
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

// Upload and parse PDF file
async function uploadAndParsePdf(file) {
    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE}/upload_pdf`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || 'Upload failed');
        }
        
        const results = await response.json();
        
        if (results.length === 0) {
            showError('PDF 解析失败');
            return;
        }
        
        timeline.innerHTML = '';
        currentPaperList = [];
        window.scrollTo(0, 0);
        
        results.forEach(paper => {
            timeline.appendChild(createPaperCard(paper));
        });
        currentPaperList = results.map(p => p.id);
        showEndMarker();
        loadMoreBtn.style.display = 'none';
        
        showSuccess('PDF 已解析，正在分析中...');
        openPaperModal(results[0].id);
    } catch (error) {
        console.error('Error uploading PDF:', error);
        showError('上传失败: ' + (error.message || 'Unknown error'));
    } finally {
        showLoading(false);
    }
}

// Search Papers
async function searchPapers(query) {
    const isAiSearch = /^ai[:：]\s*/i.test(query);
    showLoading(true);
    currentPage = 0;
    hasMorePapers = false;
    hideEndMarker();

    if (isAiSearch) {
        await searchPapersAiStream(query);
    } else {
        try {
            const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&limit=50&sort_by=${currentSortBy || 'relevance'}`);
            const results = await response.json();
            saveSearchResults(query, results || []);
            renderSearchResults(results);
        } catch (error) {
            console.error('Error searching:', error);
            showError('Search failed');
        }
    }
    showLoading(false);
}

function _searchStateKey(tab) {
    return `search_state_${tab || currentTab || 'all'}`;
}

function saveSearchResults(query, results) {
    try {
        const key = _searchStateKey(currentTab);
        sessionStorage.setItem(key, JSON.stringify({ query: query || '', results: results || [] }));
    } catch (_) {}
}

function restoreSearchResults(tab) {
    try {
        const t = tab || currentTab || 'all';
        const raw = sessionStorage.getItem(_searchStateKey(t));
        if (raw) {
            const data = JSON.parse(raw);
            if (data && Array.isArray(data.results)) {
                return { query: data.query || '', results: data.results };
            }
        }
    } catch (_) {}
    return null;
}

async function searchPapersAiStream(query) {
    let statusContainer = document.getElementById('searchStatusContainer');
    if (!statusContainer) {
        const container = document.querySelector('.search-container .container');
        if (container) {
            statusContainer = document.createElement('div');
            statusContainer.id = 'searchStatusContainer';
            statusContainer.className = 'search-status-container';
            container.appendChild(statusContainer);
        }
    }
    if (!statusContainer) return;

    timeline.innerHTML = '';
    currentPaperList = [];
    window.scrollTo(0, 0);
    statusContainer.innerHTML = '';
    statusContainer.style.display = 'block';

    const logEl = document.createElement('div');
    logEl.className = 'ai-search-log';
    statusContainer.appendChild(logEl);

    const scrollToBottom = () => {
        if (statusContainer) statusContainer.scrollTop = statusContainer.scrollHeight;
    };

    const appendItem = (el) => {
        logEl.appendChild(el);
        scrollToBottom();
    };

    const toolNameMap = {
        search_papers: '关键词搜索',
        search_generated_content: 'AI 总结搜索',
        search_full_text: '全文搜索',
        get_paper_ids_by_query: '获取 ID 列表',
        get_paper: '获取论文',
        submit_ranking: '提交结果',
    };

    const addThinking = (text) => {
        const p = document.createElement('div');
        p.className = 'ai-search-thinking';
        p.textContent = text;
        appendItem(p);
    };

    const formatToolResult = (tool, count) => {
        if (count === undefined) return '';
        if (tool === 'get_paper') return count ? '已获取' : '—';
        if (tool === 'submit_ranking') return count ? `${count} 篇` : '';
        return count ? `${count} 篇` : '0 篇';
    };

    const addToolChip = (tool, query, status, count) => {
        const chip = document.createElement('div');
        const label = toolNameMap[tool] || tool;
        chip.className = `ai-tool-chip ${status}`;
        chip.innerHTML = `
            <span class="tool-icon">${status === 'done' ? '✓' : '<span class="tool-spinner"></span>'}</span>
            <span class="tool-name">${escapeHtml(label)}</span>
            ${query ? `<span class="tool-query" title="${escapeHtml(String(query))}">${escapeHtml(String(query))}</span>` : '<span class="tool-query"></span>'}
            ${status === 'done' && count !== undefined ? `<span class="tool-result">${escapeHtml(formatToolResult(tool, count))}</span>` : ''}
        `;
        chip.dataset.tool = tool;
        chip.dataset.query = query || '';
        appendItem(chip);
        return chip;
    };

    const addToolBatch = (count, tools) => {
        const batch = document.createElement('div');
        batch.className = 'ai-search-batch-label';
        batch.style.cssText = 'font-size: 12px; color: var(--text-muted); padding: 4px 0;';
        const labels = (tools || []).map(t => toolNameMap[t] || t).slice(0, 3);
        batch.textContent = `并行执行 ${count} 个工具${labels.length ? '：' + labels.join('、') : ''}`;
        appendItem(batch);
    };

    const pendingToolChips = [];
    const finishWithResults = (results) => {
        statusContainer.style.display = 'none';
        saveSearchResults(query, results || []);
        if (results && results.length > 0) {
            renderSearchResults(results);
        } else {
            timeline.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px;">未找到相关论文</p>';
        }
    };

    const doStreamSearch = async () => {
        const params = new URLSearchParams({ q: query, limit: '50', sort_by: (currentSortBy || 'relevance') });
        const response = await fetch(`${API_BASE}/search/ai/stream?${params}`);
        if (!response.ok) throw new Error(response.statusText);
        if (!response.body) throw new Error('No stream');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let results = null;
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let idx;
            while ((idx = buffer.indexOf('\n\n')) !== -1) {
                const msg = buffer.substring(0, idx);
                buffer = buffer.substring(idx + 2);
                if (!msg.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(msg.slice(6));
                    if (data.type === 'thinking' && data.text) addThinking(data.text);
                    else if (data.type === 'tool_start') pendingToolChips.push(addToolChip(data.tool, data.query, 'running'));
                    else if (data.type === 'tool_done') {
                        const chip = pendingToolChips.shift();
                        const label = toolNameMap[data.tool] || data.tool;
                        const result = formatToolResult(data.tool, data.count);
                        if (chip) {
                            chip.className = 'ai-tool-chip done';
                            chip.innerHTML = `
                                <span class="tool-icon">✓</span>
                                <span class="tool-name">${escapeHtml(label)}</span>
                                ${(data.query || '') ? `<span class="tool-query" title="${escapeHtml(String(data.query))}">${escapeHtml(String(data.query))}</span>` : '<span class="tool-query"></span>'}
                                ${result ? `<span class="tool-result">${escapeHtml(result)}</span>` : ''}
                            `;
                        } else {
                            addToolChip(data.tool, data.query, 'done', data.count);
                        }
                        scrollToBottom();
                    } else if (data.type === 'tool_batch') addToolBatch(data.count || 0, data.tools);
                    else if (data.type === 'progress') addThinking(data.message || 'AI 处理中...');
                    else if (data.type === 'done') results = data.results || [];
                    else if (data.type === 'error') throw new Error(data.message || 'Search failed');
                } catch (e) {
                    if (!(e instanceof SyntaxError)) throw e;
                }
            }
        }
        return results;
    };

    const doNonStreamSearch = async () => {
        addThinking('正在使用备用模式搜索（适合后台标签页）...');
        const params = new URLSearchParams({ q: query, limit: '50', sort_by: (currentSortBy || 'relevance') });
        const response = await fetch(`${API_BASE}/search/ai?${params}`);
        if (!response.ok) throw new Error(response.statusText);
        return await response.json();
    };

    try {
        let results = null;
        try {
            results = await doStreamSearch();
        } catch (streamErr) {
            const isAbort = streamErr.name === 'AbortError' || /abort|fetch/i.test(streamErr.message || '');
            if (isAbort || streamErr.message?.includes('NetworkError')) {
                try {
                    results = await doNonStreamSearch();
                } catch (e) {
                    throw streamErr;
                }
            } else {
                throw streamErr;
            }
        }
        finishWithResults(results);
    } catch (error) {
        console.error('AI search error:', error);
        showError('AI 搜索失败: ' + (error.message || 'Unknown error'));
        statusContainer.style.display = 'none';
    }
}

function renderSearchResults(results) {
    timeline.innerHTML = '';
    currentPaperList = [];
    if (results.length === 0) {
        timeline.innerHTML = '<p style="text-align: center; color: var(--text-muted); padding: 40px;">未找到相关论文</p>';
    } else {
        results.forEach(paper => {
            timeline.appendChild(createPaperCard(paper));
        });
        currentPaperList = results.map(p => p.id);
        showEndMarker();
    }
    loadMoreBtn.style.display = 'none';
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
    
    const stage2Badge = (paper.stage2_pending ?? (paper.is_relevant && !(paper.detailed_summary && paper.detailed_summary.trim()))) ? 
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
    
    // Update current paper index for navigation
    currentPaperIndex = currentPaperList.indexOf(paperId);
    
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
            
            ${paper.url ? `
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
            </div>
            ` : `
            <div class="detail-section">
                <h3>PDF</h3>
                <p style="color: var(--text-muted);">本地上传论文，无在线 PDF 链接</p>
            </div>
            `}
            
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
        
        // Update star button state
        const starModalBtn = document.getElementById('starModalBtn');
        if (starModalBtn) {
            if (paper.is_starred) {
                starModalBtn.textContent = '★';
                starModalBtn.classList.add('starred');
            } else {
                starModalBtn.textContent = '☆';
                starModalBtn.classList.remove('starred');
            }
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
        
        // Poll for Stage 2 progress when pending
        if (stage2PollInterval) {
            clearInterval(stage2PollInterval);
            stage2PollInterval = null;
        }
        if (paper.stage2_pending) {
            stage2PollInterval = setInterval(async () => {
                if (currentPaperId !== paperId) return;
                try {
                    const r = await fetch(`${API_BASE}/papers/${paperId}`);
                    const p = await r.json();
                    if (!p.stage2_pending) {
                        clearInterval(stage2PollInterval);
                        stage2PollInterval = null;
                    }
                    updateModalPaperContent(p);
                } catch (e) {
                    console.warn('Stage 2 poll failed:', e);
                }
            }, 4000);
        }
    } catch (error) {
        console.error('Error loading paper:', error);
        showError('Failed to load paper details');
    }
}

function updateModalPaperContent(paper) {
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
                ${(!paper.detailed_summary || !paper.detailed_summary.trim()) ? `
                <button class="btn btn-secondary btn-compact" style="margin-top: 8px;" onclick="requestFullSummary('${escapeHtml(paper.id)}')">
                    📝 生成全文详细摘要
                </button>
                ` : ''}
            </div>
        ` : `
            <div class="detail-section">
                <h3>摘要</h3>
                <p>${escapeHtml(paper.abstract)}</p>
            </div>
        `}
        ${paper.url ? `
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
        </div>
        ` : `
        <div class="detail-section">
            <h3>PDF</h3>
            <p style="color: var(--text-muted);">本地上传论文，无在线 PDF 链接</p>
        </div>
        `}
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
}

// Ask Question (with streaming, reasoning, and follow-up support)
async function requestFullSummary(paperId) {
    try {
        const r = await fetch(`${API_BASE}/papers/${paperId}/request_full_summary`, { method: 'POST', credentials: 'include' });
        const d = await r.json().catch(() => ({}));
        if (r.ok && d.ok) {
            showSuccess('正在生成全文摘要...');
            if (stage2PollInterval) clearInterval(stage2PollInterval);
            stage2PollInterval = setInterval(async () => {
                if (currentPaperId !== paperId) return;
                const pr = await fetch(`${API_BASE}/papers/${paperId}`);
                const p = await pr.json();
                if (!p.stage2_pending) {
                    clearInterval(stage2PollInterval);
                    stage2PollInterval = null;
                    updateModalPaperContent(p);
                }
            }, 2000);
        } else {
            showError(d.error || '请求失败');
        }
    } catch (e) {
        showError('请求失败: ' + (e.message || 'Unknown'));
    }
}

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

function getConfigFormState() {
    const keywords = document.getElementById('filterKeywords')?.value.split(',').map(k => k.trim()).filter(k => k) || [];
    const negKeywords = document.getElementById('negativeKeywords')?.value.split(',').map(k => k.trim()).filter(k => k) || [];
    const questions = document.getElementById('presetQuestions')?.value.split('\n').map(q => q.trim()).filter(q => q) || [];
    const systemPrompt = document.getElementById('systemPrompt')?.value.trim() || '';
    const model = document.getElementById('model')?.value.trim() || '';
    const temperature = document.getElementById('temperature')?.value || '';
    const maxTokens = document.getElementById('maxTokens')?.value || '';
    const fetchInterval = document.getElementById('fetchInterval')?.value || '';
    const maxPapersPerFetch = document.getElementById('maxPapersPerFetch')?.value || '';
    const concurrentPapers = document.getElementById('concurrentPapers')?.value || '';
    const minRelevanceScoreForStage2 = document.getElementById('minRelevanceScoreForStage2')?.value || '';
    const starCategoriesEl = document.getElementById('starCategories');
    const starCategoriesList = starCategoriesEl ? starCategoriesEl.value.split('\n').map(s => s.trim()).filter(s => s) : [];
    return JSON.stringify({
        keywords, negKeywords, questions, systemPrompt, model,
        temperature, maxTokens, fetchInterval, maxPapersPerFetch,
        concurrentPapers, minRelevanceScoreForStage2, starCategoriesList
    });
}

function isConfigDirty() {
    if (!configInitialState) return false;
    return getConfigFormState() !== configInitialState;
}

function resetConfigCloseWarning() {
    configCloseWarningShown = false;
    const banner = document.getElementById('configUnsavedBanner');
    if (banner) banner.style.display = 'none';
}

function handleConfigModalClose() {
    if (!configModal?.classList.contains('active')) return;
    if (configCloseWarningShown) {
        resetConfigCloseWarning();
        closeModal(configModal);
        configInitialState = null;
        return;
    }
    if (isConfigDirty()) {
        configCloseWarningShown = true;
        const banner = document.getElementById('configUnsavedBanner');
        if (banner) banner.style.display = 'flex';
        return;
    }
    resetConfigCloseWarning();
    closeModal(configModal);
    configInitialState = null;
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
        
        // Star categories
        const sc = config.star_categories || ['高效视频生成', 'LLM稀疏注意力', '注意力机制', 'Roll-out方法'];
        document.getElementById('starCategories').value = sc.join('\n');
        
        configInitialState = getConfigFormState();
        configCloseWarningShown = false;
        resetConfigCloseWarning();
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
    
    const starCategoriesInput = document.getElementById('starCategories');
    const starCategoriesList = starCategoriesInput ? starCategoriesInput.value
        .split('\n').map(s => s.trim()).filter(s => s) : [];
    
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
                min_relevance_score_for_stage2: minRelevanceScoreForStage2,
                star_categories: starCategoriesList.length > 0 ? starCategoriesList : ['高效视频生成', 'LLM稀疏注意力', '注意力机制', 'Roll-out方法']
            })
        });
        
        const result = await response.json();
        
        if (starCategoriesList.length > 0) {
            starCategories = starCategoriesList;
            renderCategoryTabs();
        }
        
        configInitialState = getConfigFormState();
        resetConfigCloseWarning();
        closeModal(configModal);
        showSuccess(result.message || 'Configuration saved');
    } catch (error) {
        console.error('Error saving config:', error);
        showError('Failed to save configuration');
    }
}

// Utilities
function closeModal(modal) {
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    if (modal === paperModal) {
        if (stage2PollInterval) {
            clearInterval(stage2PollInterval);
            stage2PollInterval = null;
        }
    }
}

function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

function showError(message) {
    showToast(message, 'error');
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showToast(message, type = 'error') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    container.classList.add('has-toasts');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'error' ? '✕' : '✓';
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;
    const dismiss = () => {
        if (toast.dataset.dismissed) return;
        toast.dataset.dismissed = '1';
        toast.style.animation = 'none';
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            toast.remove();
            if (container.children.length === 0) container.classList.remove('has-toasts');
        }, 200);
    };
    toast.addEventListener('click', dismiss);
    setTimeout(dismiss, 3500);
    container.appendChild(toast);
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
            
            // Only remove card when unstarring on a category tab (starred papers stay in main list)
            if (!isStarred && currentTab !== 'all') {
                card.style.transition = 'opacity 0.3s ease-out';
                card.style.opacity = '0';
                setTimeout(() => {
                    card.remove();
                }, 300);
            }
        }
        
        // Update modal star button if this paper is currently open
        if (currentPaperId === paperId) {
            const starModalBtn = document.getElementById('starModalBtn');
            if (starModalBtn) {
                if (isStarred) {
                    starModalBtn.textContent = '★';
                    starModalBtn.classList.add('starred');
                } else {
                    starModalBtn.textContent = '☆';
                    starModalBtn.classList.remove('starred');
                }
            }
        }
        
        // Also update starred items in the starred section
        updateStarredItemButton(paperId, isStarred);
        
    } catch (error) {
        console.error('Error toggling star:', error);
    }
}

// Toggle star from modal
async function toggleStarFromModal(paperId) {
    await toggleStar(paperId);
}

// Update star button in starred section (if viewing from there)
function updateStarredItemButton(paperId, isStarred) {
    // This function is no longer needed with tab-based approach
    // The card removal is handled in toggleStar
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
        showError('Please enter a score between 0 and 10');
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

// Removed addStarredPapersSection and toggleStarredSection - now using tab-based approach


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
    
    // Sync currentSortBy with sortSelect value before refresh
    if (sortSelect) {
        currentSortBy = sortSelect.value;
    }
    
    // Check if there's a search query
    const searchQuery = searchInput.value.trim();
    if (searchQuery) {
        searchPapers(searchQuery);
    } else {
        loadPapers();
    }
}

// Clear search state for current tab only
function clearSearchState() {
    try {
        sessionStorage.removeItem(_searchStateKey(currentTab));
    } catch (_) {}
}

// Restore search state on page load for current tab. Returns true if results were restored.
function restoreSearchState() {
    const cached = restoreSearchResults(currentTab);
    if (cached && timeline) {
        if (cached.query && searchInput) {
            searchInput.value = cached.query;
        }
        renderSearchResults(cached.results);
        return true;
    }
    return false;
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

// Toggle PDF viewer - open fullscreen preview
function togglePdfViewer(paperId) {
    const fullscreenViewer = document.getElementById('fullscreenPdfViewer');
    const fullscreenFrame = document.getElementById('fullscreenPdfFrame');
    const pdfViewerLink = document.getElementById('pdfViewerLink');
    
    // Get paper URL and convert to PDF
    fetch(`${API_BASE}/papers/${paperId}`)
        .then(res => res.json())
        .then(paper => {
            const pdfUrl = getPdfUrl(paper.url);
            pdfViewerLink.href = pdfUrl;
            fullscreenFrame.src = pdfUrl;
            fullscreenViewer.style.display = 'flex';
            document.body.classList.add('pdf-preview-open');
        })
        .catch(err => {
            console.error('Error loading PDF:', err);
            showError('无法加载 PDF');
        });
}

// Close fullscreen PDF viewer
function closeFullscreenPdf() {
    const fullscreenViewer = document.getElementById('fullscreenPdfViewer');
    const fullscreenFrame = document.getElementById('fullscreenPdfFrame');
    
    fullscreenViewer.style.display = 'none';
    fullscreenFrame.src = ''; // Clear iframe to stop loading
    document.body.classList.remove('pdf-preview-open');
}

// Export paper to markdown
async function exportPaperToMarkdown(paperId) {
    if (!paperId) return;
    
    try {
        const response = await fetch(`${API_BASE}/papers/${paperId}`);
        const paper = await response.json();
        
        // Build markdown content
        let markdown = `# ${paper.title}\n\n`;
        
        // Authors
        if (paper.authors && paper.authors.length > 0) {
            markdown += `**Authors:** ${paper.authors.join(', ')}\n\n`;
        }
        
        // Published date
        if (paper.published_date) {
            try {
                const date = new Date(paper.published_date);
                markdown += `**Published:** ${date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}\n\n`;
            } catch (e) {
                // Skip invalid dates
            }
        }
        
        // URL
        if (paper.url) {
            markdown += `**URL:** ${paper.url}\n\n`;
        }
        
        // Relevance score
        if (paper.relevance_score !== null && paper.relevance_score !== undefined) {
            markdown += `**Relevance Score:** ${paper.relevance_score}/10\n\n`;
        }
        
        // Keywords
        if (paper.extracted_keywords && paper.extracted_keywords.length > 0) {
            markdown += `**Keywords:** ${paper.extracted_keywords.join(', ')}\n\n`;
        }
        
        markdown += `---\n\n`;
        
        // Abstract
        if (paper.abstract) {
            markdown += `## Abstract\n\n${paper.abstract}\n\n`;
        }
        
        // Detailed summary
        if (paper.detailed_summary) {
            markdown += `## AI Detailed Summary\n\n${paper.detailed_summary}\n\n`;
        } else if (paper.one_line_summary) {
            markdown += `## AI Summary\n\n${paper.one_line_summary}\n\n`;
        }
        
        // Q&A pairs
        if (paper.qa_pairs && paper.qa_pairs.length > 0) {
            markdown += `## Questions & Answers\n\n`;
            paper.qa_pairs.forEach((qa, index) => {
                markdown += `### Q${index + 1}: ${qa.question}\n\n`;
                if (qa.thinking) {
                    markdown += `**Thinking Process:**\n\n${qa.thinking}\n\n`;
                }
                markdown += `**Answer:**\n\n${qa.answer}\n\n`;
                if (qa.parent_qa_id !== null && qa.parent_qa_id !== undefined) {
                    markdown += `*This is a follow-up question*\n\n`;
                }
                markdown += `---\n\n`;
            });
        }
        
        // Create download link
        const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Sanitize filename
        const safeTitle = paper.title.replace(/[^a-z0-9]/gi, '_').substring(0, 50);
        a.download = `${safeTitle}_${paperId}.md`;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error exporting paper:', error);
        showError('导出失败');
    }
}

// Navigate to previous/next paper
function navigateToPaper(direction) {
    if (currentPaperList.length === 0 || currentPaperIndex === -1) {
        return;
    }
    
    const newIndex = currentPaperIndex + direction;
    
    if (newIndex < 0 || newIndex >= currentPaperList.length) {
        return;  // Already at first/last paper
    }
    
    const newPaperId = currentPaperList[newIndex];
    if (newPaperId) {
        currentPaperIndex = newIndex;
        openPaperModal(newPaperId);
    }
}

