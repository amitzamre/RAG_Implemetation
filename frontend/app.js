document.addEventListener("DOMContentLoaded", () => {
    // Elements Selector
    const chatForm = document.getElementById("chat-form");
    const userQueryInput = document.getElementById("user-query");
    const chatMessages = document.getElementById("chat-messages");
    const llmModeSelect = document.getElementById("llm-mode-select");
    const systemStatus = document.getElementById("system-status");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const sendBtn = document.getElementById("send-btn");
    
    // RAG Inspector Elements
    const chunksList = document.getElementById("chunks-list");
    const promptContent = document.getElementById("prompt-content");
    const timeRetrieval = document.getElementById("time-retrieval");
    const timeGeneration = document.getElementById("time-generation");
    const timeTotal = document.getElementById("time-total");

    // Initialize State
    let config = { llm_mode: "MOCK", available_modes: [] };

    // Fetch initial configuration on load
    fetchConfig();

    // ==========================================
    // CONFIGURATION MANAGEMENT
    // ==========================================
    async function fetchConfig() {
        try {
            const res = await fetch("/api/config");
            if (!res.ok) throw new Error("Failed to load backend config");
            config = await res.json();
            
            // Set select value
            llmModeSelect.value = config.llm_mode;
            updateStatusText(config.llm_mode);
        } catch (err) {
            console.error("Config fetch error:", err);
            systemStatus.textContent = "Server Connection Failed";
            systemStatus.parentElement.querySelector(".pulse-dot").style.backgroundColor = "#ff3333";
            systemStatus.parentElement.querySelector(".pulse-dot").style.boxShadow = "0 0 10px #ff3333";
        }
    }

    async function updateConfig(newMode) {
        try {
            setLoadingState(true);
            systemStatus.textContent = "Switching Engine Mode...";
            
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ llm_mode: newMode })
            });
            
            if (!res.ok) throw new Error("Failed to update config");
            const updatedConfig = await res.json();
            config.llm_mode = updatedConfig.llm_mode;
            
            updateStatusText(config.llm_mode);
            setLoadingState(false);
            
            // System message confirming change
            appendSystemMessage(`System: Dynamically switched to <strong>${config.llm_mode}</strong> LLM mode.`);
        } catch (err) {
            console.error("Config update error:", err);
            appendSystemMessage(`<span style="color: #ff3333">Error: Failed to change LLM mode.</span>`);
            llmModeSelect.value = config.llm_mode; // Revert select
            setLoadingState(false);
        }
    }

    function updateStatusText(mode) {
        if (mode === "MOCK") {
            systemStatus.textContent = "RAG Ready: MOCK Mode";
        } else if (mode === "LOCAL") {
            systemStatus.textContent = "RAG Ready: LOCAL QWEN";
        } else if (mode === "CLOUD_GEMINI") {
            systemStatus.textContent = "RAG Ready: CLOUD GEMINI";
        }
        systemStatus.parentElement.querySelector(".pulse-dot").style.backgroundColor = "#00ff66";
        systemStatus.parentElement.querySelector(".pulse-dot").style.boxShadow = "0 0 10px #00ff66";
    }

    function setLoadingState(loading) {
        llmModeSelect.disabled = loading;
        sendBtn.disabled = loading;
        if (loading) {
            sendBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
        } else {
            sendBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i>`;
        }
    }

    // ==========================================
    // CHAT & SUBMISSION
    // ==========================================
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = userQueryInput.value.trim();
        if (!query) return;
        
        submitQuery(query);
    });

    // Handle suggestion chips
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("query-chip")) {
            const query = e.target.textContent;
            submitQuery(query);
        }
    });

    async function submitQuery(query) {
        // 1. Clear input
        userQueryInput.value = "";
        
        // 2. Append User Message
        appendMessage("user", query);
        
        // 3. Show typing indicator
        const typingBubble = showTypingIndicator();
        setLoadingState(true);
        
        try {
            // 4. Send API Request
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });
            
            // Remove typing indicator
            typingBubble.remove();
            
            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Server error");
            }
            
            const data = await res.json();
            
            // 5. Append Bot Answer
            appendMessage("bot", data.answer);
            
            // 6. Update RAG Inspector Panel
            updateRAGInspector(data);
            
        } catch (err) {
            typingBubble.remove();
            console.error("API error:", err);
            appendMessage("bot", `⚠️ <strong>Error connecting to RAG engine:</strong> ${err.message || 'Please make sure python virtual environment packages are fully installed and server is running.'}`);
        } finally {
            setLoadingState(false);
        }
    }

    // Append standard messages
    function appendMessage(sender, text) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", `${sender}-message`);
        
        const contentDiv = document.createElement("div");
        contentDiv.classList.add("message-content");
        contentDiv.innerHTML = formatMarkdown(text);
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll container to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendSystemMessage(htmlContent) {
        const systemDiv = document.createElement("div");
        systemDiv.className = "message system-message";
        systemDiv.style.animation = "slideUpFade 0.3s ease-out";
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.style.padding = "8px 15px";
        contentDiv.style.background = "rgba(0, 191, 255, 0.05)";
        contentDiv.style.border = "1px solid rgba(0, 191, 255, 0.15)";
        contentDiv.style.borderRadius = "10px";
        contentDiv.style.fontSize = "0.85rem";
        contentDiv.style.textAlign = "center";
        contentDiv.style.color = "var(--text-muted)";
        contentDiv.innerHTML = htmlContent;
        
        systemDiv.appendChild(contentDiv);
        chatMessages.appendChild(systemDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const bubble = document.createElement("div");
        bubble.className = "message bot-message";
        
        const content = document.createElement("div");
        content.className = "message-content typing-bubble";
        content.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        `;
        
        bubble.appendChild(content);
        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return bubble;
    }

    // Quick custom parser for subset of markdown tags (###, *, -, ol, bold)
    function formatMarkdown(text) {
        let html = text;
        
        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Unordered lists
        html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
        html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
        
        // Wrap <li> elements in <ul> if needed. Easy heuristic: wrap consecutive lists
        // Note: For education purposes, a simple wrapper is sufficient.
        html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
        
        // Replace double newlines with paragraphs
        // Only if they are not in lists/headers
        html = html.split('\n\n').map(p => {
            if (p.trim().startsWith('<h') || p.trim().startsWith('<ul') || p.trim().startsWith('<ol')) {
                return p;
            }
            return `<p>${p.replace(/\n/g, '<br>')}</p>`;
        }).join('');
        
        return html;
    }

    // ==========================================
    // RAG INSPECTOR UPDATES
    // ==========================================
    function updateRAGInspector(data) {
        // 1. Update Tab 1: Retrieval Chunks
        chunksList.innerHTML = "";
        
        if (data.retrieved_chunks.length === 0) {
            chunksList.innerHTML = `
                <div class="empty-inspector-state">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <p>No document chunks passed the similarity threshold of 0.1.</p>
                </div>`;
        } else {
            data.retrieved_chunks.forEach((chunk, index) => {
                const scorePercent = Math.round(chunk.similarity_score * 100);
                
                const card = document.createElement("div");
                card.className = "chunk-card";
                card.style.animationDelay = `${index * 100}ms`;
                
                card.innerHTML = `
                    <div class="chunk-card-header">
                        <span class="chunk-title"><i class="fa-regular fa-file-lines"></i> SOURCE ${index + 1}: ${chunk.title}</span>
                        <div class="chunk-score-group">
                            <span class="chunk-score-label">Cosine Similarity:</span>
                            <span class="chunk-score-badge">${chunk.similarity_score.toFixed(3)}</span>
                        </div>
                    </div>
                    <div class="chunk-similarity-progress">
                        <div class="chunk-similarity-fill" id="fill-chunk-${index}"></div>
                    </div>
                    <div class="chunk-card-body">
                        <div class="chunk-text">${chunk.text.replace('SECTION: ' + chunk.title, '').trim()}</div>
                    </div>
                `;
                
                chunksList.appendChild(card);
                
                // Animate progress bar fill after appending
                setTimeout(() => {
                    const fill = document.getElementById(`fill-chunk-${index}`);
                    if (fill) fill.style.width = `${scorePercent}%`;
                }, 50);
            });
        }

        // 2. Update Tab 2: Prompt Visualizer
        // Color code components of the prompt template for educational impact
        const promptRaw = data.raw_prompt;
        
        // Highlight parts of the prompt
        let highlightedPrompt = promptRaw
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
            
        // Wrap sections in span tags for css styling
        highlightedPrompt = highlightedPrompt
            .replace(/(You are an AI assistant specialized in IRCTC.*DETAILED ANSWER:)/s, '<span class="prompt-system">$1</span>')
            .replace(/(=== RETRIEVED OFFICIAL POLICY CONTEXTS ===.*?==========================================)/s, '<span class="prompt-context">$1</span>')
            .replace(/(USER QUESTION:.*?)$/s, '<span class="prompt-query">$1</span>');
            
        promptContent.innerHTML = highlightedPrompt;

        // 3. Update Tab 3: Performance Latency
        timeRetrieval.textContent = `${data.timings.retrieval_ms}ms`;
        timeGeneration.textContent = `${data.timings.generation_ms}ms`;
        timeTotal.textContent = `${data.timings.total_ms}ms`;
    }

    // ==========================================
    // INTERACTIVE CONTROLS
    // ==========================================
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            // Remove active classes
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            
            // Add active class
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // LLM mode changes
    llmModeSelect.addEventListener("change", (e) => {
        const selectedMode = e.target.value;
        updateConfig(selectedMode);
    });
});
