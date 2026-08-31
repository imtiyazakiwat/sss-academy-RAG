document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("questionForm");
  const input = document.getElementById("questionInput");
  const submitBtn = document.getElementById("submitBtn");
  const chipsContainer = document.getElementById("sampleChips");

  const emptyState = document.getElementById("emptyState");
  const resultsSection = document.getElementById("resultsSection");
  
  const modeBadge = document.getElementById("modeBadge");
  const similarityValue = document.getElementById("similarityValue");
  const similarityBar = document.getElementById("similarityBar");
  const latencyValue = document.getElementById("latencyValue");
  const topicTag = document.getElementById("topicTag");
  const answerBody = document.getElementById("answerBody");
  const topMatchesList = document.getElementById("topMatchesList");
  const copyBtn = document.getElementById("copyBtn");
  const copyText = document.getElementById("copyText");
  const indexedCount = document.getElementById("indexedCount");

  // Engine & Key elements
  const keyModal = document.getElementById("keyModal");
  const engineToggleBtn = document.getElementById("engineToggleBtn");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const saveKeyBtn = document.getElementById("saveKeyBtn");
  const engineLabel = document.getElementById("engineLabel");
  const engineIcon = document.getElementById("engineIcon");
  const radioMLX = document.getElementById("radioMLX");
  const radioGroq = document.getElementById("radioGroq");

  // Engine State (Default: MLX Apple Silicon)
  let currentEngine = localStorage.getItem("APP_ENGINE") || "mlx";
  const DEFAULT_GROQ_KEY = "";
  let savedApiKey = localStorage.getItem("GROQ_API_KEY") || DEFAULT_GROQ_KEY;

  updateEngineUI();

  // Load backend stats & sample chips
  fetchStats();
  fetchSampleQuestions();

  // Modal Listeners
  if (engineToggleBtn) {
    engineToggleBtn.addEventListener("click", () => {
      if (currentEngine === "mlx") {
        radioMLX.checked = true;
      } else {
        radioGroq.checked = true;
      }
      apiKeyInput.value = savedApiKey;
      keyModal.classList.remove("hidden");
    });
  }

  closeModalBtn.addEventListener("click", () => {
    keyModal.classList.add("hidden");
  });

  saveKeyBtn.addEventListener("click", () => {
    if (radioMLX.checked) {
      currentEngine = "mlx";
    } else {
      currentEngine = "groq";
    }
    localStorage.setItem("APP_ENGINE", currentEngine);

    const enteredKey = apiKeyInput.value.trim();
    if (enteredKey) {
      savedApiKey = enteredKey;
      localStorage.setItem("GROQ_API_KEY", savedApiKey);
    }
    updateEngineUI();
    keyModal.classList.add("hidden");
  });

  function updateEngineUI() {
    if (currentEngine === "mlx") {
      if (engineLabel) engineLabel.innerText = "Qwen 2.5 (LoRA Fine-Tuned)";
      if (engineIcon) engineIcon.innerText = "🍎";
    } else {
      if (engineLabel) engineLabel.innerText = "Groq LPU (Qwen 27B)";
      if (engineIcon) engineIcon.innerText = "⚡";
    }
  }

  // Handle Form Submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    await executeQuery(query);
  });

  // Handle Ctrl+Enter / Cmd+Enter in textarea
  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      form.dispatchEvent(new Event("submit"));
    }
  });

  // Handle Copy Button
  copyBtn.addEventListener("click", () => {
    const text = answerBody.innerText;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      copyText.innerText = "Copied!";
      setTimeout(() => {
        copyText.innerText = "Copy";
      }, 2000);
    });
  });

  async function executeQuery(questionText) {
    // UI Loading State
    submitBtn.disabled = true;
    submitBtn.querySelector(".btn-text").innerText = "Searching...";
    emptyState.classList.add("hidden");
    resultsSection.classList.remove("hidden");
    
    answerBody.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: #94a3b8;">
        <span class="status-dot" style="animation: pulse 1.2s infinite;"></span>
        <span>Retrieving from notebook index & generating via ${currentEngine === 'mlx' ? 'Qwen 2.5 LoRA (Apple Silicon MLX)' : 'Groq LPU'}...</span>
      </div>
    `;

    const startTime = performance.now();

    try {
      const res = await fetch("/api/ask-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: questionText,
          engine: currentEngine,
          api_key: savedApiKey || undefined
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let streamedText = "";
      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim().startsWith("data: ")) continue;
          const jsonStr = block.replace(/^data:\s*/, "").trim();
          try {
            const data = JSON.parse(jsonStr);

            if (data.type === "exact") {
              renderResults({
                mode: data.mode,
                engine: data.engine,
                similarity: data.similarity,
                latency_ms: data.latency_ms,
                topic: data.topic,
                answer: data.text,
                top_matches: data.top_matches
              });
              return;
            }

            if (data.type === "meta") {
              // Set initial headers
              modeBadge.className = "badge badge-generated";
              if (data.engine === "mlx") {
                modeBadge.innerText = "🍎 Apple M4 (Fine-Tuned LoRA)";
              } else {
                modeBadge.innerText = "⚡ Groq LPU (Qwen 3.8-27B)";
              }
              
              const sim = Math.min(Math.max(data.similarity, 0), 1);
              similarityValue.innerText = sim.toFixed(2);
              similarityBar.style.width = `${(sim * 100).toFixed(0)}%`;
              topicTag.innerText = data.topic || "ETL Testing";
              renderTopMatches(data.top_matches || []);
            }

            if (data.type === "token") {
              if (isFirstChunk) {
                answerBody.innerHTML = "";
                isFirstChunk = false;
                const ttft = (performance.now() - startTime).toFixed(0);
                latencyValue.innerText = `${ttft} ms (TTFT)`;
              }
              streamedText += data.text;
              answerBody.innerHTML = formatMarkdown(streamedText);
            }

            if (data.type === "done") {
              const totalElapsed = (performance.now() - startTime).toFixed(0);
              latencyValue.innerText = `${totalElapsed} ms`;
              answerBody.innerHTML = formatMarkdown(streamedText);
            }
          } catch (e) {
            console.error("Error parsing stream chunk:", e);
          }
        }
      }

      if (streamedText) {
        answerBody.innerHTML = formatMarkdown(streamedText);
      }
    } catch (err) {
      console.error(err);
      answerBody.innerHTML = `<span style="color: #f43f5e;">⚠️ Error executing query: ${escapeHtml(err.message)}</span>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector(".btn-text").innerText = "Search / Answer";
    }
  }

  function renderResults(data) {
    // Update Mode Badge
    modeBadge.className = "badge";
    if (data.mode === "exact_match") {
      modeBadge.classList.add("badge-exact");
      modeBadge.innerText = "⚡ Exact Match (Notebook)";
    } else if (data.engine === "mlx") {
      modeBadge.classList.add("badge-generated");
      modeBadge.innerText = "🍎 Qwen 2.5 (LoRA Fine-Tuned)";
    } else {
      modeBadge.classList.add("badge-generated");
      modeBadge.innerText = "⚡ Groq LPU (Qwen 27B)";
    }

    // Update Similarity
    const sim = Math.min(Math.max(data.similarity, 0), 1);
    similarityValue.innerText = sim.toFixed(2);
    similarityBar.style.width = `${(sim * 100).toFixed(0)}%`;

    // Update Latency
    latencyValue.innerText = `${data.latency_ms} ms`;

    // Update Topic
    topicTag.innerText = data.topic || "ETL Testing";

    // Format & Render Answer
    answerBody.innerHTML = formatMarkdown(data.answer);

    // Render Matched Context Drawer
    renderTopMatches(data.top_matches || []);
  }

  function renderTopMatches(matches) {
    topMatchesList.innerHTML = "";
    if (!matches || matches.length === 0) {
      topMatchesList.innerHTML = `<div style="color: #64748b; font-size: 13px;">No direct notebook context retrieved.</div>`;
      return;
    }

    matches.slice(0, 3).forEach((item, index) => {
      const card = document.createElement("div");
      card.className = "context-card";

      const score = Math.min(Math.max(item.similarity || 0, 0), 1);
      const percent = (score * 100).toFixed(0);

      card.innerHTML = `
        <div class="context-card-header">
          <span class="context-title">#${index + 1} ${escapeHtml(item.question)}</span>
          <span class="context-score">${percent}% Match</span>
        </div>
        <div class="context-snippet">${escapeHtml(item.answer.slice(0, 180))}...</div>
      `;

      card.addEventListener("click", () => {
        input.value = item.question;
        executeQuery(item.question);
      });

      topMatchesList.appendChild(card);
    });
  }

  async function fetchStats() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      indexedCount.innerText = data.indexed_count || "47";
      if (data.mlx_active) {
        updateEngineUI();
      }
    } catch (e) {
      console.warn("Could not fetch stats:", e);
    }
  }

  async function fetchSampleQuestions() {
    try {
      const res = await fetch("/api/sample-questions");
      const samples = await res.json();
      renderChips(samples);
    } catch (e) {
      console.warn("Could not fetch sample questions:", e);
    }
  }

  function renderChips(samples) {
    chipsContainer.innerHTML = "";
    samples.forEach(item => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "sample-chip";
      chip.innerText = item.text;
      chip.addEventListener("click", () => {
        input.value = item.text;
        executeQuery(item.text);
      });
      chipsContainer.appendChild(chip);
    });
  }

  function formatMarkdown(text) {
    if (!text) return "";

    // 1. If marked.js is available from CDN, use it
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        return window.marked.parse(text, { breaks: true, gfm: true });
      } catch (e) {
        console.warn("marked.parse error, using fallback:", e);
      }
    }

    // 2. Comprehensive built-in fallback parser
    let html = escapeHtml(text);

    // Code blocks ``` ... ```
    html = html.replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Headers: ###, ##, #
    html = html.replace(/^###\s+(.*$)/gim, '<h4>$1</h4>');
    html = html.replace(/^##\s+(.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^#\s+(.*$)/gim, '<h3>$1</h3>');

    // Bold: **text** or __text__ -> <strong>text</strong>
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_ -> <em>text</em>
    html = html.replace(/\*([^\*\n]+)\*/g, '<em>$1</em>');
    html = html.replace(/_([^_\n]+)_/g, '<em>$1</em>');

    // Inline code: `code` -> <code>code</code>
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Unordered lists: • item, - item, * item -> <li>item</li>
    html = html.replace(/^[\s]*[•\-\*]\s+(.*$)/gim, '<li>$1</li>');

    // Ordered lists: 1. item -> <li>item</li>
    html = html.replace(/^[\s]*(\d+)\.\s+(.*$)/gim, '<li value="$1">$2</li>');

    // Group lists into <ol> or <ul>
    html = html.replace(/(<li value="\d+">.*?<\/li>(\n| )*)+/gis, (match) => {
      return `<ol>${match}</ol>`;
    });
    html = html.replace(/(<li>.*?<\/li>(\n| )*)+/gis, (match) => {
      return `<ul>${match}</ul>`;
    });

    // Paragraphs
    const blocks = html.split(/\n{2,}/);
    html = blocks.map(block => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (trimmed.startsWith('<ol') || trimmed.startsWith('<ul') || trimmed.startsWith('<pre') || trimmed.startsWith('<h3') || trimmed.startsWith('<h4')) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).filter(Boolean).join('');

    return html;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
