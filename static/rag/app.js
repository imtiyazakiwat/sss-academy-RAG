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
  const retrievalValue = document.getElementById("retrievalValue");
  const ttftValue = document.getElementById("ttftValue");
  const answerBody = document.getElementById("answerBody");
  const sourcesList = document.getElementById("sourcesList");
  const copyBtn = document.getElementById("copyBtn");
  const copyText = document.getElementById("copyText");
  const indexedCount = document.getElementById("indexedCount");

  const historyList = document.getElementById("historyList");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  let history = JSON.parse(localStorage.getItem("RAG_HISTORY") || "[]");
  let isFirstChunk = true;
  let streamingEnabled = false;
  let streamAccum = "";

  renderHistory();
  fetchStats();
  fetchSampleQuestions();

  // Form submit
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    executeQuery(q);
  });

  // Cmd/Ctrl+Enter
  input.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      form.dispatchEvent(new Event("submit"));
    }
  });

  // Copy answer
  copyBtn.addEventListener("click", () => {
    const text = answerBody.innerText;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      copyText.innerText = "Copied!";
      setTimeout(() => (copyText.innerText = "Copy"), 2000);
    });
  });

  // Clear history
  clearHistoryBtn.addEventListener("click", () => {
    history = [];
    localStorage.removeItem("RAG_HISTORY");
    renderHistory();
  });

  function pushHistory(q, mode) {
    history = history.filter((h) => h.q !== q);
    history.unshift({ q, mode, ts: Date.now() });
    history = history.slice(0, 20);
    localStorage.setItem("RAG_HISTORY", JSON.stringify(history));
    renderHistory();
  }

  function renderHistory() {
    historyList.innerHTML = "";
    if (history.length === 0) {
      historyList.innerHTML = `<div class="history-empty">No questions yet. Ask something below.</div>`;
      return;
    }
    history.forEach((h) => {
      const item = document.createElement("div");
      item.className = "history-item";
      const time = new Date(h.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      item.innerHTML = `
        <div class="history-q">${escapeHtml(h.q)}</div>
        <div class="history-meta"><span>[${escapeHtml(h.mode || "generated")}]</span><span>${time}</span></div>
      `;
      item.addEventListener("click", () => {
        input.value = h.q;
        executeQuery(h.q);
      });
      historyList.appendChild(item);
    });
  }

  async function executeQuery(questionText) {
    submitBtn.disabled = true;
    submitBtn.querySelector(".btn-text").innerText = "Working…";
    emptyState.classList.add("hidden");
    resultsSection.classList.remove("hidden");
    isFirstChunk = true;
    streamingEnabled = true;
    streamAccum = "";

    answerBody.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px; color: #94a3b8;">
        <span class="status-dot" style="animation: pulse 1.2s infinite;"></span>
        <span>Retrieving & generating grounded answer…</span>
      </div>
    `;
    sourcesList.innerHTML = "";
    modeBadge.className = "badge badge-generated";
    modeBadge.innerText = "…";
    similarityValue.innerText = "—";
    similarityBar.style.width = "0%";
    retrievalValue.innerText = "—";
    ttftValue.innerText = "—";

    const startTime = performance.now();
    try {
      const res = await fetch("/api/ask-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: questionText }),
      });
      if (!res.ok) throw new Error(`Server returned status ${res.status}`);

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
            handleSSE(data, { startTime, questionText });
            if (data.type === "token") {
              streamedText += data.text;
            }
          } catch (e) {
            console.error("Parse error", e);
          }
        }
      }
    } catch (err) {
      console.error(err);
      answerBody.innerHTML = `<span style="color: #f43f5e;">⚠️ ${escapeHtml(err.message)}</span>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector(".btn-text").innerText = "Answer";
    }
  }

  function handleSSE(data, ctx) {
    if (data.type === "meta") {
      renderMetrics({ mode: data.mode, confidence: data.confidence });
      ttftValue.innerText = `${data.ttft_ms} ms`;
      renderSources(data.evidence || []);
      if (isStreamingMode(data.mode)) {
        answerBody.innerHTML = "";
        streamingEnabled = true;
      } else {
        streamingEnabled = false;
      }
    }
    if (data.type === "token") {
      if (isFirstChunk) {
        isFirstChunk = false;
        ttftValue.innerText = `${(performance.now() - ctx.startTime).toFixed(0)} ms (TTFT)`;
      }
      streamAccum += data.text;
      answerBody.innerHTML = formatMarkdown(streamAccum);
    }
    if (data.type === "complete") {
      renderMetrics({ mode: data.mode, confidence: data.confidence });
      retrievalValue.innerText = `${data.retrieval_ms} ms`;
      ttftValue.innerText = `${data.ttft_ms} ms`;
      answerBody.innerHTML = formatMarkdown(data.answer);
      renderSources(data.evidence || []);
      pushHistory(ctx.questionText, data.mode);
    }
    if (data.type === "done") {
      retrievalValue.innerText = `${data.total_ms} ms`;
      ttftValue.innerText = `${data.ttft_ms ?? ""} ms`;
      answerBody.innerHTML = formatMarkdown(data.answer);
      pushHistory(ctx.questionText, "generated");
    }
  }

  function isStreamingMode(mode) {
    return mode === "generated";
  }

  function renderMetrics({ mode, confidence }) {
    modeBadge.className = "badge";
    if (mode === "unsupported") {
      modeBadge.classList.add("badge-exact");
      modeBadge.innerText = "⚠️ Not in KB";
    } else if (mode === "extracted") {
      modeBadge.classList.add("badge-extracted");
      modeBadge.innerText = "📄 Extracted";
    } else {
      modeBadge.classList.add("badge-generated");
      modeBadge.innerText = "🧠 Grounded RAG";
    }
    const conf = Math.min(Math.max(confidence || 0, 0), 1);
    similarityValue.innerText = conf.toFixed(2);
    similarityBar.style.width = `${(conf * 100).toFixed(0)}%`;
  }

  function renderSources(evidence) {
    sourcesList.innerHTML = "";
    if (!evidence || evidence.length === 0) {
      sourcesList.innerHTML = `<div style="color:#64748b;font-size:13px;">No relevant source found.</div>`;
      return;
    }
    evidence.forEach((e) => {
      const card = document.createElement("div");
      card.className = "source-card";
      const snippet = escapeHtml((e.content || "").slice(0, 220));
      card.innerHTML = `
        <div class="source-top">
          <span class="source-topic">${escapeHtml(e.topic || "General")}</span>
          <span class="source-chip">Page ${e.page ?? "?"} • conf ${e.score?.toFixed ? e.score.toFixed(2) : e.score}</span>
        </div>
        <div class="source-snippet">${snippet}${snippet.length >= 220 ? "…" : ""}</div>
      `;
      card.title = "Click to copy this source text";
      card.addEventListener("click", async () => {
        await navigator.clipboard.writeText(e.content || "");
        card.classList.add("copied");
        setTimeout(() => card.classList.remove("copied"), 1200);
      });
      sourcesList.appendChild(card);
    });
  }

  async function fetchStats() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      indexedCount.innerText = data.chunks ?? "—";
    } catch (e) {
      indexedCount.innerText = "—";
    }
  }

  async function fetchSampleQuestions() {
    try {
      const res = await fetch("/api/sample-questions");
      renderChips(await res.json());
    } catch (e) {
      /* ignore */
    }
  }

  function renderChips(samples) {
    chipsContainer.innerHTML = "";
    samples.forEach((item) => {
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

  // --- Markdown + escaping (kept from prior version) ---
  function formatMarkdown(text) {
    if (!text) return "";
    if (window.marked && typeof window.marked.parse === "function") {
      try {
        return window.marked.parse(text, { breaks: true, gfm: true });
      } catch (e) { /* fall through */ }
    }
    let html = escapeHtml(text);
    html = html.replace(/```([a-z]*)\n([\s\S]*?)```/g, (m, lang, code) => `<pre><code>${code.trim()}</code></pre>`);
    html = html.replace(/^###\s+(.*)$/gim, "<h4>$1</h4>");
    html = html.replace(/^##\s+(.*)$/gim, "<h3>$1</h3>");
    html = html.replace(/^#\s+(.*)$/gim, "<h3>$1</h3>");
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    html = html.replace(/^[\s]*[•\-*]\s+(.*)$/gim, "<li>$1</li>");
    html = html.replace(/^[\s]*(\d+)\.\s+(.*)$/gim, '<li value="$1">$2</li>');
    html = html.replace(/(<li value="\d+">.*?<\/li>(\n| )?)+/gis, (m) => `<ol>${m}</ol>`);
    html = html.replace(/(<li>.*?<\/li>(\n| )?)+/gis, (m) => `<ul>${m}</ul>`);
    const blocks = html.split(/\n{2,}/);
    html = blocks.map((block) => {
      const t = block.trim();
      if (!t) return "";
      if (t.startsWith("<ol") || t.startsWith("<ul") || t.startsWith("<pre") || t.startsWith("<h3") || t.startsWith("<h4")) return t;
      return `<p>${t.replace(/\n/g, "<br>")}</p>`;
    }).filter(Boolean).join("");
    return html;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
});
