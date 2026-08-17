import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

// ============================================================
// HELPERS
// ============================================================
function formatTime(seconds) {
  seconds = parseFloat(seconds);
  if (isNaN(seconds)) return "—";
  if (seconds < 60) return seconds.toFixed(2) + "s";
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return mins + "m " + secs + "s";
}

function scoreClass(val, good, warn) {
  return val >= good ? "good" : val >= warn ? "warn" : "bad";
}

function generateId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return "id-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

const TRACE_ICONS = {
  extract_start: "📄", workflow_start: "⚙️",
  agent_start: "🤖", agent_thought: "💭",
  agent_search: "🔍", agent_action: "⚡", agent_done: "✅",
};

// ============================================================
// METRICS SUBCOMPONENT (unchanged)
// ============================================================
function MetricsBody({ m }) {
  const core = [
    { label: "Type", value: m.type === "summary" ? "Summary" : "Q&A" },
    { label: "Total Time", value: formatTime(m.response_time_sec) },
    { label: "Extraction", value: formatTime(m.extraction_time_sec) },
  ];
  if (m.type === "summary") {
    core.push({ label: "Summary Time", value: formatTime(m.summary_time_sec) });
    core.push({ label: "Summary Words", value: (m.summary_length_words || 0).toLocaleString() });
    core.push({ label: "LLM Calls", value: m.llm_calls || "—" });
  } else {
    core.push({ label: "QA Time", value: formatTime(m.qa_time_sec) });
    core.push({ label: "Model", value: m.model_used || "—" });
    core.push({ label: "LLM Calls", value: m.llm_calls ?? "—" });
  }
  core.push({ label: "Pages", value: m.pages_processed || "—" });
  core.push({ label: "Words", value: (m.words_processed || 0).toLocaleString() });

  const ttft = parseFloat(m.ttft_sec || 0);
  const e2e = parseFloat(m.e2e_latency_sec || 0);
  const tps = parseFloat(m.tps || 0);
  const latency = [
    { label: "TTFT", value: formatTime(ttft), cls: ttft < e2e ? "good" : "warn" },
    { label: "E2E Latency", value: formatTime(e2e), cls: "warn" },
    { label: "TPS", value: tps.toFixed(1), cls: tps >= 5 ? "good" : tps >= 2 ? "warn" : "bad" },
  ];

  let quality = null;
  let decision = null;
  if (m.type !== "summary") {
    const retrieval = parseFloat(m.retrieval_score || 0);
    const conf = parseFloat(m.confidence_score || 0);
    const recall = parseFloat(m.recall_at_k || 0);
    quality = [
      { label: "Retrieval", value: retrieval.toFixed(1) + "%", cls: scoreClass(retrieval, 40, 25) },
      { label: "Confidence", value: conf.toFixed(1) + "%", cls: scoreClass(conf, 60, 30) },
      { label: "Recall@K", value: recall.toFixed(1) + "%", cls: scoreClass(recall, 70, 40) },
    ];
    const dt = m.decision_type || "accepted";
    decision = {
      label: dt === "accepted" ? "✅ Accepted" : dt === "hard_reject" ? "🚫 Hard Reject" : "❌ Rejected",
      cls: dt === "accepted" ? "decision-accepted" : "decision-rejected",
    };
  }

  return (
    <div className="metrics-grid">
      <div className="metrics-section-title">📊 Core Performance</div>
      {core.map((item, i) => (
        <div className="metric-card" key={"core" + i}>
          <div className="metric-label">{item.label}</div>
          <div className="metric-value">{item.value}</div>
        </div>
      ))}

      <div className="metrics-section-title">⚡ Latency</div>
      {latency.map((item, i) => (
        <div className="metric-card" key={"lat" + i}>
          <div className="metric-label">{item.label}</div>
          <div className={"metric-value " + item.cls}>{item.value}</div>
        </div>
      ))}

      {quality && (
        <>
          <div className="metrics-section-title">🎯 RAG Quality</div>
          {quality.map((item, i) => (
            <div className="metric-card" key={"q" + i}>
              <div className="metric-label">{item.label}</div>
              <div className={"metric-value " + item.cls}>{item.value}</div>
            </div>
          ))}
          <div className="metrics-section-title">🧠 Decision</div>
          <div className="metric-card" style={{ gridColumn: "1/-1" }}>
            <div className="metric-label">Decision Type</div>
            <div style={{ marginTop: 6 }}>
              <span className={"decision-badge " + decision.cls}>{decision.label}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================
// SINGLE TURN (unchanged)
// ============================================================
function Turn({ turn }) {
  const [traceOpen, setTraceOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);

  return (
    <div className="turn">
      <div className="turn-question">
        <div className="turn-avatar avatar-user">U</div>
        <div className="bubble-user">{turn.question}</div>
      </div>

      {turn.status === "processing" ? (
        <div className="processing-turn">
          <div className="turn-avatar avatar-ai">✦</div>
          <div className="thinking-bubble">
            <div className="thinking-dots"><span></span><span></span><span></span></div>
            <div className="thinking-label">{turn.thinkingLabel || "Processing…"}</div>
          </div>
        </div>
      ) : (
        <>
          <div className="turn-answer">
            <div className="turn-avatar avatar-ai">✦</div>
            <div className="bubble-ai">{turn.answer}</div>
          </div>

          {turn.rewritten && turn.rewritten !== turn.question && (
            <div className="rewrite-note">🧠 Query rewritten: "{turn.rewritten}"</div>
          )}

          {turn.trace && turn.trace.length > 0 && (
            <div className="trace-wrap">
              <button
                className={"trace-toggle" + (traceOpen ? " open" : "")}
                onClick={() => setTraceOpen(o => !o)}
              >
                REASONING TRACE
              </button>
              <div className={"trace-list" + (traceOpen ? " open" : "")}>
                {turn.trace.map((item, i) => (
                  <div className="trace-item" key={i}>
                    <span className="trace-icon">{TRACE_ICONS[item.type] || "·"}</span>
                    <span className="trace-text">{item.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {turn.metrics && Object.keys(turn.metrics).length > 0 && (
            <div className="metrics-wrap">
              <button
                className={"metrics-toggle" + (metricsOpen ? " open" : "")}
                onClick={() => setMetricsOpen(o => !o)}
              >
                BENCHMARK METRICS
              </button>
              <div className={"metrics-body" + (metricsOpen ? " open" : "")}>
                {metricsOpen && <MetricsBody m={turn.metrics} />}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [question, setQuestion] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [turns, setTurns] = useState([]);

  const textareaRef = useRef(null);
  const historyEndRef = useRef(null);
  const evtSourceRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [question]);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // ── ASK ──
  function fillQ(text) {
    setQuestion(text);
    textareaRef.current?.focus();
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  }

  function askQuestion() {
    const q = question.trim();
    if (!q || isProcessing) return;
    // No PDF-upload gate — the fixed 3GPP corpus is always ready.

    setIsProcessing(true);
    const requestId = generateId();
    const turnId = requestId;

    setQuestion("");
    setTurns(prev => [...prev, { id: turnId, question: q, status: "processing", thinkingLabel: "Searching 3GPP specs…", trace: [] }]);

    const evtSource = new EventSource(`${API_BASE}/stream?request_id=${requestId}`);
    evtSourceRef.current = evtSource;
    evtSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "heartbeat") return;
      if (data.type === "done") { evtSource.close(); return; }
      if (data.type === "token") return;

      setTurns(prev => prev.map(t => {
        if (t.id !== turnId) return t;
        const nextTrace = [...t.trace, { type: data.type, message: data.message }];
        const nextLabel = (data.type === "extract_start" || data.type === "workflow_start")
          ? data.message.replace(/📄|⚙️/g, "").trim()
          : t.thinkingLabel;
        return { ...t, trace: nextTrace, thinkingLabel: nextLabel };
      }));
    };
    evtSource.onerror = () => evtSource.close();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3600000);

    fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, request_id: requestId }),
      signal: controller.signal,
    })
      .then(r => r.json())
      .then(data => {
        clearTimeout(timeout);
        evtSource.close();
        finishTurn(turnId, data);
      })
      .catch(err => {
        clearTimeout(timeout);
        evtSource.close();
        finishTurn(turnId, {
          answer: err.name === "AbortError" ? "Request timed out." : "Error: " + err.message,
          metrics: null,
        });
      });
  }

  function finishTurn(turnId, data) {
    setIsProcessing(false);
    setTurns(prev => prev.map(t => {
      if (t.id !== turnId) return t;
      return {
        ...t,
        status: "done",
        answer: data.answer || "No answer returned.",
        rewritten: data.rewritten_query,
        metrics: data.metrics,
      };
    }));
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-name">DCO <em>Mind</em></div>
          <div className="brand-sub">3GPP Standards Assistant</div>
        </div>

        <div className="sidebar-divider"></div>

        <div className="upload-section">
          <div className="section-label">Knowledge Base</div>
          <div className="stack-pill"><span className="stack-dot dot-green"></span> TS 38.300 — loaded</div>
          <div className="stack-pill"><span className="stack-dot dot-green"></span> TS 38.331 — loaded</div>
        </div>

        <div className="sidebar-divider"></div>

        <div className="stack-section">
          <div className="section-label">Stack</div>
          <div className="stack-pill"><span className="stack-dot dot-green"></span> LLaMA 3.2 · Ollama</div>
          <div className="stack-pill"><span className="stack-dot dot-blue"></span> LangGraph · ReAct</div>
          <div className="stack-pill"><span className="stack-dot dot-orange"></span> FAISS · RoBERTa</div>
          <div className="stack-pill"><span className="stack-dot dot-purple"></span> RAPTOR · Reranker</div>
        </div>
      </aside>

      <main className="main">
        <div className="chat-history">
          {turns.length === 0 && (
            <div className="empty-state">
              <div className="empty-title">3GPP Standards<br /><em>Assistant</em></div>
              <div className="empty-sub">Ask questions about TS 38.300 and TS 38.331 — no upload needed, both specs are already loaded.</div>
              <div className="suggestions">
                <div className="suggestion-chip" onClick={() => fillQ("What is the RRC_IDLE state?")}>What is RRC_IDLE?</div>
                <div className="suggestion-chip" onClick={() => fillQ("What happens when a UE goes to RRC_IDLE?")}>UE → RRC_IDLE</div>
                <div className="suggestion-chip" onClick={() => fillQ("What are the UE RRC states in NR?")}>UE RRC states in NR</div>
              </div>
            </div>
          )}

          {turns.map(turn => <Turn key={turn.id} turn={turn} />)}
          <div ref={historyEndRef} />
        </div>

        <div className="input-bar">
          <div className="input-inner">
            <textarea
              ref={textareaRef}
              className="question-input"
              rows={1}
              placeholder="Ask a question about TS 38.300 or TS 38.331…"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button className="send-btn" onClick={askQuestion} disabled={isProcessing}>↑</button>
          </div>
          <div className="input-hint">Enter to send · Shift+Enter for new line</div>
        </div>
      </main>
    </div>
  );
}