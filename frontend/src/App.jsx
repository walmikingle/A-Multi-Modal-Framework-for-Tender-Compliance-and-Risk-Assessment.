import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("single");

  // -----------------------------
  // Single PDF state
  // -----------------------------
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [singleStatus, setSingleStatus] = useState("offline");
  const [singleInitTime, setSingleInitTime] = useState(null);

  // -----------------------------
  // Multi-document state
  // -----------------------------
  const [multiStatus, setMultiStatus] = useState("offline");
  const [multiDocumentCount, setMultiDocumentCount] = useState(12);
  const [multiInitTime, setMultiInitTime] = useState(null);

  // -----------------------------
  // Common query state
  // -----------------------------
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [queryTime, setQueryTime] = useState(null);

  const [loadingEngine, setLoadingEngine] = useState(false);
  const [loadingAsk, setLoadingAsk] = useState(false);
  const [error, setError] = useState("");

  // --------------------------------
  // Helpers
  // --------------------------------
  const resetResponse = () => {
    setAnswer("");
    setSources([]);
    setQueryTime(null);
    setError("");
  };

  const handleModeChange = (nextMode) => {
    setMode(nextMode);
    resetResponse();
    setQuestion("");
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case "ready":
        return "Ready";
      case "initializing":
        return "Initializing";
      case "loading":
        return "Loading";
      case "error":
        return "Error";
      default:
        return "Not started";
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case "ready":
        return "status-ready";
      case "initializing":
      case "loading":
        return "status-loading";
      case "error":
        return "status-error";
      default:
        return "status-offline";
    }
  };

  // --------------------------------
  // Load multi status on startup
  // --------------------------------
  useEffect(() => {
    const loadMultiStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/multi/status`);

        if (!response.ok) {
          return;
        }

        const data = await response.json();

        setMultiStatus(data.status || "offline");

        if (data.document_count) {
          setMultiDocumentCount(data.document_count);
        }

        if (data.initialization_time !== undefined) {
          setMultiInitTime(data.initialization_time);
        }
      } catch {
        // Backend may not be running yet.
      }
    };

    loadMultiStatus();
  }, []);

  // --------------------------------
  // Select PDF
  // --------------------------------
  const handleFileSelect = (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (
      file.type !== "application/pdf" &&
      !file.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("Please select a PDF file.");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setSingleStatus("offline");
    setSingleInitTime(null);

    resetResponse();
  };

  // --------------------------------
  // Start single-document engine
  // --------------------------------
  const startSingleEngine = async () => {
    if (!selectedFile) {
      setError("Please choose a PDF first.");
      return;
    }

    setLoadingEngine(true);
    setSingleStatus("initializing");
    setError("");
    resetResponse();

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/api/single/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to initialize the single-document engine.",
        );
      }

      setSingleStatus(data.status || "ready");
      setSingleInitTime(data.initialization_time ?? null);
    } catch (err) {
      setSingleStatus("error");
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoadingEngine(false);
    }
  };

  // --------------------------------
  // Start multi-document engine
  // --------------------------------
  const startMultiEngine = async () => {
    setLoadingEngine(true);
    setMultiStatus("initializing");
    setError("");
    resetResponse();

    try {
      const response = await fetch(`${API_BASE}/api/multi/start`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Failed to initialize the multi-document engine.",
        );
      }

      setMultiStatus(data.status || "ready");

      if (data.document_count) {
        setMultiDocumentCount(data.document_count);
      }

      setMultiInitTime(data.initialization_time ?? null);
    } catch (err) {
      setMultiStatus("error");
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoadingEngine(false);
    }
  };

  // --------------------------------
  // Ask question
  // --------------------------------
  const askQuestion = async () => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Please enter a question.");
      return;
    }

    const engineReady =
      mode === "single" ? singleStatus === "ready" : multiStatus === "ready";

    if (!engineReady) {
      setError("Start the RAG engine before asking a question.");
      return;
    }

    setLoadingAsk(true);
    setError("");
    setAnswer("");
    setSources([]);
    setQueryTime(null);

    try {
      const endpoint =
        mode === "single"
          ? `${API_BASE}/api/single/ask`
          : `${API_BASE}/api/multi/ask`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to process the question.");
      }

      setAnswer(data.answer || "No answer returned.");
      setSources(Array.isArray(data.sources) ? data.sources : []);
      setQueryTime(data.query_time ?? null);
    } catch (err) {
      setError(err.message || "Failed to process the question.");
    } finally {
      setLoadingAsk(false);
    }
  };

  // --------------------------------
  // Enter / Ctrl+Enter
  // --------------------------------
  const handleQuestionKeyDown = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      askQuestion();
    }
  };

  const engineStatus = mode === "single" ? singleStatus : multiStatus;
  const engineReady = engineStatus === "ready";

  return (
    <div className="app">
      {/* -------------------------------- Header -------------------------------- */}
      <header className="topbar">
        <div className="brand">
          <h1>Tender RAG</h1>
          <p>Document intelligence and tender analysis</p>
        </div>

        <div className="engine-status">
          <span className={`status-dot ${getStatusClass(engineStatus)}`} />
          <span>{getStatusLabel(engineStatus)}</span>
        </div>
      </header>

      {/* -------------------------------- Main -------------------------------- */}
      <main className="content">
        {/* Mode switch */}
        <div className="mode-switch" role="tablist">
          <button
            className={mode === "single" ? "active" : ""}
            onClick={() => handleModeChange("single")}
          >
            Single PDF
          </button>

          <button
            className={mode === "multi" ? "active" : ""}
            onClick={() => handleModeChange("multi")}
          >
            Multi Document
          </button>
        </div>

        {/* ============================== SINGLE ============================== */}
        {mode === "single" && (
          <section className="panel">
            <div className="section-heading">
              <div>
                <h2>Single Document Analysis</h2>
                <p>
                  Upload a tender PDF and ask grounded questions with page-level
                  citations.
                </p>
              </div>
            </div>

            {!selectedFile ? (
              <div className="upload-box">
                <div className="upload-icon">PDF</div>

                <h3>Upload tender document</h3>

                <p>
                  Select a PDF to initialize the single-document RAG engine.
                </p>

                <button
                  className="primary-button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Choose PDF
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  hidden
                  onChange={handleFileSelect}
                />
              </div>
            ) : (
              <div className="document-area">
                <div className="document-row">
                  <div>
                    <span className="field-label">Document</span>
                    <div className="document-name">{selectedFile.name}</div>
                  </div>

                  <span
                    className={`inline-status ${getStatusClass(singleStatus)}`}
                  >
                    <span
                      className={`status-dot ${getStatusClass(singleStatus)}`}
                    />
                    {getStatusLabel(singleStatus)}
                  </span>
                </div>

                {!engineReady && (
                  <div className="document-actions">
                    <button
                      className="secondary-button"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Change PDF
                    </button>

                    <button
                      className="primary-button"
                      onClick={startSingleEngine}
                      disabled={loadingEngine}
                    >
                      {loadingEngine ? "Initializing..." : "Start Engine"}
                    </button>

                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,application/pdf"
                      hidden
                      onChange={handleFileSelect}
                    />
                  </div>
                )}

                {engineReady && singleInitTime !== null && (
                  <div className="engine-info">
                    Engine initialized in {Number(singleInitTime).toFixed(3)}s
                  </div>
                )}
              </div>
            )}

            {/* Question */}
            {engineReady && (
              <>
                <div className="question-section">
                  <div className="field-label">Ask a question</div>

                  <div className="question-row">
                    <textarea
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                      onKeyDown={handleQuestionKeyDown}
                      placeholder="Ask something about the tender..."
                      rows={3}
                    />

                    <button
                      className="primary-button ask-button"
                      onClick={askQuestion}
                      disabled={loadingAsk}
                    >
                      {loadingAsk ? "Asking..." : "Ask"}
                    </button>
                  </div>

                  <div className="input-hint">Press Ctrl + Enter to submit</div>
                </div>
              </>
            )}

            {/* Error */}
            {error && <div className="error-message">{error}</div>}

            {/* Answer */}
            {answer && (
              <section className="answer-section">
                <div className="answer-header">
                  <h3>Answer</h3>

                  {queryTime !== null && (
                    <span className="query-time">
                      {Number(queryTime).toFixed(2)}s
                    </span>
                  )}
                </div>

                <div className="answer-text">{answer}</div>

                {sources.length > 0 && (
                  <div className="sources">
                    <div className="field-label">Sources</div>

                    <div className="source-list">
                      {sources.map((source, index) => (
                        <span
                          className="source-chip"
                          key={`${source}-${index}`}
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}
          </section>
        )}

        {/* ============================== MULTI ============================== */}
        {mode === "multi" && (
          <section className="panel">
            <div className="section-heading">
              <div>
                <h2>Multi-Document Testing</h2>
                <p>
                  Test the RAG system across the tender documents already
                  available in the backend.
                </p>
              </div>
            </div>

            <div className="multi-overview">
              <div className="metric">
                <span>Available Documents</span>
                <strong>{multiDocumentCount}</strong>
              </div>

              <div className="metric">
                <span>Engine Status</span>
                <strong>{getStatusLabel(multiStatus)}</strong>
              </div>

              {multiInitTime !== null && (
                <div className="metric">
                  <span>Initialization</span>
                  <strong>{Number(multiInitTime).toFixed(2)}s</strong>
                </div>
              )}
            </div>

            {!engineReady && (
              <div className="multi-actions">
                <button
                  className="primary-button"
                  onClick={startMultiEngine}
                  disabled={loadingEngine}
                >
                  {loadingEngine
                    ? "Initializing..."
                    : "Start Multi-Document Engine"}
                </button>
              </div>
            )}

            {engineReady && (
              <div className="question-section multi-question">
                <div className="field-label">Ask a question</div>

                <div className="question-row">
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={handleQuestionKeyDown}
                    placeholder="Ask a question across the tender documents..."
                    rows={3}
                  />

                  <button
                    className="primary-button ask-button"
                    onClick={askQuestion}
                    disabled={loadingAsk}
                  >
                    {loadingAsk ? "Asking..." : "Ask"}
                  </button>
                </div>

                <div className="input-hint">Press Ctrl + Enter to submit</div>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {answer && (
              <section className="answer-section">
                <div className="answer-header">
                  <h3>Answer</h3>

                  {queryTime !== null && (
                    <span className="query-time">
                      {Number(queryTime).toFixed(2)}s
                    </span>
                  )}
                </div>

                <div className="answer-text">{answer}</div>

                {sources.length > 0 && (
                  <div className="sources">
                    <div className="field-label">Sources</div>

                    <div className="source-list">
                      {sources.map((source, index) => (
                        <span
                          className="source-chip"
                          key={`${source}-${index}`}
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
