import { useCallback, useMemo, useRef, useState } from "react";

// In Docker (production), use empty string for relative URLs that nginx proxies
// In development, use localhost:8000
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 
  (import.meta.env.PROD ? "" : "http://localhost:8000");

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [threadId, setThreadId] = useState(null);
  const abortControllerRef = useRef(null);
  
  const suggestions = [
    "How do I add persistence to a LangGraph agent?",
    "What's the difference between StateGraph and MessageGraph?",
    "Show me how to implement human-in-the-loop with LangGraph",
    "How do I handle errors and retries in LangGraph nodes?",
    "What are best practices for state management in LangGraph?",
  ];

  const canSend = useMemo(
    () => question.trim().length >= 3 && !isLoading,
    [question, isLoading],
  );

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!canSend) {
      return;
    }

    const trimmedQuestion = question.trim();
    setQuestion("");
    setError("");
    setIsLoading(true);
    
    // Add user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmedQuestion },
    ]);

    // Add placeholder for assistant response with reasoning section
    const assistantMessageIndex = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { 
        role: "assistant", 
        reasoning: [],  // Array of reasoning steps
        content: "",    // Final answer content
        citations: [], 
        mode: null, 
        isStreaming: true,
        iterations: 0,
      },
    ]);

    try {
      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController();
      
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          thread_id: threadId,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail?.detail || "Request failed.");
      }

      // Read SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let collectedContent = "";
      let collectedReasoning = [];
      let collectedCitations = [];
      let mode = null;
      let finalAnswerStarted = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        let currentEventType = null;

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEventType = line.slice(7);
            continue;
          }
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              
              // Handle reasoning events
              if (currentEventType === "reasoning" && parsed.message) {
                collectedReasoning.push({
                  step: parsed.step,
                  message: parsed.message,
                  tool: parsed.tool,
                  query: parsed.query,
                  decision: parsed.decision,
                });
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    ...updated[assistantMessageIndex],
                    reasoning: [...collectedReasoning],
                  };
                  return updated;
                });
              }
              // Handle final answer start marker
              else if (currentEventType === "final_answer_start") {
                finalAnswerStarted = true;
              }
              // Handle token events (final answer content)
              else if (parsed.content) {
                collectedContent += parsed.content;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    ...updated[assistantMessageIndex],
                    content: collectedContent,
                  };
                  return updated;
                });
              } 
              // Handle citation events
              else if (parsed.source_url) {
                collectedCitations.push(parsed);
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    ...updated[assistantMessageIndex],
                    citations: [...collectedCitations],
                  };
                  return updated;
                });
              } 
              // Handle done event
              else if (parsed.thread_id) {
                setThreadId(parsed.thread_id);
                mode = parsed.mode;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[assistantMessageIndex] = {
                    ...updated[assistantMessageIndex],
                    mode: parsed.mode,
                    iterations: parsed.iterations || 0,
                    isStreaming: false,
                  };
                  return updated;
                });
              } 
              // Handle error events
              else if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch (parseError) {
              // Ignore parse errors for incomplete data
              if (parseError.message && !parseError.message.includes("JSON")) {
                throw parseError;
              }
            }
          }
        }
      }

      // Finalize message
      setMessages((prev) => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          ...updated[assistantMessageIndex],
          isStreaming: false,
        };
        return updated;
      });

    } catch (err) {
      if (err.name === "AbortError") {
        // Request was cancelled
        return;
      }
      setError(err.message || "Unable to reach the server.");
      // Remove the placeholder message on error
      setMessages((prev) => prev.filter((_, i) => i !== assistantMessageIndex));
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setThreadId(null);
    setError("");
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LangGraph Helper Agent</h1>
          <p className="subtitle">Ask questions about LangGraph and LangChain</p>
        </div>
        {messages.length > 0 && (
          <button 
            type="button" 
            className="new-conversation"
            onClick={handleNewConversation}
          >
            New Conversation
          </button>
        )}
      </header>

      <main className="chat">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="suggestions-label">Try one of these:</div>
            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="suggestion"
                  onClick={() => setQuestion(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`message ${message.role} ${message.isStreaming ? 'streaming' : ''}`}
            >
              <div className="role">{message.role}</div>
              
              {/* Reasoning Process Section */}
              {message.reasoning?.length > 0 && (
                <div className="reasoning-section">
                  <div className="reasoning-header">
                    <span className="reasoning-icon">🧠</span>
                    <span>Reasoning Process</span>
                    {message.iterations > 0 && !message.isStreaming && (
                      <span className="iteration-badge">{message.iterations} iteration(s)</span>
                    )}
                  </div>
                  <div className="reasoning-steps">
                    {message.reasoning.map((step, stepIndex) => (
                      <div 
                        key={stepIndex} 
                        className={`reasoning-step ${step.step}`}
                      >
                        {step.message}
                      </div>
                    ))}
                    {message.isStreaming && !message.content && (
                      <div className="reasoning-step thinking">
                        <span className="thinking-dots">...</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* User message content */}
              {message.role === "user" && message.content && (
                <div className="content">{message.content}</div>
              )}
              
              {/* Final Answer Section (assistant only) */}
              {message.role === "assistant" && (message.content || (message.isStreaming && message.reasoning?.some(r => r.step === 'respond'))) && (
                <div className="final-answer-section">
                  <div className="final-answer-header">
                    <span className="answer-icon">✨</span>
                    <span>[FINAL ANSWER]</span>
                  </div>
                  <div className="content">
                    {message.content}
                    {message.isStreaming && <span className="cursor">▊</span>}
                  </div>
                </div>
              )}
              
              {message.citations?.length > 0 && (
                <div className="citations">
                  <div className="label">Citations</div>
                  <ul>
                    {message.citations.map((citation, citationIndex) => (
                      <li key={`${citation.source_url}-${citationIndex}`}>
                        <a
                          href={citation.source_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {citation.title || citation.source_url}
                        </a>
                        <p>{citation.snippet}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {message.mode && !message.isStreaming && (
                <div className="mode">
                  Mode: <strong>{message.mode}</strong>
                </div>
              )}
            </div>
          ))
        )}
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about LangGraph or LangChain..."
          rows={3}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && canSend) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <div className="controls">
          {threadId && (
            <span className="thread-indicator" title={`Thread: ${threadId}`}>
              Conversation active
            </span>
          )}
          <button type="submit" disabled={!canSend}>
            {isLoading ? "Thinking..." : "Send"}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </form>
    </div>
  );
}
