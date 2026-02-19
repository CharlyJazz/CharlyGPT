import { useEffect, useRef, useState } from "react";
import { streamChat } from "./api.js";

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (event) => {
    event.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError("");

    const assistantMessage = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      await streamChat({
        prompt: input,
        maxNewTokens: 512,
        onToken: (chunk) => {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: updated[updated.length - 1].content + chunk,
            };
            return updated;
          });
        },
      });
    } catch (err) {
      setError(err.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend(event);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-base-900">
      <header className="shrink-0 border-b border-base-700/60 bg-base-900/90 px-6 py-3">
        <h1 className="text-base font-semibold text-base-100">Chat</h1>
      </header>

      <main className="chat-scrollbar flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <p className="text-center text-sm text-base-200/50">Envía un mensaje para comenzar</p>
          )}
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm ${
                  msg.role === "user"
                    ? "bg-base-100 text-base-900"
                    : "bg-base-800 text-base-100"
                }`}
              >
                {msg.content || (loading && idx === messages.length - 1 ? "..." : "")}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {error && (
        <div className="shrink-0 bg-red-500/10 px-6 py-2 text-center text-xs text-red-300">
          {error}
        </div>
      )}

      <footer className="shrink-0 border-t border-base-700/60 bg-base-900/90 px-4 py-4">
        <form onSubmit={handleSend} className="mx-auto flex max-w-3xl items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Escribe un mensaje..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-base-600 bg-base-800 px-4 py-3 text-sm text-base-100 placeholder:text-base-200/50 focus:border-base-200/60 focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-xl bg-base-100 px-4 py-3 text-sm font-semibold text-base-900 transition hover:bg-base-200 disabled:opacity-50"
          >
            {loading ? "..." : "Enviar"}
          </button>
        </form>
      </footer>
    </div>
  );
}
