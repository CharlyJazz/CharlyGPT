const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function streamChat({ prompt, maxNewTokens, onToken }) {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      prompt,
      max_new_tokens: maxNewTokens
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`API error: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let done = false;

  while (!done) {
    const result = await reader.read();
    done = result.done;
    if (result.value) {
      const chunk = decoder.decode(result.value, { stream: true });
      if (chunk) {
        onToken(chunk);
      }
    }
  }
}
