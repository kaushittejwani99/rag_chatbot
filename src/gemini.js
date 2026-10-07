// Thin wrapper over the Gemini REST API (native fetch, no SDK).
const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
const CHAT_MODEL = process.env.GEMINI_CHAT_MODEL || 'gemini-3.5-flash-lite';
const EMBED_MODEL = process.env.GEMINI_EMBED_MODEL || 'gemini-embedding-001';
const BATCH_SIZE = 100; // batchEmbedContents accepts at most 100 requests

async function call(model, method, body) {
  const res = await fetch(`${BASE}/${model}:${method}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-goog-api-key': process.env.GEMINI_API_KEY,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = new Error(`Gemini ${method} failed (${res.status}): ${await res.text()}`);
    err.status = 502;
    throw err;
  }
  return res.json();
}

// taskType: RETRIEVAL_DOCUMENT for stored chunks, RETRIEVAL_QUERY for the user's question.
export async function embed(texts, taskType) {
  const vectors = [];
  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const { embeddings } = await call(EMBED_MODEL, 'batchEmbedContents', {
      requests: texts.slice(i, i + BATCH_SIZE).map((text) => ({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text }] },
        embedContentConfig: { taskType, outputDimensionality: 768 },
      })),
    });
    vectors.push(...embeddings.map((e) => e.values));
  }
  return vectors;
}

export async function generate(systemInstruction, contents) {
  const data = await call(CHAT_MODEL, 'generateContent', {
    systemInstruction: { parts: [{ text: systemInstruction }] },
    contents,
  });
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text ?? '').join('');
  if (!text) {
    const reason = data.promptFeedback?.blockReason ?? data.candidates?.[0]?.finishReason ?? 'unknown';
    const err = new Error(`Gemini returned no text (${reason})`);
    err.status = 502;
    throw err;
  }
  return text;
}
