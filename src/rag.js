// Pure retrieval logic: no network, no disk. Covered by test/rag.test.js.

// Packs whole paragraphs into chunks of up to maxChars so a chunk rarely cuts a thought in half.
// ponytail: no overlap between chunks; add a sliding window if answers that span two chunks get missed.
export function chunk(text, maxChars = 500) {
  const chunks = [];
  let current = '';
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  for (const para of paragraphs) {
    if (current && current.length + para.length + 2 > maxChars) {
      chunks.push(current);
      current = '';
    }
    if (para.length > maxChars) {
      for (let i = 0; i < para.length; i += maxChars) chunks.push(para.slice(i, i + maxChars));
      continue;
    }
    current = current ? `${current}\n\n${para}` : para;
  }
  if (current) chunks.push(current);
  return chunks;
}

export function cosine(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// ponytail: linear scan over every chunk, fine up to ~10k chunks; move to pgvector beyond that.
export function topK(queryVector, store, k = 4) {
  return store
    .map(({ source, text, vector }) => ({ source, text, score: cosine(queryVector, vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}
