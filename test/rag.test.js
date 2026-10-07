import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chunk, cosine, topK } from '../src/rag.js';

test('chunk packs paragraphs without exceeding maxChars', () => {
  const text = ['a'.repeat(40), 'b'.repeat(40), 'c'.repeat(40)].join('\n\n');
  const chunks = chunk(text, 100);
  assert.equal(chunks.length, 2);
  assert.ok(chunks.every((c) => c.length <= 100));
  assert.ok(chunks[0].includes('a') && chunks[0].includes('b'));
});

test('chunk hard-splits a paragraph longer than maxChars and drops blank input', () => {
  assert.deepEqual(chunk('x'.repeat(250), 100).map((c) => c.length), [100, 100, 50]);
  assert.deepEqual(chunk('  \n\n  '), []);
});

test('cosine: same direction 1, orthogonal 0, opposite -1, zero vector 0', () => {
  assert.equal(cosine([1, 2], [2, 4]).toFixed(6), '1.000000');
  assert.equal(cosine([1, 0], [0, 1]), 0);
  assert.equal(cosine([1, 0], [-1, 0]), -1);
  assert.equal(cosine([0, 0], [1, 1]), 0);
});

test('topK returns the closest chunks first and omits vectors', () => {
  const store = [
    { source: 'far', text: 'far', vector: [0, 1] },
    { source: 'near', text: 'near', vector: [1, 0.1] },
    { source: 'mid', text: 'mid', vector: [1, 1] },
  ];
  const hits = topK([1, 0], store, 2);
  assert.deepEqual(hits.map((h) => h.source), ['near', 'mid']);
  assert.equal(hits[0].vector, undefined);
});
