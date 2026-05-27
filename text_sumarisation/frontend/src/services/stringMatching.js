import { api } from './api';

export const ALGORITHMS = {
  naive: { name: 'Naive (Brute-force)', complexity: 'O(n · m)' },
  kmp: { name: 'Knuth-Morris-Pratt', complexity: 'O(n + m)' },
  boyer_moore: { name: 'Boyer-Moore', complexity: 'O(n/m) tốt nhất, O(n·m) xấu nhất' },
};

const ENDPOINT = {
  naive: '/string-matching/naive',
  kmp: '/string-matching/kmp',
  boyer_moore: '/string-matching/boyer-moore',
};

const normalize = (r) => ({
  algorithm: r.algorithm,
  positions: r.positions || [],
  comparisons: r.comparisons || 0,
  steps: r.steps || [],
  elapsedMs: r.elapsed_ms ?? null,
  lps: r.lps,
  badChar: r.badChar,
  goodSuffix: r.goodSuffix,
});

export async function runOne(algoKey, text, pattern, { caseSensitive = false, trace = true } = {}) {
  const url = ENDPOINT[algoKey];
  if (!url) throw new Error(`Unknown algorithm: ${algoKey}`);
  const { data } = await api.post(url, {
    text,
    pattern,
    case_sensitive: caseSensitive,
    trace,
  });
  return normalize(data?.result || {});
}

export async function runStringMatching(text, pattern, algorithms, opts = {}) {
  const entries = await Promise.all(
    algorithms.map(async (k) => [k, await runOne(k, text, pattern, opts)])
  );
  return Object.fromEntries(entries);
}
