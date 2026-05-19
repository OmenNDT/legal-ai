// String matching algorithms with step-by-step trace for visualization.
// Each algorithm returns an array of "steps" describing what happened, so
// the UI can animate the process.
//
// Step shape (common):
//   {
//     type: 'compare' | 'match' | 'mismatch' | 'found' | 'shift' | 'lps' | 'info',
//     i: number,         // current text index being compared (or shift base)
//     j: number,         // current pattern index
//     comparisons: number,
//     positions: number[], // matches found so far
//     message: string,
//     // algorithm-specific extras (lps, badChar, shiftBy, ...)
//   }

const normalize = (s, caseSensitive) => (caseSensitive ? s : s.toLowerCase());

// ---------------------------- NAIVE ----------------------------
export function naiveSearch(text, pattern, { caseSensitive = false } = {}) {
  const steps = [];
  const t = normalize(text, caseSensitive);
  const p = normalize(pattern, caseSensitive);
  const n = t.length;
  const m = p.length;
  const positions = [];
  let comparisons = 0;

  steps.push({
    type: 'info',
    i: 0,
    j: 0,
    comparisons,
    positions: [...positions],
    message: `Bắt đầu thuật toán Naive (Brute-force). Pattern dài ${m}, Text dài ${n}.`,
  });

  if (m === 0 || m > n) {
    steps.push({
      type: 'info',
      i: 0,
      j: 0,
      comparisons,
      positions: [...positions],
      message: 'Pattern rỗng hoặc dài hơn Text. Dừng.',
    });
    return { algorithm: 'naive', positions, comparisons, steps };
  }

  for (let i = 0; i <= n - m; i++) {
    let j = 0;
    steps.push({
      type: 'info',
      i,
      j: 0,
      comparisons,
      positions: [...positions],
      message: `Trượt pattern đến vị trí i=${i} trong text.`,
    });
    while (j < m) {
      comparisons++;
      const isMatch = t[i + j] === p[j];
      steps.push({
        type: isMatch ? 'match' : 'mismatch',
        i,
        j,
        comparisons,
        positions: [...positions],
        message: isMatch
          ? `So sánh text[${i + j}]='${text[i + j]}' với pattern[${j}]='${pattern[j]}' → khớp.`
          : `So sánh text[${i + j}]='${text[i + j]}' với pattern[${j}]='${pattern[j]}' → không khớp. Trượt sang phải 1.`,
      });
      if (!isMatch) break;
      j++;
    }
    if (j === m) {
      positions.push(i);
      steps.push({
        type: 'found',
        i,
        j: m - 1,
        comparisons,
        positions: [...positions],
        message: `✓ Tìm thấy pattern tại vị trí ${i}.`,
      });
    }
  }

  steps.push({
    type: 'info',
    i: n,
    j: 0,
    comparisons,
    positions: [...positions],
    message: `Hoàn tất. Tìm thấy ${positions.length} vị trí với ${comparisons} phép so sánh.`,
  });

  return { algorithm: 'naive', positions, comparisons, steps };
}

// ---------------------------- KMP ----------------------------
function buildLPS(p) {
  const m = p.length;
  const lps = new Array(m).fill(0);
  let length = 0;
  let i = 1;
  const lpsSteps = [];
  while (i < m) {
    if (p[i] === p[length]) {
      length++;
      lps[i] = length;
      lpsSteps.push({ i, length, lps: [...lps], note: `p[${i}]='${p[i]}' khớp p[${length - 1}], lps[${i}]=${length}.` });
      i++;
    } else {
      if (length !== 0) {
        lpsSteps.push({ i, length, lps: [...lps], note: `p[${i}]='${p[i]}' không khớp, lùi length=lps[${length - 1}]=${lps[length - 1]}.` });
        length = lps[length - 1];
      } else {
        lps[i] = 0;
        lpsSteps.push({ i, length, lps: [...lps], note: `Không có prefix-suffix tại i=${i}, lps[${i}]=0.` });
        i++;
      }
    }
  }
  return { lps, lpsSteps };
}

export function kmpSearch(text, pattern, { caseSensitive = false } = {}) {
  const steps = [];
  const t = normalize(text, caseSensitive);
  const p = normalize(pattern, caseSensitive);
  const n = t.length;
  const m = p.length;
  const positions = [];
  let comparisons = 0;

  steps.push({
    type: 'info',
    i: 0,
    j: 0,
    comparisons,
    positions: [...positions],
    message: `Bắt đầu thuật toán KMP. Tính mảng LPS cho pattern...`,
  });

  if (m === 0 || m > n) {
    steps.push({
      type: 'info',
      i: 0,
      j: 0,
      comparisons,
      positions: [...positions],
      message: 'Pattern rỗng hoặc dài hơn Text. Dừng.',
    });
    return { algorithm: 'kmp', positions, comparisons, steps, lps: [] };
  }

  const { lps, lpsSteps } = buildLPS(p);
  lpsSteps.forEach((s) => {
    steps.push({
      type: 'lps',
      i: 0,
      j: 0,
      comparisons,
      positions: [...positions],
      lps: s.lps,
      lpsI: s.i,
      message: `[LPS] ${s.note}`,
    });
  });
  steps.push({
    type: 'info',
    i: 0,
    j: 0,
    comparisons,
    positions: [...positions],
    lps,
    message: `LPS = [${lps.join(', ')}]. Bắt đầu duyệt text.`,
  });

  let i = 0;
  let j = 0;
  while (i < n) {
    comparisons++;
    const isMatch = t[i] === p[j];
    steps.push({
      type: isMatch ? 'match' : 'mismatch',
      i: i - j,
      j,
      comparisons,
      positions: [...positions],
      lps,
      textCursor: i,
      message: isMatch
        ? `So sánh text[${i}]='${text[i]}' với pattern[${j}]='${pattern[j]}' → khớp.`
        : `So sánh text[${i}]='${text[i]}' với pattern[${j}]='${pattern[j]}' → không khớp.`,
    });
    if (isMatch) {
      i++;
      j++;
      if (j === m) {
        positions.push(i - j);
        steps.push({
          type: 'found',
          i: i - j,
          j: m - 1,
          comparisons,
          positions: [...positions],
          lps,
          message: `✓ Tìm thấy tại vị trí ${i - j}. Dùng LPS: j = lps[${j - 1}] = ${lps[j - 1]}.`,
        });
        j = lps[j - 1];
      }
    } else {
      if (j !== 0) {
        const newJ = lps[j - 1];
        steps.push({
          type: 'shift',
          i: i - j,
          j: newJ,
          comparisons,
          positions: [...positions],
          lps,
          message: `Mismatch tại j=${j}. Dùng LPS: j = lps[${j - 1}] = ${newJ}. Text cursor không lùi.`,
        });
        j = newJ;
      } else {
        i++;
      }
    }
  }

  steps.push({
    type: 'info',
    i: n,
    j: 0,
    comparisons,
    positions: [...positions],
    lps,
    message: `Hoàn tất KMP. Tìm thấy ${positions.length} vị trí với ${comparisons} phép so sánh.`,
  });

  return { algorithm: 'kmp', positions, comparisons, steps, lps };
}

// ---------------------------- BOYER-MOORE (bad-char) ----------------------------
function buildBadChar(p) {
  const table = {};
  for (let i = 0; i < p.length; i++) table[p[i]] = i;
  return table;
}

export function boyerMooreSearch(text, pattern, { caseSensitive = false } = {}) {
  const steps = [];
  const t = normalize(text, caseSensitive);
  const p = normalize(pattern, caseSensitive);
  const n = t.length;
  const m = p.length;
  const positions = [];
  let comparisons = 0;

  steps.push({
    type: 'info',
    i: 0,
    j: 0,
    comparisons,
    positions: [...positions],
    message: `Bắt đầu thuật toán Boyer-Moore (bad-character).`,
  });

  if (m === 0 || m > n) {
    steps.push({
      type: 'info',
      i: 0,
      j: 0,
      comparisons,
      positions: [...positions],
      message: 'Pattern rỗng hoặc dài hơn Text. Dừng.',
    });
    return { algorithm: 'boyer_moore', positions, comparisons, steps, badChar: {} };
  }

  const badChar = buildBadChar(p);
  steps.push({
    type: 'info',
    i: 0,
    j: 0,
    comparisons,
    positions: [...positions],
    badChar,
    message: `Bảng bad-character: ${JSON.stringify(badChar)}. So sánh pattern từ PHẢI sang TRÁI.`,
  });

  let s = 0;
  while (s <= n - m) {
    let j = m - 1;
    steps.push({
      type: 'info',
      i: s,
      j,
      comparisons,
      positions: [...positions],
      badChar,
      message: `Đặt pattern tại shift s=${s}. Bắt đầu so sánh từ j=${m - 1}.`,
    });
    while (j >= 0) {
      comparisons++;
      const isMatch = p[j] === t[s + j];
      steps.push({
        type: isMatch ? 'match' : 'mismatch',
        i: s,
        j,
        comparisons,
        positions: [...positions],
        badChar,
        message: isMatch
          ? `So sánh text[${s + j}]='${text[s + j]}' với pattern[${j}]='${pattern[j]}' → khớp.`
          : `So sánh text[${s + j}]='${text[s + j]}' với pattern[${j}]='${pattern[j]}' → không khớp.`,
      });
      if (!isMatch) break;
      j--;
    }
    if (j < 0) {
      positions.push(s);
      const nextShift = s + m < n ? Math.max(1, m - (badChar[t[s + m]] ?? -1)) : 1;
      steps.push({
        type: 'found',
        i: s,
        j: 0,
        comparisons,
        positions: [...positions],
        badChar,
        message: `✓ Tìm thấy tại vị trí ${s}. Trượt thêm ${nextShift}.`,
      });
      s += nextShift;
    } else {
      const bc = badChar[t[s + j]] ?? -1;
      const shiftBy = Math.max(1, j - bc);
      steps.push({
        type: 'shift',
        i: s,
        j,
        comparisons,
        positions: [...positions],
        badChar,
        shiftBy,
        message: `Bad-char '${text[s + j]}' tại pattern index ${bc}. Trượt sang phải ${shiftBy} bước.`,
      });
      s += shiftBy;
    }
  }

  steps.push({
    type: 'info',
    i: n,
    j: 0,
    comparisons,
    positions: [...positions],
    badChar,
    message: `Hoàn tất Boyer-Moore. Tìm thấy ${positions.length} vị trí với ${comparisons} phép so sánh.`,
  });

  return { algorithm: 'boyer_moore', positions, comparisons, steps, badChar };
}

export const ALGORITHMS = {
  naive: { name: 'Naive (Brute-force)', complexity: 'O(n · m)', fn: naiveSearch },
  kmp: { name: 'Knuth-Morris-Pratt', complexity: 'O(n + m)', fn: kmpSearch },
  boyer_moore: { name: 'Boyer-Moore', complexity: 'O(n/m) tốt nhất, O(n·m) xấu nhất', fn: boyerMooreSearch },
};
