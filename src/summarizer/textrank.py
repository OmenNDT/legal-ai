import numpy as np
from typing import Optional

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def build_similarity_matrix(embeddings: list, threshold: float = 0.1) -> np.ndarray:
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
    return sim_matrix

def textrank(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 100, tolerance: float = 1e-5) -> np.ndarray:
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = sim_matrix / row_sums
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < tolerance:
            break
        scores = new_scores
    return scores

def compute_position_scores(num_sentences: int, first_bonus: float = 0.3) -> np.ndarray:
    positions = np.arange(num_sentences, dtype=float)
    scores = 1.0 / (1.0 + positions * 0.1)
    scores[0] += first_bonus
    if num_sentences > 1:
        scores[1] += first_bonus * 0.5
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores

def select_sentences(
    relevance_scores: np.ndarray,
    centrality_scores: np.ndarray,
    position_scores: np.ndarray,
    weights: dict = None,
    top_k: int = 4,
    min_gap: int = 0,
) -> list:
    if weights is None:
        weights = {"relevance": 0.4, "centrality": 0.4, "position": 0.2}
    combined = (
        weights["relevance"] * relevance_scores
        + weights["centrality"] * centrality_scores
        + weights["position"] * position_scores
    )
    selected = []
    for idx in np.argsort(combined)[::-1]:
        if len(selected) >= top_k:
            break
        if min_gap > 0 and any(abs(idx - s) <= min_gap for s in selected):
            continue
        selected.append(int(idx))
    return sorted(selected)