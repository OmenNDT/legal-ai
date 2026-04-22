"""TextRank centrality scoring for extractive summarization.

Computes PageRank on a sentence similarity graph to identify
the most central/important sentences in a document.

Time complexity: O(n²) for similarity matrix + O(n·k) for PageRank iterations
where n = number of sentences, k = number of iterations (typically converges in <50)
"""

import numpy as np
from typing import Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_similarity_matrix(embeddings: list[np.ndarray], threshold: float = 0.1) -> np.ndarray:
    """Build sentence similarity matrix from embeddings.

    Args:
        embeddings: List of sentence embedding vectors
        threshold: Minimum similarity to include edge (prunes weak connections)

    Returns:
        n×n adjacency matrix with cosine similarities
    """
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

    return sim_matrix


def textrank(
    sim_matrix: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Run PageRank on the sentence similarity graph.

    Args:
        sim_matrix: n×n similarity matrix
        damping: Damping factor (0.85 default, same as original PageRank)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Array of TextRank scores, one per sentence
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Normalize similarity matrix (row-stochastic transition matrix)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    transition = sim_matrix / row_sums

    # Initialize scores uniformly
    scores = np.ones(n) / n

    # Power iteration
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < tolerance:
            break
        scores = new_scores

    return scores


def compute_position_scores(num_sentences: int, first_bonus: float = 0.3) -> np.ndarray:
    """Compute position-based scores for sentences.

    First sentences in a document/section get higher scores.
    Based on the observation that legal documents often state
    key provisions early.

    Args:
        num_sentences: Total number of sentences
        first_bonus: Score bonus for the first sentence

    Returns:
        Array of position scores, normalized to [0, 1]
    """
    positions = np.arange(num_sentences, dtype=float)
    # Inverse position: earlier = higher score
    scores = 1.0 / (1.0 + positions * 0.1)
    # Bonus for very first sentences
    scores[0] += first_bonus
    if num_sentences > 1:
        scores[1] += first_bonus * 0.5
    # Normalize to [0, 1]
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
) -> list[int]:
    """Select top-K sentences using combined scoring.

    Final score = w_rel * relevance + w_cen * centrality + w_pos * position

    Args:
        relevance_scores: Query-sentence cosine similarity scores
        centrality_scores: TextRank PageRank scores
        position_scores: Position-based scores
        weights: Dict with 'relevance', 'centrality', 'position' weights
        top_k: Number of sentences to select
        min_gap: Minimum sentence gap (avoid selecting adjacent sentences)

    Returns:
        List of sentence indices in original order
    """
    if weights is None:
        weights = {"relevance": 0.4, "centrality": 0.4, "position": 0.2}

    combined = (
        weights["relevance"] * relevance_scores
        + weights["centrality"] * centrality_scores
        + weights["position"] * position_scores
    )

    # Select top-K by score, then reorder by original position
    selected = []
    for idx in np.argsort(combined)[::-1]:
        if len(selected) >= top_k:
            break
        # Skip if too close to already selected sentences
        if min_gap > 0 and any(abs(idx - s) <= min_gap for s in selected):
            continue
        selected.append(idx)

    # Return in original document order
    return sorted(selected)