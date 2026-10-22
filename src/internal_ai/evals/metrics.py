"""Evaluation metrics (PRD section 12)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


def recall_at_k(relevance: dict[str, set[str]], rankings: dict[str, list[str]], k: int) -> float:
    if not relevance:
        return 0.0
    total = 0.0
    for query_id, relevant in relevance.items():
        if not relevant:
            continue
        retrieved = set(rankings.get(query_id, [])[:k])
        total += len(relevant & retrieved) / len(relevant)
    return total / len(relevance) if relevance else 0.0


def mean_reciprocal_rank(relevance: dict[str, set[str]], rankings: dict[str, list[str]]) -> float:
    if not relevance:
        return 0.0
    total = 0.0
    for query_id, relevant in relevance.items():
        if not relevant:
            continue
        for position, doc_id in enumerate(rankings.get(query_id, []), start=1):
            if doc_id in relevant:
                total += 1 / position
                break
    return total / len(relevance)


def ndcg_at_k(relevance: dict[str, set[str]], rankings: dict[str, list[str]], k: int = 10) -> float:
    if not relevance:
        return 0.0
    total = 0.0
    for query_id, relevant in relevance.items():
        if not relevant:
            continue
        gains = [1.0 if doc_id in relevant else 0.0 for doc_id in rankings.get(query_id, [])[:k]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_gains = sorted([1.0] * min(len(relevant), k), reverse=True)
        idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
        total += dcg / idcg if idcg else 0.0
    return total / len(relevance)


def precision_recall_f1(
    predictions: Sequence[str], labels: Sequence[str], positive: str = "positive"
) -> dict[str, float]:
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    tp = sum(
        1 for pred, label in zip(predictions, labels, strict=False) if pred == label == positive
    )
    fp = sum(
        1
        for pred, label in zip(predictions, labels, strict=False)
        if pred == positive and label != positive
    )
    fn = sum(
        1
        for pred, label in zip(predictions, labels, strict=False)
        if pred != positive and label == positive
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(
        1 for pred, label in zip(predictions, labels, strict=False) if pred == label
    ) / len(labels)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    if not labels:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=False)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probability for probability, _ in members) / len(members)
        accuracy = sum(1 for _, label in members if label == 1) / len(members)
        error += (len(members) / len(labels)) * abs(confidence - accuracy)
    return round(error, 6)


def perplexity(log_probs: Sequence[float]) -> float:
    if not log_probs:
        return float("inf")
    total = sum(log_probs)
    return round(math.exp(-total / len(log_probs)), 6)


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not references:
        return 0.0
    return round(
        sum(
            1
            for pred, ref in zip(predictions, references, strict=False)
            if pred.strip() == ref.strip()
        )
        / len(references),
        6,
    )


def token_accuracy(
    predictions: Sequence[Sequence[int]], references: Sequence[Sequence[int]]
) -> float:
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references, strict=False):
        for p, r in zip(pred, ref, strict=False):
            total += 1
            correct += 1 if p == r else 0
    return round(correct / total, 6) if total else 0.0


def bm25_scores(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Keyword search baseline (PRD section 12)."""
    import re

    token_pattern = re.compile(r"\w+")
    query_terms = [term.lower() for term in token_pattern.findall(query)]
    doc_tokens = [[term.lower() for term in token_pattern.findall(doc)] for doc in documents]
    lengths = [len(tokens) for tokens in doc_tokens] or [1]
    avg_length = sum(lengths) / len(lengths)
    frequencies: Counter[str] = Counter()
    for tokens in doc_tokens:
        frequencies.update(set(tokens))
    total_docs = max(len(documents), 1)
    scores = []
    for tokens, length in zip(doc_tokens, lengths, strict=False):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = frequencies.get(term, 0)
            idf = math.log(1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + k1 * (1 - b + b * length / avg_length)
            score += idf * (term_frequency * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def rank_from_scores(scores: list[float], doc_ids: list[str]) -> list[str]:
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], doc_ids[index]))
    return [doc_ids[index] for index in order]


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-9, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-9, None)
    return left_norm @ right_norm.T


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "count": int(array.size),
    }
