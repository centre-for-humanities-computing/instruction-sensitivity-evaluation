"""
Deduplicate a list of dicts based on string similarity (normalized Levenshtein
distance) of the text under a given key (default: "generated_prompt"), using
agglomerative clustering on a precomputed distance matrix.

All other keys/values in each dict are preserved untouched.

For debugging, the function also returns a report of what was removed and
which kept item it was considered most similar to (and the distance score).

Install dependency:
    pip install rapidfuzz scikit-learn scipy

Usage:
    python dedup_prompts.py
    (edit `items` below, or import `dedup_prompt_dicts()` into your own code)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.cluster import AgglomerativeClustering


def normalized_levenshtein_distance(a: str, b: str) -> float:
    """Return a 0..1 distance: 0 = identical, 1 = completely different."""
    if not a and not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    return dist / max(len(a), len(b))


def build_distance_matrix(texts: list[str]) -> np.ndarray:
    """
    O(N^2) pairwise normalized Levenshtein distance matrix.
    Fine for a few thousand items; for much larger lists, add a
    length-based pre-filter (see note at bottom of file).
    """
    n = len(texts)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = normalized_levenshtein_distance(texts[i], texts[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def cluster_texts(
    dist_matrix: np.ndarray,
    distance_threshold: float = 0.15,
    linkage: str = "average",
) -> np.ndarray:
    """
    Cluster using agglomerative clustering on the precomputed distance matrix.
    distance_threshold: max normalized distance within a cluster (0..1).
        Lower  -> stricter (only near-identical strings merge)
        Higher -> looser (more aggressive merging)
    linkage: "average" or "complete" recommended over "single" to avoid
        chaining (A~B~C merging even when A and C aren't actually similar).
    """
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    labels = clusterer.fit_predict(dist_matrix)
    return labels


def pick_representative(indices: list[int], dist_matrix: np.ndarray) -> int:
    """
    From a cluster's indices, pick the medoid (the item with the smallest
    average distance to all others in the cluster). Change this logic if
    you'd rather keep e.g. the longest text, or the first-seen item.
    """
    if len(indices) == 1:
        return indices[0]
    sub = dist_matrix[np.ix_(indices, indices)]
    avg_dist = sub.mean(axis=1)
    best_local = int(np.argmin(avg_dist))
    return indices[best_local]


def dedup_prompt_dicts(
    items: list[dict[str, Any]],
    text_key: str = "generated_prompt",
    distance_threshold: float = 0.15,
    linkage: str = "average",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Main entry point.

    Args:
        items: list of dicts, each containing at least `text_key`.
        text_key: dict key holding the text to compare.
        distance_threshold: max normalized Levenshtein distance within a
            cluster (0..1). Lower = stricter.
        linkage: "average" or "complete" (avoid "single", causes chaining).

    Returns:
        kept: list of original dicts (all keys preserved) that survived
            dedup — one per cluster.
        removed_report: list of dicts, one per removed item, each with:
            - "removed_item": the original dict that was dropped
            - "kept_item": the original dict that was kept in its place
            - "distance": normalized Levenshtein distance between them
            - "similarity": 1 - distance, for convenience
    """
    if not items:
        return [], []

    texts = [item[text_key] for item in items]
    dist_matrix = build_distance_matrix(texts)
    labels = cluster_texts(dist_matrix, distance_threshold, linkage)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    kept: list[dict[str, Any]] = []
    removed_report: list[dict[str, Any]] = []

    for label, indices in clusters.items():
        rep_idx = pick_representative(indices, dist_matrix)
        kept.append(items[rep_idx])

        for idx in indices:
            if idx == rep_idx:
                continue
            dist = dist_matrix[idx, rep_idx]
            removed_report.append(
                {
                    "removed_item": items[idx],
                    "kept_item": items[rep_idx],
                    "distance": float(dist),
                    "similarity": float(1 - dist),
                }
            )

    # Sort removed report by similarity descending, for easier eyeballing
    removed_report.sort(key=lambda r: r["similarity"], reverse=True)

    return kept, removed_report


if __name__ == "__main__":
    import json
    items = json.load(
        open('../../data/prompts/google-gemma-4-26B-A4B-it-gen-prompts.json', 'r')
    )
    items += json.load(
        open('../../data/prompts/openai-gpt-oss-120b-gen-prompts.json', 'r')
    )
    items += json.load(
        open('../../data/prompts/Qwen-Qwen3.6-35B-A3B-gen-prompts.json', 'r')
    )

    DISTANCE = 0.1

    deduplicated_items = []
    deduplicated_reports = {}
    # Deduplicate per config
    unique_tasks = set([
        x['task'] for x in items
    ])

    for task in sorted(unique_tasks):
        task_items = [
            x for x in items if x['task'] == task
        ]

        kept, removed_report = dedup_prompt_dicts(
            task_items, text_key="generated_prompt", distance_threshold=DISTANCE
        )

        print(f"Original: {len(task_items)} items")
        print(f"Kept:     {len(kept)} items")
        print(f"Removed:  {len(removed_report)} items\n")

        print("=== Removed items (with most-similar kept item) ===")
        for r in removed_report:
            print(
                f"  REMOVED: {r['removed_item']['generated_prompt']!r} "
                f"(id={r['removed_item'].get('id')})"
            )
            print(
                f"  KEPT AS: {r['kept_item']['generated_prompt']!r} "
                f"(id={r['kept_item'].get('id')})"
            )
            print(f"  similarity={r['similarity']:.3f}  distance={r['distance']:.3f}\n")

        print("=== Final kept list (all original keys preserved) ===")
        for k in kept:
            print(f"  {k}")

        deduplicated_items += kept
        deduplicated_reports[task] = removed_report

    with open(
        '../../data/prompts/deduplicated_prompts.json', 'w'
    ) as f:
        json.dump(deduplicated_items, f)
    with open(
        '../../data/prompts/deduplicated_reports.json', 'w'
    ) as f:
        json.dump(deduplicated_reports, f)