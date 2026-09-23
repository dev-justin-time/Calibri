# SPDX-License-Identifier: MIT
# Doberwatch Finance — deterministic text similarity.
#
# Retrieval ("the ten closest responses before a model call") and anti-drift
# checks must give the *same* answer twice, on any machine, with no network
# and no model. That rules out embeddings served from a model and points at
# the classic hashing trick: a signed bag-of-words vector hashed into a fixed
# dimension, L2-normalized, compared by cosine. It is coarse on purpose —
# the watchdog grades quality with explicit rubrics, not with this score.

from __future__ import annotations

import hashlib
import re
from typing import Any, List, Sequence, Tuple

import numpy as np

DEFAULT_DIM = 256

_WORD = re.compile(r"[a-z0-9]+")

# Function words carry no discriminating signal for request matching and
# would otherwise dominate short requests.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "please", "so", "that", "the", "their", "them", "then", "there",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
})


def stem(token: str) -> str:
    """Suffix normalization so inflections of one word share a token.

    ``reversed`` / ``reversal`` / ``reversing`` / ``reverse`` all collapse to
    one token, as do ``charges``/``charge`` and ``policies``/``policy``. This
    is morphology only and deliberately crude. It does **not** handle synonyms:
    ``reversed`` and ``refunded`` stay different tokens, because deciding those
    mean the same thing is a claim about the world, not about spelling — and a
    matcher that pretends otherwise scores better than it checks.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in ("ed", "ing", "als", "al", "ions", "ion", "ers", "er"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[:-len(suffix)]
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokens, minus stopwords and one-letter noise.

    Stopwords are removed *before* stemming, so the filter still sees the word
    as written; what comes out is stemmed.
    """
    return [stem(t) for t in _WORD.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


def hash_embedding(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Signed hashing bag-of-words, L2-normalized (zero vector if empty)."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:8], "big")
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-comparable vectors, in [-1, 1]."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity(text_a: str, text_b: str, dim: int = DEFAULT_DIM) -> float:
    """Cosine similarity of two raw texts, in [-1, 1] (0.0 if either is empty)."""
    return cosine(hash_embedding(text_a, dim), hash_embedding(text_b, dim))


def top_k(query: str, candidates: Sequence[Tuple[str, Any]], k: int = 10,
          dim: int = DEFAULT_DIM) -> List[Tuple[float, Any]]:
    """Rank ``(text, payload)`` candidates by closeness to ``query``.

    Deterministic: ties break toward the candidate's original index, so the
    same inputs always produce the same "closest ten" in the same order.
    """
    if k <= 0:
        return []
    q = hash_embedding(query, dim)
    scored = [(cosine(q, hash_embedding(text, dim)), i, payload)
              for i, (text, payload) in enumerate(candidates)]
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [(score, payload) for score, _i, payload in scored[:k]]


__all__ = ["DEFAULT_DIM", "STOPWORDS", "stem", "tokenize", "hash_embedding",
           "cosine", "similarity", "top_k"]
