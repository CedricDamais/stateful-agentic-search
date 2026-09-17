from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .state import Document


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


class RetrievalTools:
    """Small deterministic retriever used by Experiment 0 and tests."""

    def __init__(self, corpus: list[Document]) -> None:
        self.corpus = corpus
        self._doc_freq = Counter(word for doc in corpus for word in set(tokenize(doc.title + " " + doc.text)))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RetrievalTools":
        import json

        corpus = [Document(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls(corpus)

    def bm25(self, query: str, k: int = 5) -> list[Document]:
        query_terms = tokenize(query)
        n = len(self.corpus)
        ranked = []
        for doc in self.corpus:
            terms = tokenize(doc.title + " " + doc.text)
            tf = Counter(terms)
            length_norm = 0.75 + 0.25 * len(terms) / max(1, sum(len(tokenize(d.text)) for d in self.corpus) / n)
            score = sum((math.log((n - self._doc_freq[t] + 0.5) / (self._doc_freq[t] + 0.5) + 1) * tf[t] / length_norm) for t in query_terms)
            ranked.append(Document(doc.id, doc.title, doc.text, score))
        return sorted(ranked, key=lambda d: d.score, reverse=True)[:k]

    def vector(self, query: str, k: int = 5) -> list[Document]:
        # Hashing vectors make this portable; swap this with embeddings in later experiments.
        def vectorize(text: str) -> Counter[int]: return Counter(hash(t) % 257 for t in tokenize(text))
        q = vectorize(query)
        ranked = []
        for doc in self.corpus:
            d = vectorize(doc.title + " " + doc.text)
            dot = sum(q[x] * d[x] for x in q)
            norm = math.sqrt(sum(v * v for v in q.values()) * sum(v * v for v in d.values()))
            ranked.append(Document(doc.id, doc.title, doc.text, dot / norm if norm else 0.0))
        return sorted(ranked, key=lambda d: d.score, reverse=True)[:k]

    def hybrid(self, query: str, k: int = 5) -> list[Document]:
        merged: dict[str, Document] = {}
        for rank, doc in enumerate(self.bm25(query, k=len(self.corpus)), start=1):
            merged[doc.id] = Document(doc.id, doc.title, doc.text, 1 / (60 + rank))
        for rank, doc in enumerate(self.vector(query, k=len(self.corpus)), start=1):
            prior = merged[doc.id]
            prior.score += 1 / (60 + rank)
        return sorted(merged.values(), key=lambda d: d.score, reverse=True)[:k]

    def rerank(self, query: str, documents: list[Document], k: int = 5) -> list[Document]:
        terms = set(tokenize(query))
        rescored = [Document(d.id, d.title, d.text, d.score + 0.2 * len(terms & set(tokenize(d.title)))) for d in documents]
        return sorted(rescored, key=lambda d: d.score, reverse=True)[:k]

    def rewrite(self, query: str) -> str:
        return " ".join(tokenize(query))
