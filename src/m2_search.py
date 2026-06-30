from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
from collections import Counter
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return re.sub(r"\s+", " ", segmented.replace("_", " ")).strip()
    except Exception:
        # Fallback: keep the text usable even if the tokenizer is unavailable.
        return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks or []
        self.corpus_tokens = []
        self.bm25 = None

        if not self.documents:
            return

        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in self.documents
        ]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            chunk = self.documents[i]
            results.append(
                SearchResult(
                    text=chunk.get("text", ""),
                    score=float(scores[i]),
                    metadata=chunk.get("metadata", {}),
                    method="bm25",
                )
            )
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None
        self._local_chunks: list[dict] = []
        self._local_vectors: list[Counter[str]] = []
        self._use_local_fallback = False

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        return self._encoder

    def _local_tokenize(self, text: str) -> list[str]:
        segmented = segment_vietnamese(text).lower()
        return re.findall(r"[\wÀ-ỹ]+", segmented)

    def _local_vector(self, text: str) -> Counter[str]:
        return Counter(self._local_tokenize(text))

    @staticmethod
    def _cosine_from_counters(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in shared)
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        return dot / (norm_a * norm_b + 1e-9)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._local_chunks = chunks or []
        self._local_vectors = [self._local_vector(c.get("text", "")) for c in self._local_chunks]

        if not self._local_chunks:
            self._use_local_fallback = True
            return

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            texts = [c["text"] for c in self._local_chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            points = []
            for i, (chunk, vector) in enumerate(zip(self._local_chunks, vectors)):
                payload = {**chunk.get("metadata", {}), "text": chunk.get("text", "")}
                points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))
            self.client.upsert(collection_name=collection, points=points)
            self._use_local_fallback = False
        except Exception:
            # Fall back to local cosine search if Qdrant/model loading is unavailable.
            self._use_local_fallback = True

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self._local_chunks:
            return []

        if not self._use_local_fallback:
            try:
                query_vector = self._get_encoder().encode(query).tolist()
                response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
                points = getattr(response, "points", response)
                results: list[SearchResult] = []
                for pt in points:
                    payload = getattr(pt, "payload", {}) or {}
                    text = payload.get("text", "")
                    results.append(
                        SearchResult(
                            text=text,
                            score=float(getattr(pt, "score", 0.0)),
                            metadata=payload,
                            method="dense",
                        )
                    )
                if results:
                    return results
            except Exception:
                self._use_local_fallback = True

        query_vec = self._local_vector(query)
        scored = [
            (
                self._cosine_from_counters(query_vec, doc_vec),
                idx,
            )
            for idx, doc_vec in enumerate(self._local_vectors)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchResult] = []
        for score, idx in scored[:top_k]:
            if score <= 0:
                continue
            chunk = self._local_chunks[idx]
            results.append(
                SearchResult(
                    text=chunk.get("text", ""),
                    score=float(score),
                    metadata={**chunk.get("metadata", {}), "text": chunk.get("text", "")},
                    method="dense",
                )
            )
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict[str, object]] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] = float(rrf_scores[result.text]["score"]) + 1.0 / (k + rank + 1)

    merged = sorted(
        rrf_scores.values(),
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:top_k]

    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in merged
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
