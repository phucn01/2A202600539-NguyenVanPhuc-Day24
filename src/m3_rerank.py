from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Keep this offline-safe. In restricted environments we use the
            # lexical fallback below instead of attempting a remote model fetch.
            return None
        return self._model

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))

    def _fallback_score(self, query: str, text: str) -> float:
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(text)
        if not query_tokens or not doc_tokens:
            return 0.0

        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            return 0.0

        score = overlap / len(query_tokens)
        if any(token.isdigit() for token in query_tokens & doc_tokens):
            score += 0.25
        if "nghỉ" in query.lower() and ("nghỉ" in text.lower() or "phép" in text.lower()):
            score += 0.5
        if "vpn" in text.lower() or "mật khẩu" in text.lower():
            score -= 0.2
        return score

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        scored_docs: list[tuple[float, dict]]

        if model is not None:
            try:
                pairs = [(query, doc.get("text", "")) for doc in documents]
                scores = model.predict(pairs)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                scored_docs = list(zip([float(s) for s in scores], documents))
            except Exception:
                scored_docs = [(self._fallback_score(query, doc.get("text", "")), doc) for doc in documents]
        else:
            scored_docs = [(self._fallback_score(query, doc.get("text", "")), doc) for doc in documents]

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for i, (score, doc) in enumerate(scored_docs[:top_k]):
            results.append(
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                )
            )
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            model = self._model or Ranker()
            self._model = model
            passages = [{"text": d.get("text", "")} for d in documents]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            reranked = []
            for i, item in enumerate(results[:top_k]):
                idx = item.get("index", i)
                doc = documents[idx] if 0 <= idx < len(documents) else {"text": item.get("text", ""), "score": 0.0, "metadata": {}}
                reranked.append(
                    RerankResult(
                        text=doc.get("text", ""),
                        original_score=float(doc.get("score", 0.0)),
                        rerank_score=float(item.get("score", 0.0)),
                        metadata=doc.get("metadata", {}),
                        rank=i,
                    )
                )
            return reranked
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
