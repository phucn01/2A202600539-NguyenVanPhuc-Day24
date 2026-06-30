from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENABLE_ENRICHMENT_API = os.getenv("ENABLE_ENRICHMENT_API", "0") == "1"


def _should_use_enrichment_api() -> bool:
    """Only call OpenAI when enrichment is explicitly enabled."""
    return ENABLE_ENRICHMENT_API and bool(OPENAI_API_KEY)


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    text = (text or "").strip()
    if not text:
        return ""

    if _should_use_enrichment_api():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            summary = resp.choices[0].message.content.strip()
            if summary:
                return summary
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text[:200]
    summary = " ".join(sentences[:2]).strip()
    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    text = (text or "").strip()
    if not text:
        return []

    if _should_use_enrichment_api():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            cleaned = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
            if cleaned:
                return cleaned[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    parts = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if not parts:
        parts = [text[:80].strip() or "đoạn văn này"]

    questions: list[str] = []
    for part in parts[:n_questions]:
        stem = part.rstrip(".!?")
        questions.append(f"Thông tin nào được nêu về {stem.lower()}?")

    return questions or [f"Đoạn văn này nói về gì?"]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    text = (text or "").strip()
    if not text:
        return ""

    if _should_use_enrichment_api():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    text = (text or "").strip()
    if not text:
        return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}

    if _should_use_enrichment_api():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return {
                    "topic": data.get("topic", "general"),
                    "entities": data.get("entities", []) or [],
                    "category": data.get("category", "policy"),
                    "language": data.get("language", "vi"),
                }
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    lowered = text.lower()
    category = "policy"
    if any(k in lowered for k in ["mật khẩu", "vpn", "it", "wireguard", "aes"]):
        category = "it"
    elif any(k in lowered for k in ["lương", "bhxh", "thử việc", "nhân viên", "nghỉ phép", "nghỉ ốm"]):
        category = "hr"
    elif any(k in lowered for k in ["doanh thu", "chi phí", "hóa đơn", "tài chính"]):
        category = "finance"

    entities = re.findall(r"\b[A-ZÀ-Ỹ][\wÀ-ỹ-]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ-]*)*", text)
    topic = "general"
    if "nghỉ phép" in lowered:
        topic = "leave policy"
    elif "mật khẩu" in lowered:
        topic = "password policy"
    elif "thử việc" in lowered:
        topic = "probation"

    return {
        "topic": topic,
        "entities": entities[:5],
        "category": category,
        "language": "vi",
    }


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    text = (text or "").strip()
    if not text:
        return {
            "summary": "",
            "questions": [],
            "context": "",
            "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
        }

    if _should_use_enrichment_api():
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                return {
                    "summary": data.get("summary", ""),
                    "questions": data.get("questions", []) or [],
                    "context": data.get("context", ""),
                    "metadata": data.get("metadata", {}) or {},
                }
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ {source}. " if source else "",
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
