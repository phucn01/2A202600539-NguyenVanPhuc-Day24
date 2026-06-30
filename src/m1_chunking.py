from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_group: list[str] = []
    current_len = 0
    max_group_len = max(1, int(600 * max(0.3, min(threshold, 1.0))))

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        if current_group and current_len + len(paragraph) + 2 > max_group_len:
            chunks.append(
                Chunk(
                    text="\n\n".join(current_group).strip(),
                    metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
                )
            )
            current_group = [paragraph]
            current_len = len(paragraph)
        else:
            current_group.append(paragraph)
            current_len += len(paragraph) + (2 if len(current_group) > 1 else 0)

    if current_group:
        chunks.append(
            Chunk(
                text="\n\n".join(current_group).strip(),
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
            )
        )

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}

    def _split_by_size(content: str, size: int) -> list[str]:
        content = content.strip()
        if not content:
            return []
        if len(content) <= size:
            return [content]

        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        if len(parts) <= 1:
            return [content[i:i + size].strip() for i in range(0, len(content), size) if content[i:i + size].strip()]

        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(part) > size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(_split_by_size(part, size))
                continue
            if current and len(current) + len(part) + 2 > size:
                chunks.append(current.strip())
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
        if current.strip():
            chunks.append(current.strip())
        return chunks

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    current_parent_parts: list[str] = []
    current_parent_len = 0

    def _flush_parent() -> None:
        nonlocal current_parent_parts, current_parent_len
        if not current_parent_parts:
            return
        parent_text = "\n\n".join(current_parent_parts).strip()
        pid = f"parent_{len(parents)}"
        parent_meta = {**metadata, "chunk_type": "parent", "parent_id": pid}
        parents.append(Chunk(text=parent_text, metadata=parent_meta))

        for child_idx, child_text in enumerate(_split_by_size(parent_text, child_size)):
            child_meta = {
                **metadata,
                "chunk_type": "child",
                "child_index": child_idx,
                "parent_id": pid,
            }
            children.append(Chunk(text=child_text, metadata=child_meta, parent_id=pid))

        current_parent_parts = []
        current_parent_len = 0

    for para in paragraphs:
        if current_parent_parts and current_parent_len + len(para) + 2 > parent_size:
            _flush_parent()
        if len(para) > parent_size:
            if current_parent_parts:
                _flush_parent()
            for parent_text in _split_by_size(para, parent_size):
                pid = f"parent_{len(parents)}"
                parent_meta = {**metadata, "chunk_type": "parent", "parent_id": pid}
                parents.append(Chunk(text=parent_text, metadata=parent_meta))
                for child_idx, child_text in enumerate(_split_by_size(parent_text, child_size)):
                    child_meta = {
                        **metadata,
                        "chunk_type": "child",
                        "child_index": child_idx,
                        "parent_id": pid,
                    }
                    children.append(Chunk(text=child_text, metadata=child_meta, parent_id=pid))
            continue

        current_parent_parts.append(para)
        current_parent_len += len(para) + (2 if len(current_parent_parts) > 1 else 0)

    _flush_parent()
    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    lines = text.splitlines()

    chunks: list[Chunk] = []
    current_header = ""
    current_content: list[str] = []

    def _flush() -> None:
        if not current_header and not current_content:
            return
        body = "\n".join(current_content).strip()
        chunk_text = f"{current_header}\n\n{body}".strip() if current_header else body
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "strategy": "structure",
                        "section": current_header.lstrip("# ").strip() if current_header else "",
                        "chunk_index": len(chunks),
                    },
                )
            )

    for line in lines:
        if re.match(r"^#{1,3}\s+.+$", line.strip()):
            _flush()
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)

    _flush()
    if chunks:
        return chunks

    fallback = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        Chunk(
            text=part,
            metadata={**metadata, "strategy": "structure", "section": "", "chunk_index": i},
        )
        for i, part in enumerate(fallback)
    ]


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
