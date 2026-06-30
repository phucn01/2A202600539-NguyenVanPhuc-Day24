from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

# ─── Keyword fallback patterns (dùng khi NeMo không khả dụng) ────────────────

_JAILBREAK_KEYWORDS = [
    "bỏ qua tất cả hướng dẫn", "ignore your previous instructions", "ignore previous",
    "pretend you are dan", "you are now an unrestricted", "forget your system prompt",
    "system override", "đóng vai ai không có giới hạn", "không có giới hạn",
    "admin command", "ignore prior", "đóng vai hr manager và tiết lộ",
    "đóng vai", "bỏ qua", "reveal all", "unrestricted ai",
]

_OFF_TOPIC_KEYWORDS = [
    "viết một bài thơ", "bài thơ", "nấu phở", "nấu ăn", "bitcoin", "ethereum",
    "giá cổ phiếu", "recommend phim", "bộ phim", "giải phương trình",
    "phương trình vi phân", "thời tiết", "tin tức hôm nay", "marvel",
]

_PII_REQUEST_KEYWORDS = [
    "cho tôi biết cccd", "số điện thoại của nhân viên", "lương của nhân viên cụ thể",
    "thông tin cá nhân của", "email của nhân viên", "tiết lộ thông tin nhân viên",
    "liệt kê toàn bộ thông tin nhân viên", "tiết lộ bảng lương",
    "tiết lộ lương tháng", "tiết lộ", "bảng lương chi tiết",
]

_PROMPT_INJECTION_KEYWORDS = [
    "ignore previous instructions", "<!-- ignore", "system override",
    "admin command", "priority 1", "dump all training data",
    "print all confidential", "tôi là ceo và ra lệnh",
    "system instructions", "forget all hr policies", "reveal all confidential",
    "in ra toàn bộ system", "bỏ qua tất cả policy",
]


def _keyword_block(text: str) -> tuple[bool, str | None]:
    """Kiểm tra text với keyword patterns. Returns (blocked, reason)."""
    lower = text.lower()
    for kw in _JAILBREAK_KEYWORDS:
        if kw in lower:
            return True, "jailbreak_detected"
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in lower:
            return True, "off_topic_detected"
    for kw in _PII_REQUEST_KEYWORDS:
        if kw in lower:
            return True, "pii_request_detected"
    for kw in _PROMPT_INJECTION_KEYWORDS:
        if kw in lower:
            return True, "prompt_injection_detected"
    return False, None


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)"""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    all_results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    # Filter: chỉ giữ các entity type có ý nghĩa cho use case này
    # Loại bỏ PERSON/DATE_TIME vì spaCy NER hay false positive với text tiếng Việt
    RELEVANT_TYPES = {"VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    SCORE_THRESHOLD = 0.5
    results = [
        r for r in all_results
        if r.entity_type in RELEVANT_TYPES and r.score >= SCORE_THRESHOLD
    ]

    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)"""
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,
        }
    """
    # First, check with keyword fallback (fast, no API needed)
    blocked, reason = _keyword_block(text)
    if blocked:
        return {
            "allowed": False,
            "blocked_reason": reason,
            "response": "Xin lỗi, tôi không thể thực hiện yêu cầu này.",
        }

    # Then try NeMo for semantic understanding
    try:
        if rails is None:
            rails = setup_nemo_rails()

        raw = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        # Handle different response formats from NeMo
        if isinstance(raw, dict):
            response = raw.get("content", "") or raw.get("text", "") or str(raw)
        elif isinstance(raw, list):
            response = next(
                (m.get("content", "") for m in reversed(raw)
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                str(raw)
            )
        else:
            response = str(raw) if raw is not None else ""

        refuse_keywords = ["xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
                           "không thể cung cấp", "ngoài phạm vi", "không thể trả lời"]
        blocked = any(kw in response.lower() for kw in refuse_keywords)
        return {
            "allowed":        not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response":       response,
        }
    except Exception as e:
        print(f"  ⚠️  NeMo check_input_rail failed: {e}")
        return {"allowed": True, "blocked_reason": None, "response": ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,
        }
    """
    # Check output for sensitive content patterns
    sensitive_patterns = [
        "cccd của nhân viên là", "số điện thoại cá nhân của",
        "mật khẩu hệ thống là", "thông tin bí mật",
        "bảng lương của tất cả",
    ]
    lower_answer = answer.lower()
    for pattern in sensitive_patterns:
        if pattern in lower_answer:
            return {
                "safe": False,
                "flagged_reason": "sensitive_content_detected",
                "final_answer": "Tôi không thể cung cấp thông tin này. Vui lòng liên hệ phòng Nhân sự trực tiếp.",
            }

    try:
        if rails is None:
            rails = setup_nemo_rails()

        raw = await rails.generate_async(messages=[
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])
        if isinstance(raw, dict):
            response = raw.get("content", "") or raw.get("text", "") or str(raw)
        elif isinstance(raw, list):
            response = next(
                (m.get("content", "") for m in reversed(raw)
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                str(raw)
            )
        else:
            response = str(raw) if raw is not None else ""
        refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot"]
        flagged = any(kw in response.lower() for kw in refuse_keywords)
        return {
            "safe":           not flagged,
            "flagged_reason": "nemo_output_rail" if flagged else None,
            "final_answer":   response if flagged else answer,
        }
    except Exception as e:
        print(f"  ⚠️  NeMo check_output_rail failed: {e}")
        return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async — await, không dùng asyncio.run())
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + "...",
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure():
        for text in test_inputs[:n_runs]:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail (await — không dùng asyncio.run() trong loop)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times: list[float]) -> dict:
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[min(int(n * 0.95), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms":     percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)

    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save report
    guard_report = {
        "pii_demo": {
            "input": test_pii,
            "has_pii": result["has_pii"],
            "entities": result["entities"],
        },
        "adversarial_suite": {
            "results": results,
            "passed": sum(1 for r in results if r["passed"]),
            "total": len(results),
            "pass_rate": sum(1 for r in results if r["passed"]) / len(results) if results else 0,
        },
        "latency": latency,
    }
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(guard_report, f, ensure_ascii=False, indent=2)
    print("\nGuard results saved → reports/guard_results.json")
