from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""

    if not OPENAI_API_KEY:
        return _heuristic_judge(question, answer_a, answer_b)

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user",   "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        winner = data.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        return {
            "winner": winner,
            "reasoning": data.get("reasoning", ""),
            "scores": data.get("scores", {"A": 0.5, "B": 0.5}),
        }
    except Exception as e:
        print(f"  ⚠️  pairwise_judge API failed: {e}")
        return _heuristic_judge(question, answer_a, answer_b)


def _heuristic_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Fallback heuristic judge dựa trên keyword overlap và độ dài."""
    import re

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"[\wÀ-ỹ]+", (text or "").lower()))

    q_tokens = tokenize(question)
    a_tokens = tokenize(answer_a)
    b_tokens = tokenize(answer_b)

    score_a = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    score_b = len(q_tokens & b_tokens) / max(len(q_tokens), 1)

    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Hai câu trả lời có chất lượng tương đương."
    elif score_a > score_b:
        winner = "A"
        reasoning = f"Answer A có độ liên quan cao hơn với câu hỏi (score {score_a:.2f} vs {score_b:.2f})."
    else:
        winner = "B"
        reasoning = f"Answer B có độ liên quan cao hơn với câu hỏi (score {score_b:.2f} vs {score_a:.2f})."

    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"].get("B", 0.0), "B": pass2_raw["scores"].get("A", 0.0)},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    if not judge_labels or not human_labels:
        return 0.0

    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    p_e = (judge_labels.count(1) / n * human_labels.count(1) / n +
           judge_labels.count(0) / n * human_labels.count(0) / n)

    if p_e == 1:
        return 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,
          "position_bias_count": int,
          "verbosity_bias": float,
          "verbosity_details": {...},
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
                "position_bias_count": 0, "verbosity_details": {}, "interpretation": ""}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # Run judge on the same 10 questions to get judge_labels
    judge_labels_list = []
    for item in human_data:
        ground_truth_answer = item["model_answer"]
        bad_answer = "Không tìm thấy thông tin phù hợp."
        r = pairwise_judge(item["question"], ground_truth_answer, bad_answer)
        judge_labels_list.append(1 if r["winner"] == "A" else 0)

    kappa = cohen_kappa(judge_labels_list, human_labels)
    print(f"Cohen's κ: {kappa:.3f}")

    # --- Bias report ---
    bias = bias_report([result])
    print(f"\nBias report: {bias}")

    # --- Save judge results ---
    import os
    os.makedirs("reports", exist_ok=True)
    report = {
        "demo_result": {
            "question": result.question,
            "winner_pass1": result.winner_pass1,
            "winner_pass2": result.winner_pass2,
            "final_winner": result.final_winner,
            "position_consistent": result.position_consistent,
            "reasoning_pass1": result.reasoning_pass1,
        },
        "cohen_kappa": kappa,
        "judge_labels": judge_labels_list,
        "human_labels": human_labels,
        "bias_report": bias,
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nJudge results saved → reports/judge_results.json")
