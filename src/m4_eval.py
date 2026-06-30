from __future__ import annotations

"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import os
import sys
import json
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", (text or "").lower()))


def _overlap_score(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)


def _flatten_contexts(contexts: list[str]) -> str:
    return " ".join(c for c in contexts if c)


def _heuristic_metrics(question: str, answer: str, contexts: list[str], ground_truth: str) -> tuple[float, float, float, float]:
    context_text = _flatten_contexts(contexts)
    answer = answer or ""
    ground_truth = ground_truth or ""

    faithfulness = _overlap_score(answer, context_text)
    answer_relevancy = _overlap_score(question, answer)
    context_precision = _overlap_score(answer or question, context_text)
    context_recall = _overlap_score(ground_truth, context_text)

    if _tokenize(ground_truth) & _tokenize(context_text):
        context_recall = min(1.0, context_recall + 0.1)
    if _tokenize(question) & _tokenize(answer):
        answer_relevancy = min(1.0, answer_relevancy + 0.1)
    if _tokenize(answer) & _tokenize(context_text):
        faithfulness = min(1.0, faithfulness + 0.1)

    clamp = lambda x: round(float(max(0.0, min(1.0, x))), 4)
    return clamp(faithfulness), clamp(answer_relevancy), clamp(context_precision), clamp(context_recall)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if os.getenv("ENABLE_RAGAS", "0") != "1" or not OPENAI_API_KEY:
        return _evaluate_heuristic(questions, answers, contexts, ground_truths)

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()

        per_question: list[EvalResult] = []
        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    contexts=row.get("contexts", []) or [],
                    ground_truth=row.get("ground_truth", ""),
                    faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                    context_precision=float(row.get("context_precision", 0.0) or 0.0),
                    context_recall=float(row.get("context_recall", 0.0) or 0.0),
                )
            )

        n = max(len(per_question), 1)
        return {
            "faithfulness": round(sum(r.faithfulness for r in per_question) / n, 4),
            "answer_relevancy": round(sum(r.answer_relevancy for r in per_question) / n, 4),
            "context_precision": round(sum(r.context_precision for r in per_question) / n, 4),
            "context_recall": round(sum(r.context_recall for r in per_question) / n, 4),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")

    return _evaluate_heuristic(questions, answers, contexts, ground_truths)


def _evaluate_heuristic(questions: list[str], answers: list[str],
                        contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question: list[EvalResult] = []
    for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
        faithfulness, answer_relevancy, context_precision, context_recall = _heuristic_metrics(q, a, c, gt)
        per_question.append(
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
            )
        )

    n = max(len(per_question), 1)
    return {
        "faithfulness": round(sum(r.faithfulness for r in per_question) / n, 4),
        "answer_relevancy": round(sum(r.answer_relevancy for r in per_question) / n, 4),
        "context_precision": round(sum(r.context_precision for r in per_question) / n, 4),
        "context_recall": round(sum(r.context_recall for r in per_question) / n, 4),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored: list[dict] = []
    for item in eval_results:
        metrics = {
            "faithfulness": float(item.faithfulness),
            "answer_relevancy": float(item.answer_relevancy),
            "context_precision": float(item.context_precision),
            "context_recall": float(item.context_recall),
        }
        worst_metric = min(metrics, key=metrics.get)
        avg_score = sum(metrics.values()) / len(metrics)
        diagnosis, suggested_fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Inspect pipeline"))
        scored.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored.sort(key=lambda row: row["score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
