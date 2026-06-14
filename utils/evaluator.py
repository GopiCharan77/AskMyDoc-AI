from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict

from models.LLM import llm


@dataclass
class EvalResult:
    faithfulness: int
    relevance: int
    context_usage: int
    hallucination_risk: str
    overall_score: int
    needs_retry: bool
    verdict: str
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fallback(reason: str) -> EvalResult:
    return EvalResult(
        faithfulness=40,
        relevance=40,
        context_usage=40,
        hallucination_risk="medium",
        overall_score=40,
        needs_retry=True,
        verdict="review",
        reasoning=reason,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found: {text}")
    return json.loads(match.group(0))


def evaluate_answer(question: str, context: str, answer: str) -> EvalResult:
    prompt = f"""
You are a strict RAG answer evaluator.

Return ONLY valid JSON with these keys:
- faithfulness: integer 0-100
- relevance: integer 0-100
- context_usage: integer 0-100
- hallucination_risk: one of "low", "medium", "high"
- overall_score: integer 0-100
- needs_retry: true or false
- verdict: one of "pass", "review", "fail"
- reasoning: short string

Rules:
- Score ONLY using the provided context.
- Faithfulness means how strongly the answer is supported by the context.
- Relevance means how well the answer addresses the question.
- Context usage means how much of the answer comes from the context.
- needs_retry should be true when faithfulness is low, hallucination risk is medium/high, or the answer is weak.

Question:
{question}

Context:
{context}

Answer:
{answer}
""".strip()

    try:
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = _extract_json(raw)

        return EvalResult(
            faithfulness=int(data["faithfulness"]),
            relevance=int(data["relevance"]),
            context_usage=int(data["context_usage"]),
            hallucination_risk=str(data["hallucination_risk"]).lower().strip(),
            overall_score=int(data["overall_score"]),
            needs_retry=bool(data["needs_retry"]),
            verdict=str(data["verdict"]).lower().strip(),
            reasoning=str(data["reasoning"]).strip(),
        )
    except Exception as exc:
        return _fallback(f"Evaluator parsing failed: {exc}")


def regenerate_answer(question: str, context: str, previous_answer: str, critique: str) -> str:
    prompt = f"""
You are a careful answer rewriter.

Rewrite the answer using ONLY the provided context.
Do not mention the critique.
Do not mention that you are rewriting.
Return only the final answer.

Question:
{question}

Context:
{context}

Previous answer:
{previous_answer}

Critique:
{critique}
""".strip()

    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)
