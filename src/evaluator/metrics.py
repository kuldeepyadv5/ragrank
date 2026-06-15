"""LLM-as-a-judge metrics for RAG quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.generator.llm_client import LLMClient


@dataclass
class EvalResult:
    context_precision: float
    answer_faithfulness: float
    answer_relevance: float
    details: dict = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        return (
            self.context_precision
            + self.answer_faithfulness
            + self.answer_relevance
        ) / 3


class RAGEvaluator:
    """Evaluate RAG outputs using structured LLM-as-a-judge prompts."""

    SYSTEM_PROMPT = (
        "You are an expert RAG evaluation judge. "
        "Score each metric from 0.0 to 1.0 with concise reasoning."
    )

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: Sequence[str],
        ground_truth: str | None = None,
    ) -> EvalResult:
        context_block = "\n---\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )

        prompt = f"""Evaluate this RAG response.

Question: {query}

Retrieved Context:
{context_block}

Generated Answer:
{answer}

{f'Ground Truth (optional): {ground_truth}' if ground_truth else ''}

Return JSON with this exact schema:
{{
  "context_precision": <float 0-1>,
  "answer_faithfulness": <float 0-1>,
  "answer_relevance": <float 0-1>,
  "reasoning": {{
    "context_precision": "<short explanation>",
    "answer_faithfulness": "<short explanation>",
    "answer_relevance": "<short explanation>"
  }}
}}

Scoring guide:
- context_precision: fraction of retrieved chunks that contain information useful for answering the question
- answer_faithfulness: are all claims in the answer supported by the retrieved context (no hallucinations)?
- answer_relevance: does the answer directly address the user's question?
"""
        if self.llm.provider == "mock":
            return self._mock_eval(query, answer, contexts)

        try:
            data = self.llm.generate_json(prompt, system=self.SYSTEM_PROMPT)
            return EvalResult(
                context_precision=float(data.get("context_precision", 0.5)),
                answer_faithfulness=float(data.get("answer_faithfulness", 0.5)),
                answer_relevance=float(data.get("answer_relevance", 0.5)),
                details=data.get("reasoning", {}),
            )
        except Exception as exc:
            return EvalResult(
                context_precision=0.0,
                answer_faithfulness=0.0,
                answer_relevance=0.0,
                details={"error": str(exc)},
            )

    def evaluate_batch(
        self,
        runs: Sequence[dict],
    ) -> dict:
        """Evaluate multiple query runs and aggregate metrics."""
        results: list[EvalResult] = []
        for run in runs:
            result = self.evaluate(
                query=run["query"],
                answer=run["answer"],
                contexts=run.get("contexts", []),
                ground_truth=run.get("ground_truth"),
            )
            results.append(result)

        if not results:
            return {
                "count": 0,
                "context_precision": 0.0,
                "answer_faithfulness": 0.0,
                "answer_relevance": 0.0,
                "average_score": 0.0,
                "runs": [],
            }

        n = len(results)
        return {
            "count": n,
            "context_precision": sum(r.context_precision for r in results) / n,
            "answer_faithfulness": sum(r.answer_faithfulness for r in results) / n,
            "answer_relevance": sum(r.answer_relevance for r in results) / n,
            "average_score": sum(r.average_score for r in results) / n,
            "runs": [
                {
                    "context_precision": r.context_precision,
                    "answer_faithfulness": r.answer_faithfulness,
                    "answer_relevance": r.answer_relevance,
                    "details": r.details,
                }
                for r in results
            ],
        }

    @staticmethod
    def _mock_eval(query: str, answer: str, contexts: Sequence[str]) -> EvalResult:
        has_context = len(contexts) > 0
        overlap = any(word.lower() in " ".join(contexts).lower() for word in query.split()[:3])
        return EvalResult(
            context_precision=0.8 if has_context and overlap else 0.5,
            answer_faithfulness=0.75 if has_context else 0.3,
            answer_relevance=0.85 if answer else 0.0,
            details={"mode": "mock_heuristic"},
        )
