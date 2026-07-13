"""LLM-as-Judge — scores conversation quality on Accuracy, Completeness, Conciseness, Tone.

Uses the same model (or a stronger one) to evaluate completed conversations.
Scores 1-5 on each dimension and provides a brief explanation.
"""

from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.eval.runner import EvalResult
from src.logging import get_logger

logger = get_logger("judge")


JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator of AI customer service conversations.

Score each conversation on 4 dimensions, each rated 1 (worst) to 5 (best):

1. **accuracy** — Does the response match known facts? No hallucinations or
   made-up information? If the agent cites order/tracking/refund data, is it
   factually grounded? Non-existent orders should be correctly flagged as
   not found.

2. **completeness** — Did the agent fully address the customer's question?
   No loose ends left hanging? If the agent couldn't help, did it clearly
   explain why and offer next steps?

3. **conciseness** — No unnecessary repetition, verbosity, or filler?
   Does the agent get to the point efficiently?

4. **tone** — Is the agent professional, polite, and appropriately
   empathetic? When delivering bad news (refund denied, order not found),
   does it sound human and helpful, not robotic or cold?

## Output Format
Reply with EXACTLY this JSON and nothing else:
{
  "accuracy": <int 1-5>,
  "completeness": <int 1-5>,
  "conciseness": <int 1-5>,
  "tone": <int 1-5>,
  "overall": <int 1-5>,
  "notes": "<one sentence summarizing the evaluation>"
}
"""


class Judge:
    """LLM-as-Judge for evaluating conversation quality.

    Usage::

        judge = Judge(model)
        scores = await judge.evaluate(result)
        # result.judge_scores is now populated
    """

    def __init__(self, model: BaseChatModel):
        self.model = model

    async def evaluate(self, result: EvalResult) -> dict:
        """Score a single conversation result.

        Returns a dict with accuracy, completeness, conciseness, tone, overall, notes.
        Also mutates ``result.judge_scores``.
        """
        if not result.conversation:
            result.judge_scores = {
                "accuracy": 0, "completeness": 0, "conciseness": 0,
                "tone": 0, "overall": 0,
                "notes": "No conversation to evaluate.",
            }
            return result.judge_scores

        # Build evaluation prompt
        conv_text = self._format_conversation(result)
        expected_text = self._build_expected(result)

        eval_messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Expected behavior: {expected_text}\n\n"
                f"Conversation to evaluate:\n{conv_text}"
            ),
        ]

        try:
            response = await self.model.ainvoke(eval_messages)
            response_text = str(response.content) if hasattr(response, "content") else str(response)
            scores = self._parse_scores(response_text)
        except Exception as exc:
            scores = {
                "accuracy": 0, "completeness": 0, "conciseness": 0,
                "tone": 0, "overall": 0,
                "notes": f"Judge evaluation failed: {exc}",
            }

        result.judge_scores = scores
        return scores

    async def evaluate_all(self, results: list[EvalResult]) -> list[EvalResult]:
        """Score all results. Returns the same list (mutated in-place)."""
        for i, result in enumerate(results):
            logger.info("[judge] %s ...", result.scenario_id)
            await self.evaluate(result)
            overall = result.judge_scores.get("overall", "?") if result.judge_scores else "?"
            logger.info("[judge] %s overall=%s", result.scenario_id, overall)
        return results

    # ── Helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _format_conversation(result: EvalResult) -> str:
        """Render conversation as readable text for the judge."""
        lines = []
        for turn in result.conversation:
            role = turn["role"].upper()
            content = turn["content"]
            if len(content) > 800:
                content = content[:800] + "..."
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_expected(result: EvalResult) -> str:
        """Short description of what the scenario expected."""
        parts = [
            f"Scenario: {result.title}",
            f"Category: {result.category}",
            f"Expected agent: {result.actual_agent}",
        ]
        if result.tools_called:
            parts.append(f"Tools called: {', '.join(result.tools_called)}")
        return " | ".join(parts)

    @staticmethod
    def _parse_scores(text: str) -> dict:
        """Parse the judge's JSON response, with fallback."""
        # Try to extract JSON block
        json_match = re.search(r'\{[^{}]*"accuracy"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return {
                    "accuracy": int(data.get("accuracy", 0)),
                    "completeness": int(data.get("completeness", 0)),
                    "conciseness": int(data.get("conciseness", 0)),
                    "tone": int(data.get("tone", 0)),
                    "overall": int(data.get("overall", 0)),
                    "notes": data.get("notes", ""),
                }
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: return zeros
        return {
            "accuracy": 0, "completeness": 0, "conciseness": 0,
            "tone": 0, "overall": 0,
            "notes": f"Failed to parse judge response: {text[:100]}",
        }
