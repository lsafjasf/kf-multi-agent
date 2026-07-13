"""Supervisor / Intent Router — classifies user requests and routes to specialists.

Uses structured output (``with_structured_output``) as the primary mechanism,
with a regex-based fallback for providers that don't support JSON schema
(e.g., DeepSeek).

Memory integration (L2/L3):
- On first activation for a session, loads L2b (historical session summaries)
  and L3 (user profile) from the DB and injects them into the system prompt.
- On every activation, updates L2a (running summary) in state.
- Specialist sub-agents NEVER see L2/L3 — only the Supervisor does.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.logging import get_logger
from src.retry import with_retries
from src.state import AgentName, CustomerServiceState

if TYPE_CHECKING:
    from src.memory.consolidator import MemoryConsolidator

logger = get_logger("supervisor")


# ── Structured output schema ────────────────────────────────────────
class RoutingDecision(BaseModel):
    """Structured routing output for the supervisor."""
    next: AgentName = Field(description="The agent to route to.")
    reasoning: str = Field(default="", description="One sentence explaining the routing decision.")


SUPERVISOR_BASE_PROMPT = """\
You are the INTAKE ROUTER for ShopFast, a large e-commerce platform.

Your job: read the customer's message and output a JSON routing decision.
Do NOT answer the customer — you only route.

## Available Agents
- order_agent: order status, order details, order history, cancel orders
- logistics_agent: tracking, delivery estimates, lost packages, shipping
- refund_agent: refunds, returns, refund eligibility, "money back"
- human_agent: customer demands a human, is furious, or conversation has broken down

## Routing Rules
1. Mentions an order ID (ORD-XXX) → order_agent
2. Mentions tracking/shipping/delivery/package → logistics_agent
3. Mentions refund/return/money back/broken/defective → refund_agent
4. Demands a human → human_agent
5. Ambiguous → order_agent (default)

## Output Format
Reply with EXACTLY this JSON and nothing else:
{"next": "<agent_name>", "reasoning": "<one sentence why>"}
"""


def _parse_routing_response(text: str) -> tuple[AgentName, str]:
    """Parse the LLM's routing JSON from its response text (fallback parser).

    Handles cases where the model wraps JSON in markdown or adds extra text.
    """
    # Try to extract JSON from the response — match the first {...} with "next"
    json_match = re.search(r'\{[^{}]*"next"\s*:\s*"[^"]*"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            next_agent = data.get("next", "order_agent")
            reasoning = data.get("reasoning", "")
            if next_agent in ("order_agent", "logistics_agent", "refund_agent", "human_agent"):
                return next_agent, reasoning
        except json.JSONDecodeError:
            pass

    # Fallback: keyword-based routing
    text_lower = text.lower()
    if "human" in text_lower or "furious" in text_lower:
        return "human_agent", "Keyword fallback: human requested"
    if any(w in text_lower for w in ("refund", "return", "money back", "broken", "defective")):
        return "refund_agent", "Keyword fallback: refund-related"
    if any(w in text_lower for w in ("tracking", "ship", "delivery", "package", "where is my")):
        return "logistics_agent", "Keyword fallback: logistics-related"
    return "order_agent", "Keyword fallback: default to order"


def build_supervisor_node(
    model: BaseChatModel,
    consolidator: "MemoryConsolidator",
):
    """Build the supervisor routing node with memory injection.

    The consolidator is used for:
    - Loading L2b + L3 memory context at session start
    - Updating L2a running summary on each activation
    """

    # Try to wrap the model with structured output; fall back to regex parser.
    try:
        structured_model = model.with_structured_output(RoutingDecision, method="json_mode")
        logger.debug("Supervisor using structured output (json_mode)")
    except Exception:
        structured_model = None
        logger.debug("Structured output unavailable — falling back to regex parser")

    # Retry-wrapped fallback invoke (used when structured output fails)
    _ainvoke_with_retry = with_retries(max_attempts=2)(model.ainvoke)

    async def supervisor_node(state: CustomerServiceState) -> Command:
        """Route the customer's latest message to the best specialist."""

        # ── Memory: load L2b + L3 context on first activation ─────
        memory_context = state.memory_context
        if not memory_context and state.user_id:
            memory_context = await consolidator.build_memory_context(state.user_id)

        # ── Memory: L2a — update running summary ──────────────────
        running_summary = await consolidator.update_running_summary(state)

        # ── Get the latest user message ────────────────────────────
        user_messages = [
            m for m in state.messages
            if hasattr(m, "type") and m.type == "human"
        ]
        latest_user = user_messages[-1].content if user_messages else ""

        # ── Build the classification prompt ────────────────────────
        system_content = SUPERVISOR_BASE_PROMPT
        if memory_context:
            system_content = memory_context + "\n\n" + SUPERVISOR_BASE_PROMPT

        classification_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Customer message: {latest_user}"),
        ]

        # ── Primary: structured output ───────────────────────────
        if structured_model is not None:
            try:
                decision: RoutingDecision = await structured_model.ainvoke(
                    classification_messages
                )
                goto = decision.next
                reasoning = decision.reasoning
            except Exception:
                logger.debug("Structured output failed, falling back to regex")
                response = await _ainvoke_with_retry(classification_messages)
                response_text = str(response.content) if hasattr(response, "content") else str(response)
                goto, reasoning = _parse_routing_response(response_text)
        else:
            response = await _ainvoke_with_retry(classification_messages)
            response_text = str(response.content) if hasattr(response, "content") else str(response)
            goto, reasoning = _parse_routing_response(response_text)

        logger.info("Routing: → %s (reason: %s)", goto, reasoning[:80])

        return Command(
            goto=goto,
            update={
                "active_agent": goto,
                "next_agent": None,
                "retry_count": 0,
                "running_summary": running_summary,
                "memory_context": memory_context,
            },
        )

    return supervisor_node
