"""Scenario runner — executes eval scenarios against the graph and collects results."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from src.db.connection import DatabaseManager
from src.eval.scenarios import EvalScenario


@dataclass
class EvalResult:
    """Collected data after running one scenario."""

    scenario_id: str
    title: str
    category: str
    passed: bool
    actual_agent: str = ""
    resolved: bool = False
    ticket_id: str | None = None
    tools_called: list[str] = field(default_factory=list)
    agent_path: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    final_message: str = ""
    conversation: list[dict] = field(default_factory=list)
    judge_scores: dict | None = None
    duration_ms: float = 0.0
    error: str | None = None


class EvalRunner:
    """Runs eval scenarios against a compiled graph + database.

    Usage::

        runner = EvalRunner(graph, db)
        results = await runner.run_all(scenarios)
        runner.print_summary(results)
    """

    def __init__(self, graph, db: DatabaseManager):
        self.graph = graph
        self.db = db

    # ── Core run logic ──────────────────────────────────────────────
    async def run_scenario(self, scenario: EvalScenario) -> EvalResult:
        """Run a single scenario end-to-end and collect results."""
        result = EvalResult(
            scenario_id=scenario.id,
            title=scenario.title,
            category=scenario.category,
            passed=True,
        )

        thread_id = f"eval-{scenario.id}-{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": thread_id}}

        input_state = {
            "messages": [HumanMessage(content=scenario.message)],
            "user_id": scenario.user_id,
            "session_id": f"eval-sess-{uuid.uuid4().hex[:12]}",
            "session_started_at": time.perf_counter(),
        }

        t_start = time.perf_counter()

        try:
            final_state = await self.graph.ainvoke(input_state, config=config)
        except Exception as exc:
            result.duration_ms = (time.perf_counter() - t_start) * 1000
            result.passed = False
            result.error = str(exc)
            result.failures.append(f"Graph invocation crashed: {exc}")
            return result

        result.duration_ms = (time.perf_counter() - t_start) * 1000

        # ── Extract state ───────────────────────────────────────────
        result.actual_agent = final_state.get("active_agent", "?")
        result.resolved = final_state.get("resolved", False)
        result.ticket_id = final_state.get("support_ticket_id")

        # Extract agent path from messages (names set by subgraph nodes)
        messages = final_state.get("messages", [])
        seen_agents = []
        for msg in messages:
            name = getattr(msg, "name", None)
            if name and name in (
                "supervisor", "order_agent", "logistics_agent",
                "refund_agent", "human_agent",
            ) and name not in seen_agents:
                seen_agents.append(name)
        # Also capture agents from tool calls
        tool_names = set()
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.add(tc.get("name", ""))
            # AIMessage with name="handoff" or "escalation"
            if hasattr(msg, "name") and msg.name in ("handoff", "escalation"):
                content = getattr(msg, "content", "")
                if content:
                    # Parse AIMessage content for handoff target
                    pass

        result.tools_called = sorted(tool_names)
        result.agent_path = seen_agents

        # Extract final AI message text
        for msg in reversed(messages):
            if hasattr(msg, "content") and getattr(msg, "type", "") == "ai" and msg.content:
                result.final_message = str(msg.content)[:500]
                break

        # Build conversation for judge (human + ai messages only)
        result.conversation = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type in ("human", "ai"):
                content = getattr(msg, "content", "")
                if isinstance(content, str):
                    result.conversation.append({
                        "role": "user" if msg_type == "human" else "assistant",
                        "content": content[:1000],
                    })

        # ── Run assertions ──────────────────────────────────────────
        self._check_assertions(scenario, result)
        await self._check_db_assertions(scenario, result)

        return result

    def _check_assertions(
        self, scenario: EvalScenario, result: EvalResult
    ) -> None:
        """Run all soft assertions against result state."""
        # Final agent
        if scenario.expected_agent is not None:
            if result.actual_agent != scenario.expected_agent:
                result.passed = False
                result.failures.append(
                    f"Expected final agent '{scenario.expected_agent}', "
                    f"got '{result.actual_agent}'"
                )

        # Resolution status
        if scenario.expected_resolved is not None:
            if result.resolved != scenario.expected_resolved:
                result.passed = False
                result.failures.append(
                    f"Expected resolved={scenario.expected_resolved}, "
                    f"got resolved={result.resolved}"
                )

        # Support ticket
        if scenario.expected_ticket and not result.ticket_id:
            result.passed = False
            result.failures.append("Expected a support ticket to be created, got none")

        # Expected tools
        if scenario.expected_tools:
            for tool_name in scenario.expected_tools:
                if tool_name not in result.tools_called:
                    result.passed = False
                    result.failures.append(
                        f"Expected tool '{tool_name}' was not called. "
                        f"Tools called: {result.tools_called}"
                    )

        # Forbidden tools
        if scenario.forbidden_tools:
            for tool_name in scenario.forbidden_tools:
                if tool_name in result.tools_called:
                    result.passed = False
                    result.failures.append(
                        f"Forbidden tool '{tool_name}' was called."
                    )

        # Agent trace contains
        if scenario.agent_trace_contains:
            for agent_name in scenario.agent_trace_contains:
                if agent_name not in result.agent_path:
                    result.passed = False
                    result.failures.append(
                        f"Expected agent '{agent_name}' in trace, "
                        f"got {result.agent_path}"
                    )

    async def _check_db_assertions(
        self, scenario: EvalScenario, result: EvalResult
    ) -> None:
        """Run post-scenario DB checks."""
        for check in scenario.db_checks:
            try:
                row = await self.db.fetch_one(
                    check["sql"], tuple(check.get("params", []))
                )
            except Exception as exc:
                result.passed = False
                result.failures.append(
                    f"DB check '{check['description']}' failed: {exc}"
                )
                continue

            expected = check.get("expected")
            if expected is not None:
                if row is None:
                    result.passed = False
                    result.failures.append(
                        f"DB check '{check['description']}': no row returned"
                    )
                else:
                    for key, value in expected.items():
                        actual = row.get(key)
                        if actual != value:
                            result.passed = False
                            result.failures.append(
                                f"DB check '{check['description']}': "
                                f"{key} expected '{value}', got '{actual}'"
                            )

    # ── Batch runner ─────────────────────────────────────────────────
    async def run_all(self, scenarios: list[EvalScenario]) -> list[EvalResult]:
        """Run a list of scenarios sequentially and return results."""
        results = []
        for i, scenario in enumerate(scenarios):
            print(f"  [{i+1}/{len(scenarios)}] {scenario.id} ...", end=" ", flush=True)
            result = await self.run_scenario(scenario)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            if result.error:
                status = "✗ ERROR"
            print(f"{status}  ({result.duration_ms:.0f}ms)")
            results.append(result)
        return results

    # ── Reporting ────────────────────────────────────────────────────
    @staticmethod
    def print_summary(results: list[EvalResult]) -> dict:
        """Print a summary table and return aggregate stats."""
        by_category: dict[str, dict] = {}
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        print("\n" + "=" * 80)
        print(f"  EVALUATION SUMMARY  —  {passed}/{total} passed")
        print("=" * 80)

        # Header
        print(f"  {'Scenario':<36} {'Category':<12} {'Result':<8} {'Agent':<18} {'Details'}")
        print("  " + "-" * 76)

        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            if r.error:
                status = "✗ ERR"

            details = ""
            if r.ticket_id:
                details += f"ticket={r.ticket_id} "
            if r.tools_called:
                tools_str = ",".join(r.tools_called[:3])
                if len(r.tools_called) > 3:
                    tools_str += f" +{len(r.tools_called)-3}"
                details += f"tools=[{tools_str}]"

            print(
                f"  {r.title[:34]:<36} {r.category:<12} "
                f"{status:<8} {r.actual_agent:<18} {details[:50]}"
            )

        # Per-category breakdown
        print("\n  ── By Category ──")
        for r in results:
            c = r.category
            if c not in by_category:
                by_category[c] = {"total": 0, "passed": 0}
            by_category[c]["total"] += 1
            if r.passed:
                by_category[c]["passed"] += 1

        for cat in ("normal", "handoff", "edge", "error", "escalation"):
            stats = by_category.get(cat)
            if stats:
                print(
                    f"  {cat:<12} {stats['passed']}/{stats['total']} passed  "
                    f"({stats['passed']/stats['total']*100:.0f}%)"
                )

        # Failures detail
        failures = [r for r in results if not r.passed]
        if failures:
            print(f"\n  ── Failures ({len(failures)}) ──")
            for r in failures:
                print(f"  [{r.scenario_id}] {r.title}")
                for f in r.failures:
                    print(f"    └ {f}")
                if r.error:
                    print(f"    └ [ERROR] {r.error[:200]}")

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0,
            "by_category": by_category,
            "total_duration_ms": sum(r.duration_ms for r in results),
        }

    @staticmethod
    def to_json(results: list[EvalResult]) -> str:
        """Serialize results to a JSON string."""
        import json

        data = []
        for r in results:
            data.append({
                "scenario_id": r.scenario_id,
                "title": r.title,
                "category": r.category,
                "passed": r.passed,
                "actual_agent": r.actual_agent,
                "resolved": r.resolved,
                "ticket_id": r.ticket_id,
                "tools_called": r.tools_called,
                "agent_path": r.agent_path,
                "failures": r.failures,
                "final_message": r.final_message[:300],
                "judge_scores": r.judge_scores,
                "duration_ms": r.duration_ms,
                "error": r.error,
            })
        return json.dumps(data, indent=2, ensure_ascii=False)
