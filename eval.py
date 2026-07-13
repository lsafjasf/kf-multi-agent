#!/usr/bin/env python3
"""ShopFast Multi-Agent System — End-to-End Evaluation Runner.

Usage:
    python eval.py                  # Run all scenarios, print summary
    python eval.py --judge          # Run + LLM-as-Judge quality scoring
    python eval.py --verbose        # Full output per scenario
    python eval.py --scenario normal   # Filter by category (normal/handoff/edge/error/escalation)
    python eval.py --scenario order_query_by_id  # Run a single scenario by ID
    python eval.py --json           # JSON output (for CI / programmatic use)
    python eval.py --judge --json   # Combined: run + judge + JSON
    python eval.py --seed-only      # Only re-seed the database (useful after mutation-heavy runs)

Environment variables (same as main.py):
    OPENAI_API_KEY      Your OpenAI / DeepSeek API key (required)
    MODEL_NAME          Model name (default: gpt-4o)
    OPENAI_BASE_URL     Optional custom base URL
    DB_PATH             SQLite database path (default: data/shopfast.db)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.db.connection import DatabaseManager
from src.db.seed import seed_database
from src.graph import build_customer_service_graph
from src.eval.scenarios import ALL_SCENARIOS, EvalScenario, ScenarioCategory
from src.eval.runner import EvalRunner, EvalResult
from src.eval.judge import Judge


# ── CLI entry point ──────────────────────────────────────────────────
async def run_eval(
    scenarios: list[EvalScenario],
    *,
    with_judge: bool = False,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """Run scenarios, optionally score with judge, and report."""

    config = get_config()

    # -- Validate API key -----------------------------------------------
    if config.openai_api_key in ("sk-...", ""):
        print("[ERROR] OPENAI_API_KEY environment variable is required.")
        print("   Set it via: export OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    # -- Model ----------------------------------------------------------
    model_kwargs: dict = {
        "model": config.model_name,
        "temperature": 0.0,  # deterministic for eval
        "api_key": config.openai_api_key,
    }
    if config.openai_base_url:
        model_kwargs["base_url"] = config.openai_base_url

    model = ChatOpenAI(**model_kwargs)

    # -- Database -------------------------------------------------------
    db_path = Path(config.db_path)

    async with DatabaseManager(db_path) as db:
        await db.init_schema()
        # Wipe all existing data so seed inserts work on a clean slate
        await db.execute("DELETE FROM support_tickets")
        await db.execute("DELETE FROM refunds")
        await db.execute("DELETE FROM shipments")
        await db.execute("DELETE FROM orders")
        await db.execute("DELETE FROM customers")
        await db.commit()
        # Now seed fresh
        await seed_database(db)
        print(f"[DB] Database reset and re-seeded at {db_path}")

        # -- Graph -------------------------------------------------------
        print(f"[Model] {config.model_name}")
        print(f"[Eval] Running {len(scenarios)} scenario(s)...\n")
        graph = await build_customer_service_graph(model, db)
        runner = EvalRunner(graph, db)

        t_start = time.perf_counter()

        # Run all scenarios
        results = await runner.run_all(scenarios)

        # Optional judge
        if with_judge:
            print(f"\n[Judge] Scoring {len(results)} conversations...\n")
            judge = Judge(model)
            results = await judge.evaluate_all(results)

        total_time = time.perf_counter() - t_start

        # -- Output -----------------------------------------------------
        if json_output:
            import json
            payload = {
                "summary": EvalRunner.print_summary.__wrapped__  # won't work
            }
            # Just print JSON of results + aggregate
            summary = {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "pass_rate": sum(1 for r in results if r.passed) / len(results) if results else 0,
                "total_duration_s": round(total_time, 1),
                "results": [],
            }
            for r in results:
                entry = {
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
                    "final_message": r.final_message[:300] if r.final_message else "",
                    "judge_scores": r.judge_scores,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                summary["results"].append(entry)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            summary = runner.print_summary(results)
            print(f"\n  Total time: {total_time:.1f}s")

        # Verbose: print full conversations
        if verbose and not json_output:
            print("\n" + "=" * 80)
            print("  VERBOSE OUTPUT")
            print("=" * 80)
            for r in results:
                print(f"\n── {r.scenario_id} ──────────────────────────────")
                print(f"   Agent: {r.actual_agent} | Resolved: {r.resolved} | "
                      f"Passed: {r.passed} | Time: {r.duration_ms:.0f}ms")
                if r.ticket_id:
                    print(f"   Ticket: {r.ticket_id}")
                if r.tools_called:
                    print(f"   Tools: {r.tools_called}")
                if r.agent_path:
                    print(f"   Path: {' → '.join(r.agent_path)}")
                if r.judge_scores:
                    js = r.judge_scores
                    print(f"   Judge: accuracy={js['accuracy']} completeness={js['completeness']} "
                          f"conciseness={js['conciseness']} tone={js['tone']} overall={js['overall']}")
                    print(f"   Notes: {js.get('notes', '')}")
                print(f"   Final ({len(r.final_message)} chars): {r.final_message[:400]}")
                if r.failures:
                    for f in r.failures:
                        print(f"   ✗ {f}")
                if r.error:
                    print(f"   ⚠ ERROR: {r.error[:300]}")

        # Return exit code
        if summary.get("failed", 0) > 0:
            sys.exit(1)


# ── Seed-only mode ───────────────────────────────────────────────────
async def seed_only() -> None:
    """Re-initialize the database with fresh seed data."""
    config = get_config()
    db_path = Path(config.db_path)
    async with DatabaseManager(db_path) as db:
        await db.init_schema()
        await seed_database(db)
    print(f"[OK] Database re-seeded at {db_path}")


# ── Argument parsing ─────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShopFast Multi-Agent Evaluation Runner"
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-Judge quality scoring after each scenario"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full conversation + scores for each scenario"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (for CI)"
    )
    parser.add_argument(
        "--scenario",
        help="Filter: category (normal/handoff/edge/error/escalation) or scenario ID"
    )
    parser.add_argument(
        "--seed-only", action="store_true",
        help="Only re-seed the database, then exit"
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────
async def main() -> None:
    args = parse_args()

    if args.seed_only:
        await seed_only()
        return

    # Filter scenarios
    scenarios = ALL_SCENARIOS
    if args.scenario:
        # First try as category filter
        valid_categories = ("normal", "handoff", "edge", "error", "escalation")
        if args.scenario in valid_categories:
            scenarios = [s for s in ALL_SCENARIOS if s.category == args.scenario]
            if not scenarios:
                print(f"[ERROR] No scenarios in category '{args.scenario}'")
                sys.exit(1)
        else:
            # Try as exact ID match
            matches = [s for s in ALL_SCENARIOS if s.id == args.scenario]
            if not matches:
                print(f"[ERROR] No scenario with id '{args.scenario}'")
                print(f"  Available: {[s.id for s in ALL_SCENARIOS]}")
                sys.exit(1)
            scenarios = matches

    print(f"[Eval] Selected {len(scenarios)} scenario(s)\n")
    for s in scenarios:
        print(f"  [{s.category}] {s.title}")

    await run_eval(
        scenarios,
        with_judge=args.judge,
        verbose=args.verbose,
        json_output=args.json,
    )


if __name__ == "__main__":
    asyncio.run(main())
