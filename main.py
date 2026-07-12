#!/usr/bin/env python3
"""ShopFast Customer Service — LangGraph Multi-Agent System.

Usage:
    python main.py                  # Interactive CLI chat
    python main.py --demo           # Run 5 built-in demo scenarios
    python main.py --seed-only      # Only initialize DB and seed data

Environment variables:
    OPENAI_API_KEY      Your OpenAI API key (required)
    MODEL_NAME          Model to use (default: gpt-4o)
    OPENAI_BASE_URL     Optional custom base URL
    DB_PATH             SQLite database path (default: data/shopfast.db)
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys

import sys
import io

# Force UTF-8 output to avoid UnicodeEncodeError on Windows GBK terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.config import get_config
from src.db.connection import DatabaseManager
from src.db.seed import seed_database
from src.graph import build_customer_service_graph


# -- Demo scenarios --------------------------------------------------
DEMO_SCENARIOS = [
    {
        "title": "1. Order Query (order_agent)",
        "user": "CUST-001",
        "message": "Hi, I'd like to check the status of my order ORD-001.",
        "expected": "Order agent queries SQLite, returns order details.",
    },
    {
        "title": "2. Logistics Handoff (order_agent → logistics_agent)",
        "user": "CUST-001",
        "message": "Where is my package for order ORD-002? I need a tracking update.",
        "expected": "Order agent looks up order, sees it's shipped, hands off to logistics. Logistics tracks the shipment.",
    },
    {
        "title": "3. Refund Request (refund_agent)",
        "user": "CUST-004",
        "message": "I received order ORD-008 but the desk lamp arrived broken. I want a refund.",
        "expected": "Refund agent checks eligibility (within 30 days), processes refund.",
    },
    {
        "title": "4. Lost Package → Escalation (logistics_agent → human_agent)",
        "user": "CUST-003",
        "message": "My package for order ORD-005 with tracking TRK-20003 has been stuck in exception for 2 weeks! This is a $1,299 laptop — I need this resolved NOW.",
        "expected": "Logistics tracks, sees exception, reports lost package, escalates to human due to high value + long delay.",
    },
    {
        "title": "5. Explicit Human Request (supervisor → human_agent)",
        "user": "CUST-002",
        "message": "I don't want to talk to a bot. Get me a real human agent right now.",
        "expected": "Supervisor routes directly to human_agent. Support ticket created.",
    },
]


# -- Interactive mode ------------------------------------------------
async def run_interactive(graph) -> None:
    """Simple CLI chat loop for manual testing."""
    print("\n" + "=" * 60)
    print("  ShopFast Customer Service — Multi-Agent System")
    print("  Type 'quit' / 'exit' to leave, 'reset' to start fresh")
    print("=" * 60)

    config = {"configurable": {"thread_id": "cli-session"}}
    state_snapshot = None

    while True:
        try:
            user_input = input("\n[You] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            config = {"configurable": {"thread_id": f"cli-{os.urandom(4).hex()}"}}
            state_snapshot = None
            print("[Reset] Session reset.")
            continue

        if not user_input:
            continue

        # Build input state — carry over user context from previous turns
        input_state = {"messages": [HumanMessage(content=user_input)]}
        if state_snapshot:
            input_state["user_id"] = state_snapshot.get("user_id")
            input_state["order_id"] = state_snapshot.get("order_id")

        print(f"\n... Processing...", end="\r")

        # Stream through the graph to show agent transitions
        last_agent = None
        async for chunk in graph.astream(input_state, config=config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name != last_agent:
                    print(f"[Route] Routed to: {node_name}" + " " * 30)
                    last_agent = node_name

                # Show tool calls if present
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"    [Tool]  Calling: {tc['name']}({str(tc.get('args', {}))[:80]}...)")
                    elif hasattr(msg, "name") and msg.name and "transfer" in str(msg.name):
                        print(f"  [Reset] {msg.content[:120]}")

        # Get final state
        result = await graph.ainvoke(input_state, config=config)
        state_snapshot = result

        # Print the last AI message
        final_msgs = [m for m in result.get("messages", []) if hasattr(m, "content") and m.type == "ai"]
        if final_msgs:
            print(f"\n[Agent] Agent: {final_msgs[-1].content}")

        # Show ticket if escalated
        if result.get("support_ticket_id"):
            print(f"\n  --- Support Ticket: {result['support_ticket_id']}")

        # Show resolution path
        agent_path = result.get("active_agent", "?")
        print(f"  [OK] Resolved: {result.get('resolved')}  |  Last agent: {agent_path}")


# -- Demo mode -------------------------------------------------------
async def run_demo(graph) -> None:
    """Run through pre-defined demo scenarios."""
    print("\n" + "=" * 60)
    print("  ShopFast Multi-Agent Demo — 5 Scenarios")
    print("=" * 60)

    for i, scenario in enumerate(DEMO_SCENARIOS):
        print(f"\n{'-' * 60}")
        print(f"--- SCENARIO {scenario['title']}")
        print(f"   Expected: {scenario['expected']}")
        print(f"{'-' * 60}")
        print(f"[You] Customer: {scenario['message']}")

        config = {"configurable": {"thread_id": f"demo-{i}"}}

        input_state = {
            "messages": [HumanMessage(content=scenario["message"])],
            "user_id": scenario["user"],
        }

        print(f"... Processing...", end="\r")

        last_agent = None
        try:
            async for chunk in graph.astream(input_state, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name != last_agent:
                        print(f"  [Route] → {node_name}")
                        last_agent = node_name

                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_str = str(tc.get("args", {}))
                                if len(args_str) > 100:
                                    args_str = args_str[:100] + "...}"
                                print(f"       [Tool]  {tc['name']}({args_str})")
                        elif hasattr(msg, "name") and msg.name and "transfer" in str(msg.name):
                            print(f"     [Reset] {msg.content[:150]}")
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            continue

        result = await graph.ainvoke(input_state, config=config)

        # Show final response
        final_ai = [
            m for m in result.get("messages", [])
            if hasattr(m, "content") and m.type == "ai" and m.content
        ]
        if final_ai:
            content = final_ai[-1].content
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n  [Agent] Response: {content}")

        ticket = result.get("support_ticket_id")
        if ticket:
            print(f"\n  --- ESCALATED → Ticket #{ticket}")

        print(f"  [OK] Resolved: {result.get('resolved')}  |  Path: {result.get('active_agent', '?')}")


# -- Main ------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShopFast Customer Service — LangGraph Multi-Agent System"
    )
    parser.add_argument("--demo", action="store_true", help="Run built-in demo scenarios")
    parser.add_argument("--seed-only", action="store_true", help="Only initialize DB and exit")
    args = parser.parse_args()

    config = get_config()

    # -- Database -------------------------------------------------
    print(f"[DB] Initializing database at {config.db_path}...")
    async with DatabaseManager(config.db_path) as db:
        await db.init_schema()
        await seed_database(db)
        print("   Database ready with seed data (4 customers, 8 orders).")

        if args.seed_only:
            print("[OK] Seed complete. Exiting.")
            return

        # Validate API key (only needed beyond seed-only)
        if config.openai_api_key in ("sk-...", ""):
            print("[ERROR] OPENAI_API_KEY environment variable is required.")
            print("   Set it via: export OPENAI_API_KEY=sk-your-key-here")
            sys.exit(1)

        # -- Model -------------------------------------------------
        model_kwargs = {
            "model": config.model_name,
            "temperature": config.model_temperature,
            "api_key": config.openai_api_key,
        }
        if config.openai_base_url:
            model_kwargs["base_url"] = config.openai_base_url

        model = ChatOpenAI(**model_kwargs)
        print(f"[Model] {config.model_name}")

        # -- Build Graph -------------------------------------------
        print("[Graph] Building agent graph...")
        graph = await build_customer_service_graph(model, db)
        print("   supervisor -> order_agent / logistics_agent / refund_agent / human_agent")

        # -- Run ---------------------------------------------------
        if args.demo:
            await run_demo(graph)
        else:
            await run_interactive(graph)


if __name__ == "__main__":
    asyncio.run(main())
