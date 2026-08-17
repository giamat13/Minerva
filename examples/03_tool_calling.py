#!/usr/bin/env python3
"""Example 3: tool calling.

Defines a custom tool, hands it to Swift alongside the built-ins, and lets the
agent loop run it. The tool really executes - watch the print statement inside
it fire.

Run it::

    python examples/03_tool_calling.py
"""

from __future__ import annotations

import sys

from minerva import Agent, ToolRegistry, load_model, tool
from minerva.errors import MinervaError
from minerva.tools import default_registry

# A small, real "database". A production tool would query a real one; the point
# is that the model cannot know this from its weights - it has to ask.
_STOCK = {
    "swift": 42,
    "kestrel": 7,
    "owl": 0,
}


@tool
def check_stock(product: str) -> dict:
    """Check how many units of a product are currently in the warehouse.

    Use this whenever you are asked about availability or stock levels. Never
    guess a stock figure.

    Args:
        product: The product name, e.g. "swift" or "kestrel".
    """
    print(f"    [the tool really ran: check_stock(product={product!r})]")
    key = product.strip().lower()
    if key not in _STOCK:
        return {"product": product, "known": False, "known_products": sorted(_STOCK)}
    return {"product": key, "known": True, "units_in_stock": _STOCK[key]}


def main() -> int:
    model = load_model("swift")

    # Combine the built-in tools with our own. A registry is just a set of
    # tools, so different agents can be given different surfaces.
    tools = ToolRegistry([*default_registry(), check_stock])
    print(f"tools available to the model: {', '.join(tools.names())}\n")

    agent = Agent(
        model,
        tools=tools,
        thinking="fa",
        on_tool_call=lambda call: print(f"  -> model asked for: {call}"),
        on_tool_result=lambda result: print(f"  <- tool returned:  {result.content}"),
    )

    for question in (
        "How many units of kestrel do we have in stock?",
        "We have 42 swifts at 19 shekels each. Use the calculator for the total value.",
        "What is today's date in Asia/Jerusalem?",
    ):
        print(f"\nQ: {question}")
        run = agent.run(question)
        print(f"A: {run.answer.strip()}")
        print(
            f"   ({len(run.steps)} model turn(s), "
            f"{len(run.tool_calls)} tool call(s), "
            f"{run.duration_seconds:.1f}s)"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MinervaError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
