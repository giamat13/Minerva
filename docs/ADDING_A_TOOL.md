# Adding a tool

A tool is a Python function the model may call mid-conversation. Minerva
derives the JSON Schema the model sees from your **real type hints and
docstring**, so there is no second copy of the signature to drift out of sync.

---

## The quick way: the `@tool` decorator

```python
from minerva import tool

@tool
def check_stock(product: str, warehouse: str = "main") -> dict:
    """Check how many units of a product are in a warehouse.

    Use this whenever you are asked about availability. Never guess a
    stock figure.

    Args:
        product: The product name, e.g. "swift".
        warehouse: Which warehouse to check. Defaults to the main one.
    """
    return {"product": product, "units": lookup(product, warehouse)}
```

That produces exactly this schema:

```json
{
  "type": "function",
  "function": {
    "name": "check_stock",
    "description": "Check how many units of a product are in a warehouse. Use this whenever you are asked about availability. Never guess a stock figure.",
    "parameters": {
      "type": "object",
      "properties": {
        "product":   {"type": "string", "description": "The product name, e.g. \"swift\"."},
        "warehouse": {"type": "string", "description": "Which warehouse to check. Defaults to the main one.", "default": "main"}
      },
      "required": ["product"],
      "additionalProperties": false
    }
  }
}
```

Use it:

```python
from minerva import Agent, ToolRegistry, load_model

agent = Agent(load_model("swift"), tools=ToolRegistry([check_stock]))
print(agent.run("How many swifts are in stock?").answer)
```

### What the docstring is for

The description is **prompt text**. It is the only thing telling the model when
to reach for this tool, so write it for the model:

- Say what the tool does *and when to use it*.
- Say when **not** to use it, if there is an obvious wrong case.
- Spell out units, formats and conventions ("ISO date", "radians", "IANA time
  zone name") — a model cannot infer them.

Both Google style (`Args:`) and Sphinx style (`:param x:`) are understood.

### Supported types

| Python | JSON Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[T]` | `array` with `items` |
| `dict[str, T]` | `object` with `additionalProperties` |
| `Literal["a", "b"]` | `enum` |
| an `Enum` subclass | `enum` of its values |
| `T \| None` | `T`, and the field is not required |
| `A \| B` | `anyOf` |

Anything else degrades to an unconstrained schema rather than raising — a
slightly loose schema beats a tool that refuses to be defined.

---

## The full way: subclass `Tool`

Use this when the tool needs state — a database handle, an HTTP session, a
cache — or a schema too intricate to infer.

```python
from typing import Any
from minerva.tools import Tool

class DatabaseQueryTool(Tool):
    name = "query_orders"
    description = "Run a read-only query against the orders table."

    def __init__(self, connection):
        self._connection = connection

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "..."},
                "limit": {"type": "integer", "default": 10, "maximum": 100},
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        }

    def run(self, customer_id: str, limit: int = 10) -> list[dict]:
        return self._connection.fetch(customer_id, limit)
```

Override `validate_arguments` too if you want stricter checking than the
built-in structural check (missing required / unknown argument).

---

## How failures are handled

**A tool that raises is not a crash.** The registry catches it, wraps it as a
`ToolResult` with `is_error=True`, and feeds the message back to the model so
it can retry with different arguments or explain the problem:

```
model  -> check_stock(product="swfit")
tool   -> ToolExecutionError: unknown product 'swfit'; known: swift, kestrel, owl
model  -> check_stock(product="swift")
tool   -> {"product": "swift", "units": 42}
model  -> "There are 42 swifts in stock."
```

So write error messages **for the model**: say what went wrong and what a valid
input looks like. That sentence is what it gets to act on.

---

## Making a tool a built-in

Built-ins ship with the platform and load into `default_registry()`.

1. Write it in `src/minerva/tools/builtin/<name>.py`.
2. Import it in `src/minerva/tools/builtin/__init__.py` and add it to
   `_BUILTIN_TOOLS`.
3. Add tests to `tests/test_tools_builtin.py` covering the real behaviour
   **and the failure paths**.

Hold built-ins to a higher bar than a one-off tool: every model in every
conversation pays for their description in context, so they must earn it.

### Safety

`calculate` is the reference for a built-in that touches untrusted input: it
parses the expression into an AST and walks an explicit allow-list of nodes, so
attribute access, imports, name binding and arbitrary calls are rejected
*before* anything executes. If your tool touches the filesystem, the network or
a shell, decide deliberately what it may reach and enforce it in the same way —
never with a blocklist.

---

## Checklist

- [ ] Type hints on every parameter
- [ ] A docstring written for the model, with an `Args:` entry per parameter
- [ ] Units, formats and conventions stated explicitly
- [ ] Error messages that tell the model how to correct itself
- [ ] Tests for the happy path **and** the failure paths
- [ ] For built-ins: registered in `builtin/__init__.py`
