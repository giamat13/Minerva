"""A real arithmetic evaluator, exposed as a tool.

Small models are unreliable at arithmetic, so handing them a calculator is one
of the highest-value tools you can give them.

This is a genuine expression evaluator built on Python's :mod:`ast` module, not
a wrapper around :func:`eval`. The expression is parsed into a syntax tree and
then walked node by node; any node that is not on the allow-list is rejected
before anything executes. That means no attribute access, no name binding, no
imports, no function calls other than the whitelisted maths functions - a
hostile expression has nothing to reach for.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from ..base import tool

__all__ = ["calculate"]

# Binary operators the evaluator accepts.
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Constants the expression may reference by name.
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Functions the expression may call. Everything here is pure and total-ish;
# domain errors surface as ordinary ValueError/ZeroDivisionError.
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    # sum() accepts either a single sequence, sum([1, 2, 3]), or loose
    # arguments, sum(1, 2, 3) - models write both.
    "sum": lambda *args: (
        sum(args[0]) if len(args) == 1 and hasattr(args[0], "__iter__") else sum(args)
    ),
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "exp": math.exp,
    "log": math.log,  # log(x) natural, log(x, base) with a base
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "hypot": math.hypot,
    "copysign": math.copysign,
    "fmod": math.fmod,
}

# Guard against expressions that are cheap to type and catastrophic to compute,
# e.g. 9**9**9. Applied to `**` and factorial arguments.
_MAX_EXPONENT = 1_000
_MAX_FACTORIAL = 1_000


class _SafeEvaluator(ast.NodeVisitor):
    """Walks an expression tree, computing values for allowed nodes only."""

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(
                f"unsupported syntax in expression: {type(node).__name__}. "
                f"Only arithmetic, comparisons and the documented maths functions are allowed."
            )
        return method(node)

    # -- literals --------------------------------------------------------

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError(f"unsupported literal in expression: {node.value!r}")

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(item) for item in node.elts)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(item) for item in node.elts]

    # -- names -----------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(
            f"unknown name {node.id!r}; known constants: {', '.join(sorted(_CONSTANTS))}"
        )

    # -- operators -------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and _too_big_a_power(left, right):
            raise ValueError(
                f"refusing to evaluate {left}**{right}: exponent exceeds the "
                f"{_MAX_EXPONENT} limit"
            )
        return op(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op_node, comparator in zip(node.ops, node.comparators, strict=True):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise ValueError(f"unsupported comparison: {type(op_node).__name__}")
            right = self.visit(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    # -- calls -----------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct calls to the documented maths functions are allowed")
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(
                f"unknown function {node.func.id!r}; available: "
                f"{', '.join(sorted(_FUNCTIONS))}"
            )
        if node.keywords:
            raise ValueError("keyword arguments are not supported in expressions")
        args = [self.visit(arg) for arg in node.args]
        if (
            node.func.id == "factorial"
            and args
            and isinstance(args[0], int)
            and args[0] > _MAX_FACTORIAL
        ):
            raise ValueError(
                f"refusing to evaluate factorial({args[0]}): above the {_MAX_FACTORIAL} limit"
            )
        return func(*args)


def _too_big_a_power(left: Any, right: Any) -> bool:
    try:
        return abs(right) > _MAX_EXPONENT and abs(left) > 1
    except TypeError:
        return False


@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression exactly and return the result.

    Use this for any arithmetic instead of computing in your head - it is exact
    where mental arithmetic is not. Supports + - * / // % **, parentheses,
    comparisons, the constants pi/e/tau, and functions including sqrt, cbrt,
    exp, log (natural, or log(x, base)), log2, log10, sin, cos, tan, asin,
    acos, atan, atan2, degrees, radians, floor, ceil, round, abs, min, max,
    factorial, gcd, lcm and hypot.

    Args:
        expression: The expression to evaluate, e.g. "(17 * 43) / 6" or
            "sqrt(2) * log(100, 10)". Angles for trigonometric functions are in
            radians; use radians(x) to convert from degrees.
    """
    text = expression.strip()
    if not text:
        raise ValueError("expression is empty")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression {text!r}: {exc.msg}") from exc

    result = _SafeEvaluator().visit(tree)
    return _format_result(result)


def _format_result(value: Any) -> str:
    """Render a result without lying about precision.

    Floats that are exactly integral print as integers (``6.0`` -> ``6``);
    everything else keeps full repr precision so the model never sees a
    silently rounded number.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_result(item) for item in value)
    return str(value)
