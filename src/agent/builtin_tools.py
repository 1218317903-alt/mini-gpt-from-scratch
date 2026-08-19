"""Small built-in tools with deliberately narrow capabilities."""

from __future__ import annotations

import ast
import math
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from agent.tools.base import ToolSpec
from agent.tools.registry import ToolRegistry


class CalculatorArguments(BaseModel):
    """Arguments accepted by the safe arithmetic calculator."""

    expression: str = Field(min_length=1, max_length=256)


class CurrentTimeArguments(BaseModel):
    """Arguments accepted by the current-time tool."""

    timezone: str = Field(default="UTC", min_length=1, max_length=64)


def _safe_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Only integer and decimal numeric literals are allowed.")
    if not math.isfinite(float(value)):
        raise ValueError("The number must be finite.")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _safe_number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100:
                raise ValueError("Exponent is too large.")
            result = left**right
        else:
            raise ValueError("Only basic arithmetic operators are allowed.")
        try:
            result_as_float = float(result)
        except OverflowError as error:
            raise ValueError(
                "Arithmetic result is outside the safe numeric range."
            ) from error
        if not math.isfinite(result_as_float) or abs(result_as_float) > 1e100:
            raise ValueError("Arithmetic result is outside the safe numeric range.")
        return result
    raise ValueError("Only numeric literals and basic arithmetic are allowed.")


def calculate(expression: str) -> int | float:
    """Evaluate a bounded arithmetic expression without executing code."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"Invalid arithmetic expression: {error}") from error
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def current_time(timezone: str = "UTC") -> str:
    """Return an ISO-8601 timestamp for an IANA timezone."""
    try:
        zone = ZoneInfo(timezone)
    except Exception as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error
    return datetime.now(zone).isoformat(timespec="seconds")


def build_default_registry() -> ToolRegistry:
    """Build the safe default tool allow-list for the local Agent."""
    return ToolRegistry(
        [
            ToolSpec(
                name="calculator",
                description="Evaluate a simple arithmetic expression safely.",
                arguments_model=CalculatorArguments,
                handler=calculate,
            ),
            ToolSpec(
                name="current_time",
                description="Return the current ISO-8601 time in an IANA timezone.",
                arguments_model=CurrentTimeArguments,
                handler=current_time,
            ),
        ]
    )
