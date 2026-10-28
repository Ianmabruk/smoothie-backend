"""Safe calculation layer for Smoothie AI.

Deterministic numerical tools that back the AI's math answers. Nothing here
executes arbitrary user-supplied code: expressions are parsed with the Python
``ast`` module and only a small whitelist of node types is allowed.
"""

from __future__ import annotations

import ast
import math
import operator as op
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class CalculationError(ValueError):
    """Raised when a calculation cannot be performed safely."""


# Binary operators allowed in arithmetic expressions.
_BINARY_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Unary operators allowed in arithmetic expressions.
_UNARY_OPS = {
    ast.UAdd: lambda x: x,
    ast.USub: op.neg,
}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

# Hard limits to keep evaluation bounded.
_MAX_POWER_EXPONENT = 100
_MAX_ABS_RESULT = 1e18


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise CalculationError("Unsupported constant value")
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # pragma: no cover - Python < 3.8 compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        func = _BINARY_OPS.get(type(node.op))
        if func is None:
            raise CalculationError("Unsupported operator")
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_POWER_EXPONENT:
                raise CalculationError("Exponent too large")
        result = func(left, right)
    elif isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise CalculationError("Unsupported unary operator")
        result = func(_eval_node(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculationError("Only named function calls are allowed")
        name = node.func.id
        if name not in _ALLOWED_NAMES:
            raise CalculationError(f"Unsupported function: {name}")
        func = _ALLOWED_NAMES[name]
        args = [_eval_node(a) for a in node.args]
        if node.keywords:
            raise CalculationError("Keyword arguments are not allowed")
        if name == "round" and len(args) == 1:
            result = round(args[0])
        elif name == "round" and len(args) == 2:
            result = round(args[0], int(args[1]))
        else:
            result = func(*args)
    elif isinstance(node, ast.Compare):
        raise CalculationError("Comparisons are not allowed in calculations")
    else:
        raise CalculationError(f"Unsupported expression element: {type(node).__name__}")

    if isinstance(result, complex) or math.isnan(result) or math.isinf(result):
        raise CalculationError("Result is not a finite real number")
    if abs(result) > _MAX_ABS_RESULT:
        raise CalculationError("Result is too large")
    return float(result)


def evaluate_expression(expression: str) -> Dict[str, Any]:
    """Safely evaluate a mathematical expression.

    Returns ``{"ok": True, "result": float}`` or ``{"ok": False, "error": str}``.
    """
    if not expression or not expression.strip():
        return {"ok": False, "error": "Empty expression."}
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return {"ok": True, "result": result, "expression": expression.strip()}
    except CalculationError as exc:
        return {"ok": False, "error": str(exc)}
    except SyntaxError:
        return {"ok": False, "error": "Invalid mathematical expression."}
    except Exception as exc:  # noqa: BLE001 - defensive boundary for user input
        return {"ok": False, "error": f"Calculation failed: {exc}"}


@dataclass
class CalculationStep:
    label: str
    detail: str


def simple_interest(principal: float, rate_percent: float, years: float) -> Dict[str, Any]:
    """Simple interest: I = P * R * T / 100."""
    steps: List[CalculationStep] = [
        CalculationStep("Identify the values", f"Principal P = {principal}; annual rate R = {rate_percent}%; time T = {years} year(s)."),
        CalculationStep("Formula", "I = (P x R x T) / 100"),
        CalculationStep("Substitute", f"I = ({principal} x {rate_percent} x {years}) / 100"),
    ]
    interest = (principal * rate_percent * years) / 100
    steps.append(CalculationStep("Calculate", f"I = {principal * rate_percent * years} / 100 = {interest}"))
    steps.append(CalculationStep("Verify", f"{rate_percent}% of {principal} per year is {(principal * rate_percent / 100)}; over {years} year(s): {(principal * rate_percent / 100) * years} = {interest}"))
    steps.append(CalculationStep("Final answer", f"The simple interest is {interest}."))
    return _result("simple interest", interest, steps)


def compound_interest(principal: float, rate_percent: float, years: float, compounds_per_year: int = 1) -> Dict[str, Any]:
    """Compound interest: A = P(1 + r/n)^(nt); interest = A - P."""
    r = rate_percent / 100
    n = int(compounds_per_year)
    t = years
    steps: List[CalculationStep] = [
        CalculationStep("Identify the values", f"Principal P = {principal}; annual rate R = {rate_percent}%; time T = {t} year(s); compounding n = {n} time(s)/year."),
        CalculationStep("Formula", "A = P x (1 + r/n)^(n x t), where r = R/100. Interest I = A - P."),
        CalculationStep("Substitute", f"r = {rate_percent}/100 = {r}; A = {principal} x (1 + {r}/{n})^({n} x {t})"),
    ]
    amount = principal * math.pow(1 + r / n, n * t)
    interest = amount - principal
    steps.append(CalculationStep("Calculate", f"A = {principal} x (1 + {r})^({n * t}) = {amount}; I = {amount} - {principal} = {interest}"))
    steps.append(CalculationStep("Final answer", f"The compound interest is {interest}; total amount is {amount}."))
    return _result("compound interest", interest, steps, extra={"amount": amount})


def linear_equation(a: float, b: float, c: float) -> Dict[str, Any]:
    """Solve a x + b = c for x."""
    steps: List[CalculationStep] = [
        CalculationStep("Identify the equation", f"{a}x + {b} = {c}"),
        CalculationStep("Formula", "Subtract b from both sides, then divide by a: x = (c - b) / a"),
        CalculationStep("Substitute", f"x = ({c} - {b}) / {a}"),
    ]
    if a == 0:
        if b == c:
            return _result("linear equation", None, steps + [CalculationStep("Result", "Infinitely many solutions (any real x).")], extra={"infinite": True})
        return _result("linear equation", None, steps + [CalculationStep("Result", "No solution (0 x = non-zero).")], extra={"none": True})
    x = (c - b) / a
    steps.append(CalculationStep("Calculate", f"x = ({c} - {b}) / {a} = {c - b} / {a} = {x}"))
    steps.append(CalculationStep("Verify", f"{a} x {x} + {b} = {a * x + b} = {c}"))
    steps.append(CalculationStep("Final answer", f"x = {x}"))
    return _result("linear equation", x, steps)


def quadratic_equation(a: float, b: float, c: float) -> Dict[str, Any]:
    """Solve a x^2 + b x + c = 0."""
    steps: List[CalculationStep] = [
        CalculationStep("Identify the equation", f"{a}x^2 + {b}x + {c} = 0"),
        CalculationStep("Formula", "x = (-b ± sqrt(b^2 - 4ac)) / (2a)"),
        CalculationStep("Discriminant", f"D = b^2 - 4ac = {b}^2 - 4({a})({c}) = {b * b - 4 * a * c}"),
    ]
    discriminant = b * b - 4 * a * c
    if a == 0:
        return linear_equation(b, c, 0.0)
    if discriminant < 0:
        steps.append(CalculationStep("Result", "No real solutions (negative discriminant)."))
        return _result("quadratic equation", None, steps, extra={"real_roots": False})
    if discriminant == 0:
        x = -b / (2 * a)
        steps.append(CalculationStep("Calculate", f"x = -{b} / (2 x {a}) = {x}"))
        steps.append(CalculationStep("Final answer", f"One real solution (repeated root): x = {x}"))
        return _result("quadratic equation", x, steps, extra={"roots": [x], "repeated": True})
    sqrt_d = math.sqrt(discriminant)
    x1 = (-b + sqrt_d) / (2 * a)
    x2 = (-b - sqrt_d) / (2 * a)
    steps.append(CalculationStep("Calculate", f"x1 = (-{b} + {sqrt_d}) / (2 x {a}) = {x1}"))
    steps.append(CalculationStep("Calculate", f"x2 = (-{b} - {sqrt_d}) / (2 x {a}) = {x2}"))
    steps.append(CalculationStep("Verify", f"Check x1: {a} x {x1}^2 + {b} x {x1} + {c} ≈ 0"))
    steps.append(CalculationStep("Final answer", f"Two real solutions: x = {x1} and x = {x2}"))
    return _result("quadratic equation", None, steps, extra={"roots": [x1, x2], "real_roots": True})


def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """Convert between common metric/imperial units."""
    # Base unit = SI where applicable.
    length_to_m = {
        "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
    }
    mass_to_kg = {
        "mg": 1e-6, "g": 0.001, "kg": 1.0, "t": 1000.0,
        "oz": 0.0283495, "lb": 0.453592,
    }
    from_unit_l = from_unit.lower()
    to_unit_l = to_unit.lower()
    table = None
    if from_unit_l in length_to_m and to_unit_l in length_to_m:
        table = length_to_m
    elif from_unit_l in mass_to_kg and to_unit_l in mass_to_kg:
        table = mass_to_kg
    if table is None:
        return {"ok": False, "error": f"Unsupported units: {from_unit} -> {to_unit}"}
    result = value * table[from_unit_l] / table[to_unit_l]
    steps: List[CalculationStep] = [
        CalculationStep("Identify", f"Convert {value} {from_unit} to {to_unit}."),
        CalculationStep("Calculate", f"{value} x ({table[from_unit_l]} / {table[to_unit_l]}) = {result}"),
        CalculationStep("Final answer", f"{value} {from_unit} = {result} {to_unit}."),
    ]
    return _result("unit conversion", result, steps, extra={"from_unit": from_unit, "to_unit": to_unit})


def _result(kind: str, value: Any, steps: List[CalculationStep], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "tool": kind,
        "result": value,
        "steps": [{"label": s.label, "detail": s.detail} for s in steps],
        **(extra or {}),
    }


def solve_physics_kinematics(*, initial_velocity=None, final_velocity=None, acceleration=None, time=None, displacement=None) -> Dict[str, Any]:
    """Solve constant-acceleration kinematics using v = u + at and s = ut + 1/2 a t^2.

    Exactly three of the five quantities must be provided.
    """
    known = {
        "u": initial_velocity,
        "v": final_velocity,
        "a": acceleration,
        "t": time,
        "s": displacement,
    }
    provided = {k: v for k, v in known.items() if v is not None}
    steps: List[CalculationStep] = [
        CalculationStep("Known values", ", ".join(f"{k} = {v}" for k, v in provided.items()) or "none"),
    ]
    if len(provided) < 3:
        return {"ok": False, "error": "Provide at least three of: initial_velocity, final_velocity, acceleration, time, displacement."}

    u = provided.get("u")
    v = provided.get("v")
    a = provided.get("a")
    t = provided.get("t")
    s = provided.get("s")

    # v = u + a t
    if v is None and u is not None and a is not None and t is not None:
        v = u + a * t
        steps.append(CalculationStep("Formula", "v = u + a t"))
        steps.append(CalculationStep("Calculate", f"v = {u} + ({a} x {t}) = {v}"))
    elif u is None and v is not None and a is not None and t is not None:
        u = v - a * t
        steps.append(CalculationStep("Formula", "u = v - a t"))
        steps.append(CalculationStep("Calculate", f"u = {v} - ({a} x {t}) = {u}"))
    elif a is None and u is not None and v is not None and t is not None:
        if t == 0:
            return {"ok": False, "error": "Time cannot be zero when solving for acceleration."}
        a = (v - u) / t
        steps.append(CalculationStep("Formula", "a = (v - u) / t"))
        steps.append(CalculationStep("Calculate", f"a = ({v} - {u}) / {t} = {a}"))
    elif t is None and u is not None and v is not None and a is not None:
        if a == 0:
            return {"ok": False, "error": "Acceleration is zero; velocity does not change, so time cannot be determined from this equation."}
        t = (v - u) / a
        steps.append(CalculationStep("Formula", "t = (v - u) / a"))
        steps.append(CalculationStep("Calculate", f"t = ({v} - {u}) / {a} = {t}"))

    # s = u t + 1/2 a t^2  (only when t is known)
    if s is None and u is not None and t is not None and a is not None:
        s = u * t + 0.5 * a * t * t
        steps.append(CalculationStep("Formula", "s = u t + 1/2 a t^2"))
        steps.append(CalculationStep("Calculate", f"s = ({u} x {t}) + 0.5 x {a} x {t}^2 = {s}"))

    steps.append(CalculationStep("Final answer", "Computed missing kinematic quantity(ies)."))
    return _result("kinematics", None, steps, extra={"u": u, "v": v, "a": a, "t": t, "s": s})
