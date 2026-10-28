"""Tests for the safe calculation engine (deterministic, no provider involved)."""

import pytest

from app import calc_tools


def test_evaluate_basic_arithmetic():
    assert calc_tools.evaluate_expression("2 + 3 * 4")["ok"] is True
    assert calc_tools.evaluate_expression("2 + 3 * 4")["result"] == 14
    assert calc_tools.evaluate_expression("(10 - 2) / 4")["result"] == 2
    assert calc_tools.evaluate_expression("2 ** 3")["result"] == 8


def test_evaluate_percentage_and_division():
    assert calc_tools.evaluate_expression("100 / 4")["result"] == 25
    assert calc_tools.evaluate_expression("7 + 3")["result"] == 10


def test_evaluate_rejects_dangerous_input():
    assert calc_tools.evaluate_expression("__import__('os')")["ok"] is False
    assert calc_tools.evaluate_expression("open('/etc/passwd')")["ok"] is False
    assert calc_tools.evaluate_expression("1/0")["ok"] is False
    assert calc_tools.evaluate_expression("")["ok"] is False


def test_simple_interest():
    result = calc_tools.simple_interest(10000, 5, 2)
    assert result["ok"] is True
    assert result["result"] == 1000
    assert any("1000" in s["detail"] for s in result["steps"])
    assert result["tool"] == "simple interest"


def test_compound_interest():
    result = calc_tools.compound_interest(1000, 10, 2, compounds_per_year=1)
    assert result["ok"] is True
    # 1000 * 1.1^2 = 1210, interest = 210
    assert abs(result["result"] - 210) < 1e-6
    assert abs(result["amount"] - 1210) < 1e-6


def test_linear_equation():
    result = calc_tools.linear_equation(2, 4, 12)
    assert result["ok"] is True
    assert result["result"] == 4  # 2x + 4 = 12 -> x = 4


def test_linear_equation_no_solution():
    result = calc_tools.linear_equation(0, 5, 10)
    assert result["ok"] is True
    assert result["none"] is True


def test_quadratic_equation_two_roots():
    result = calc_tools.quadratic_equation(1, -3, 2)
    assert result["ok"] is True
    roots = sorted(result["roots"])
    assert roots == [1, 2]


def test_quadratic_equation_no_real_roots():
    result = calc_tools.quadratic_equation(1, 0, 1)
    assert result["ok"] is True
    assert result["real_roots"] is False


def test_percentage_calculation():
    out = calc_tools.evaluate_expression("200 * 15 / 100")
    assert out["ok"] is True
    assert out["result"] == 30


def test_unit_conversion():
    result = calc_tools.convert_units(10, "km", "m")
    assert result["ok"] is True
    assert result["result"] == 10000
    result = calc_tools.convert_units(5, "mi", "km")
    assert result["ok"] is True
    assert abs(result["result"] - 8.04672) < 1e-4


def test_unit_conversion_unsupported():
    result = calc_tools.convert_units(10, "kg", "m")
    assert result["ok"] is False
