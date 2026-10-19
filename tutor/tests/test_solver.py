"""Unit tests for SymPySolver."""

import pytest
from tutor.core.solver import SymPySolver


@pytest.fixture
def solver():
    return SymPySolver()


def test_solve_linear_equation(solver):
    sol = solver.solve("3*x + 7 = 22", operation="solve", variable="x")
    assert sol.success is True
    assert "x = 5" in sol.final_answer
    assert len(sol.steps) >= 3
    # Check that steps explain isolation and division
    step_titles = [s.title for s in sol.steps]
    assert any("Isolate" in t for t in step_titles)


def test_solve_quadratic_equation(solver):
    sol = solver.solve("x^2 - 5*x + 6 = 0", operation="solve", variable="x")
    assert sol.success is True
    assert "2" in sol.final_answer and "3" in sol.final_answer
    assert len(sol.steps) >= 3
    # Check that discriminant or factoring is mentioned
    step_descs = " ".join(s.description for s in sol.steps)
    assert "discriminant" in step_descs.lower() or "factor" in step_descs.lower()


def test_solve_differentiation(solver):
    sol = solver.solve("x^3 + 2*x^2 - 5*x + 1", operation="diff", variable="x")
    assert sol.success is True
    assert "3*x**2 + 4*x - 5" in sol.final_answer or "3*x^2 + 4*x - 5" in sol.final_answer
    assert len(sol.steps) >= 2


def test_solve_integration(solver):
    sol = solver.solve("3*x^2 + 4*x - 2", operation="integrate", variable="x")
    assert sol.success is True
    assert "+ C" in sol.final_answer
    assert "x**3 + 2*x**2 - 2*x" in sol.final_answer or "x^3 + 2*x^2 - 2*x" in sol.final_answer


def test_solve_simplification(solver):
    sol = solver.solve("(x^2 - 4)/(x - 2)", operation="simplify", variable="x")
    assert sol.success is True
    assert "x + 2" in sol.final_answer


def test_solve_factoring(solver):
    sol = solver.solve("x^2 - 5*x + 6", operation="factor", variable="x")
    assert sol.success is True
    assert "(x - 3)*(x - 2)" in sol.final_answer or "(x - 2)*(x - 3)" in sol.final_answer


def test_invalid_input_handling(solver):
    sol = solver.solve("invalid +++ syntax === ???", operation="auto", variable="x")
    assert sol.success is False
    assert sol.error_message is not None
