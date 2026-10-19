"""Comprehensive regression test suite for MathLab AI.

Covers:
1. Differentiation bug fixes (natural language queries, trig/exp product forms, variable extraction).
2. Equation solving & implicit multiplication parsing (e.g. 3(2x - 4) + 5 = 2(x + 7) + 9).
3. Core single-variable operations (simplification, factorisation, expansion, integration).
4. Matplotlib graphing pipeline (dual derivative plots, shaded definite integrals, function roots).
"""

import pytest
import sympy as sp
from tutor.core.parser import MathParser
from tutor.core.solver import SymPySolver
from tutor.core.graphing import MathPlotter


@pytest.fixture
def solver():
    return SymPySolver()


class TestDifferentiationRegression:
    """Tests differentiation query parsing and symbolic differentiation."""

    def test_differentiate_with_respect_to_x(self, solver):
        """Test: 'Differentiate y = x^3 sin(x) with respect to x'."""
        sol = solver.solve("Differentiate y = x^3 sin(x) with respect to x")
        assert sol.success is True
        assert sol.operation == "diff"
        assert sol.variable == "x"
        # Verify the symbolic derivative equivalence
        x = sp.Symbol("x")
        expected = 3 * x**2 * sp.sin(x) + x**3 * sp.cos(x)
        assert sp.simplify(sol.raw_result - expected) == 0
        assert "3*x**2*sin(x)" in sol.final_answer or "x**3*cos(x)" in sol.final_answer

    def test_find_dy_dx_for_y(self, solver):
        """Test: 'Find dy/dx for y = x^2 + 3x'."""
        sol = solver.solve("Find dy/dx for y = x^2 + 3x")
        assert sol.success is True
        x = sp.Symbol("x")
        assert sp.simplify(sol.raw_result - (2 * x + 3)) == 0

    def test_derivative_of_trig_exp(self, solver):
        """Test: 'Derivative of sin(x)*exp(x)'."""
        sol = solver.solve("Derivative of sin(x)*exp(x)")
        assert sol.success is True
        x = sp.Symbol("x")
        expected = sp.exp(x) * sp.sin(x) + sp.exp(x) * sp.cos(x)
        assert sp.simplify(sol.raw_result - expected) == 0

    def test_differentiate_polynomial(self, solver):
        """Test: 'Differentiate x^2 + 3x - 5' and 'Differentiate x^3'."""
        sol1 = solver.solve("Differentiate x^2 + 3x - 5")
        assert sol1.success is True
        x = sp.Symbol("x")
        assert sp.simplify(sol1.raw_result - (2 * x + 3)) == 0

        sol2 = solver.solve("Differentiate x^3")
        assert sol2.success is True
        assert sp.simplify(sol2.raw_result - (3 * x**2)) == 0


class TestEquationSolvingAndAlgebraParsing:
    """Tests equation solving with implicit multiplication and brackets."""

    def test_equation_with_implicit_multiplication(self, solver):
        """Test: '3(2x - 4) + 5 = 2(x + 7) + 9'."""
        sol = solver.solve("3(2x - 4) + 5 = 2(x + 7) + 9")
        assert sol.success is True
        assert sol.operation == "solve"
        # Check simplified intermediate step (6x - 7 = 2x + 23)
        step_texts = " ".join([s.description + " " + s.expression for s in sol.steps])
        assert "6*x - 7" in step_texts or "6x - 7" in step_texts
        assert "2*x + 23" in step_texts or "2x + 23" in step_texts
        # Check final answer x = 15/2 = 7.5
        assert "15/2" in sol.final_answer
        assert "7.5" in sol.final_answer

    def test_implicit_multiplication_patterns(self):
        """Verify parser handles implicit multiplication across forms."""
        test_cases = [
            ("3(2x - 4)", 3 * (2 * sp.Symbol("x") - 4)),
            ("2(x + 7)", 2 * (sp.Symbol("x") + 7)),
            ("2x + 3", 2 * sp.Symbol("x") + 3),
            ("x(x + 2)", sp.Symbol("x") * (sp.Symbol("x") + 2)),
            ("x^2 + 3x + 2", sp.Symbol("x")**2 + 3 * sp.Symbol("x") + 2),
            ("sqrt(x)", sp.sqrt(sp.Symbol("x"))),
            ("sin(x)", sp.sin(sp.Symbol("x"))),
        ]
        for raw_str, expected_expr in test_cases:
            res = MathParser.parse(raw_str)
            assert res.success is True, f"Failed on {raw_str}: {res.error_message}"
            assert sp.simplify(res.parsed_object - expected_expr) == 0


class TestCoreOperations:
    """Tests simplification, factorisation, expansion, and integration."""

    def test_algebraic_expansion(self, solver):
        """Test: 'Expand (x + 2)(x + 3)'."""
        sol = solver.solve("Expand (x + 2)(x + 3)", operation="expand")
        assert sol.success is True
        x = sp.Symbol("x")
        assert sp.simplify(sol.raw_result - (x**2 + 5 * x + 6)) == 0

    def test_algebraic_simplification(self, solver):
        """Test: 'Simplify (x^2 - 1)/(x - 1)'."""
        sol = solver.solve("Simplify (x^2 - 1)/(x - 1)", operation="simplify")
        assert sol.success is True
        x = sp.Symbol("x")
        assert sp.simplify(sol.raw_result - (x + 1)) == 0

    def test_algebraic_factorisation(self, solver):
        """Test: 'Factor x^2 + 5x + 6'."""
        sol = solver.solve("Factor x^2 + 5x + 6", operation="factor")
        assert sol.success is True
        assert "(x + 2)" in sol.final_answer
        assert "(x + 3)" in sol.final_answer

    def test_indefinite_integral(self, solver):
        """Test: 'Integrate x^2'."""
        sol = solver.solve("Integrate x^2", operation="integrate")
        assert sol.success is True
        assert "+ C" in sol.final_answer
        x = sp.Symbol("x")
        assert sp.simplify(sol.raw_result - (x**3 / 3)) == 0

    def test_definite_integral(self, solver):
        """Test: 'Integrate x^2 from 0 to 2'."""
        sol = solver.solve("Integrate x^2 from 0 to 2", operation="integrate", limits=(0, 2))
        assert sol.success is True
        assert "8/3" in sol.final_answer
        assert sp.simplify(sol.raw_result - sp.Rational(8, 3)) == 0


class TestGraphingVisualisation:
    """Tests Matplotlib figure generation without rendering to screen."""

    def test_plot_derivative(self):
        fig = MathPlotter.plot_derivative("x^3 * sin(x)", "x", x_range=[-3, 3])
        assert fig is not None
        assert len(fig.axes) == 1
        # Both f(x) and f'(x) curves plotted
        assert len(fig.axes[0].lines) >= 2

    def test_plot_definite_integral(self):
        fig = MathPlotter.plot_integral("x**2", "x", limits=(0, 2), x_range=[-1, 3])
        assert fig is not None
        assert len(fig.axes) == 1
        # Shaded region exists
        assert len(fig.axes[0].collections) > 0

    def test_plot_function_with_roots(self):
        fig = MathPlotter.plot_function("x**2 - 4", "x", x_range=[-3, 3])
        assert fig is not None
        assert len(fig.axes) == 1
