"""SymPy Ground-Truth Solver with Pedagogical Step Extraction.

Parses mathematical problems (equations, expressions, derivatives, integrals)
and generates granular, step-by-step solutions with mathematical rules and explanations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tutor.core.parser import MathParser

logger = logging.getLogger(__name__)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class Step:
    """A single pedagogical step in solving a math problem."""
    step_number: int
    title: str
    description: str
    expression: str
    latex: str
    rule_applied: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "expression": self.expression,
            "latex": self.latex,
            "rule_applied": self.rule_applied,
        }


@dataclass
class TutorSolution:
    """Complete solution containing metadata and ordered steps."""
    problem: str
    operation: str
    variable: str
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    final_answer_latex: str = ""
    raw_result: Any = None
    success: bool = True
    error_message: Optional[str] = None
    original_expr: Any = None
    limits: Optional[Tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "operation": self.operation,
            "variable": self.variable,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "final_answer_latex": self.final_answer_latex,
            "success": self.success,
            "error_message": self.error_message,
        }


class SymPySolver:
    """Ground-truth mathematical solver leveraging SymPy and MathParser."""

    def __init__(self, default_var: str = "x") -> None:
        self.default_var = default_var

    def clean_input(self, raw: str) -> str:
        """Sanitize raw string notation using MathParser."""
        return MathParser.sanitize_math_string(raw)

    def parse(self, raw: str, variable: str = "x") -> Tuple[Any, sp.Symbol, bool]:
        """Parse raw mathematical input using MathParser.

        Returns:
            Tuple of (parsed_object, symbol, is_equation)
        """
        parsed_res = MathParser.parse(raw, default_variable=variable)
        if not parsed_res.success:
            raise ValueError(parsed_res.error_message or f"Unable to parse input: {raw}")
        return parsed_res.parsed_object, sp.Symbol(parsed_res.variable), parsed_res.is_equation

    def solve(
        self,
        problem: str,
        operation: str = "auto",
        variable: str = "x",
        limits: Optional[Tuple[float, float]] = None,
    ) -> TutorSolution:
        """Solve a math problem and produce step-by-step pedagogical explanations.

        Args:
            problem: The mathematical problem string.
            operation: One of 'auto', 'solve', 'simplify', 'factor', 'expand', 'diff', 'integrate'.
            variable: The independent variable symbol (default 'x').
            limits: Optional limits for definite integration (a, b).

        Returns:
            TutorSolution containing the sequence of steps and final answer.
        """
        # First check if this is a known word-problem template
        word_sol = self._try_solve_word_problem(problem, variable)
        if word_sol is not None:
            return word_sol

        try:
            # Parse input with the unified MathParser
            parsed_res = MathParser.parse(
                raw_query=problem,
                default_operation=operation,
                default_variable=variable,
            )

            if not parsed_res.success:
                return TutorSolution(
                    problem=problem,
                    operation=operation,
                    variable=variable,
                    success=False,
                    error_message=parsed_res.error_message,
                    final_answer="Could not parse problem.",
                    final_answer_latex=r"\text{Could not parse problem}",
                )

            parsed_obj = parsed_res.parsed_object
            var_sym = sp.Symbol(parsed_res.variable)
            is_equation = parsed_res.is_equation
            op = operation.lower().strip() if operation.lower().strip() != "auto" else (parsed_res.detected_operation or "auto")

            if op == "auto":
                op = "solve" if is_equation else "simplify"

            active_limits = limits or parsed_res.limits

            if op == "solve" or (is_equation and op not in ("diff", "integrate", "simplify", "expand", "factor")):
                return self._solve_equation(parsed_obj, var_sym, problem)
            elif op in ("diff", "differentiate"):
                expr = parsed_obj.lhs - parsed_obj.rhs if isinstance(parsed_obj, sp.Eq) else parsed_obj
                return self._solve_derivative(expr, var_sym, problem)
            elif op in ("integrate", "integral"):
                expr = parsed_obj.lhs - parsed_obj.rhs if isinstance(parsed_obj, sp.Eq) else parsed_obj
                return self._solve_integral(expr, var_sym, problem, limits=active_limits)
            elif op == "expand":
                expr = parsed_obj.lhs - parsed_obj.rhs if isinstance(parsed_obj, sp.Eq) else parsed_obj
                return self._solve_expand(expr, var_sym, problem)
            elif op == "factor":
                expr = parsed_obj.lhs - parsed_obj.rhs if isinstance(parsed_obj, sp.Eq) else parsed_obj
                return self._solve_factor(expr, var_sym, problem)
            elif op == "simplify":
                expr = parsed_obj.lhs - parsed_obj.rhs if isinstance(parsed_obj, sp.Eq) else parsed_obj
                return self._solve_simplify(expr, var_sym, problem)
            else:
                return self._solve_equation(parsed_obj, var_sym, problem)

        except Exception as e:
            logger.debug("SymPy solve failed for '%s': %s", problem, e)
            return TutorSolution(
                problem=problem,
                operation=operation,
                variable=variable,
                success=False,
                error_message=str(e),
                final_answer="Could not solve problem symbolically.",
                final_answer_latex=r"\text{Could not solve symbolically}",
            )

    def _try_solve_word_problem(self, problem: str, variable: str) -> Optional[TutorSolution]:
        """Recognize and solve common arithmetic / algebra word problem templates."""
        p_lower = problem.lower().strip()

        # Pattern 1: Sum from 1 to N: "sum of all natural numbers from 1 to 390" or "Find 1 + 2 + ... + 97"
        m1 = re.search(r"(?:1\s*\+\s*2\s*\+\s*\.\.\.\s*\+\s*(\d+)|from\s+1\s+to\s+(\d+))", p_lower)
        if m1:
            n_str = m1.group(1) or m1.group(2)
            n = int(n_str)
            ans = (n * (n + 1)) // 2
            steps = [
                Step(
                    step_number=1,
                    title="Identify Problem Type",
                    description=f"This is an arithmetic series: the sum of the first $n = {n}$ natural numbers.",
                    expression=f"S_n = (n*(n + 1))/2 for n = {n}",
                    latex=rf"S_n = \frac{{n(n+1)}}{{2}} \quad \text{{with}} \quad n = {n}",
                    rule_applied="Arithmetic Series Formula (Gauss Sum)",
                ),
                Step(
                    step_number=2,
                    title="Substitute Value into Formula",
                    description=f"Substitute $n = {n}$ into $\\frac{{{n}({n} + 1)}}{{2}} = \\frac{{{n} \\times {n + 1}}}{{2}}$.",
                    expression=f"({n} * {n + 1}) / 2 = {ans}",
                    latex=rf"\frac{{{n} \times {n + 1}}}{{2}} = {ans}",
                    rule_applied="Substitution and Computation",
                ),
            ]
            return TutorSolution(
                problem=problem,
                operation="solve",
                variable=variable,
                steps=steps,
                final_answer=str(ans),
                final_answer_latex=str(ans),
                raw_result=ans,
                success=True,
            )

        # Pattern 2: Sum of digits: "sum of all decimal digits in 84049" or "digit-sum of 176"
        m2 = re.search(r"(?:digit-sum|sum of all decimal digits|sum of digits)\s*(?:in|of)?\s*(\d+)", p_lower)
        if m2:
            num_str = m2.group(1)
            digits = [int(d) for d in num_str]
            digit_sum = sum(digits)
            steps = [
                Step(
                    step_number=1,
                    title="Break Down Number into Decimal Digits",
                    description=f"The digits of `{num_str}` are: {', '.join(str(d) for d in digits)}.",
                    expression=" + ".join(str(d) for d in digits),
                    latex=" + ".join(str(d) for d in digits),
                    rule_applied="Decimal Digit Extraction",
                ),
                Step(
                    step_number=2,
                    title="Compute the Sum",
                    description=f"Add the individual digits together: {' + '.join(str(d) for d in digits)} = {digit_sum}.",
                    expression=str(digit_sum),
                    latex=str(digit_sum),
                    rule_applied="Addition",
                ),
            ]
            return TutorSolution(
                problem=problem,
                operation="solve",
                variable=variable,
                steps=steps,
                final_answer=str(digit_sum),
                final_answer_latex=str(digit_sum),
                raw_result=digit_sum,
                success=True,
            )

        # Pattern 3: Last / units digit of N: "last (units) digit of 329"
        m3 = re.search(r"last\s*(?:\(units\))?\s*digit\s*of\s*(\d+)", p_lower)
        if m3:
            num_str = m3.group(1)
            last_digit = int(num_str[-1])
            steps = [
                Step(
                    step_number=1,
                    title="Determine the Units Digit",
                    description=f"The units (last) digit corresponds to $N \\pmod{{10}}$. For ${num_str}$, the last digit is ${last_digit}$.",
                    expression=f"{num_str} % 10 = {last_digit}",
                    latex=rf"{num_str} \pmod{{10}} = {last_digit}",
                    rule_applied="Modulo 10 Arithmetic",
                )
            ]
            return TutorSolution(
                problem=problem,
                operation="solve",
                variable=variable,
                steps=steps,
                final_answer=str(last_digit),
                final_answer_latex=str(last_digit),
                raw_result=last_digit,
                success=True,
            )

        # Pattern 4: Sum of K consecutive numbers starting at A: "sum of 7 consecutive natural numbers starting at 9"
        m4 = re.search(r"sum\s*of\s*(\d+)\s*consecutive\s*natural\s*numbers\s*starting\s*at\s*(\d+)", p_lower)
        if m4:
            count = int(m4.group(1))
            start = int(m4.group(2))
            # S = count * (2*start + count - 1) / 2
            tot = count * (2 * start + count - 1) // 2
            steps = [
                Step(
                    step_number=1,
                    title="Formulate Consecutive Sum",
                    description=(
                        f"The sum of $n = {count}$ consecutive numbers starting at $a = {start}$ "
                        f"is an arithmetic progression with first term $a = {start}$, difference $d = 1$, and $n = {count}$ terms."
                    ),
                    expression=f"S = n/2 * (2*a + (n-1)*d)",
                    latex=rf"S = \frac{{n}}{{2}}\left[2a + (n-1)d\right]",
                    rule_applied="Arithmetic Progression Sum",
                ),
                Step(
                    step_number=2,
                    title="Calculate Result",
                    description=f"Substitute $a = {start}$, $n = {count}$, and $d = 1$: $\\frac{{{count}}}{{2}}[2({start}) + ({count}-1)] = {tot}$.",
                    expression=str(tot),
                    latex=str(tot),
                    rule_applied="Computation",
                ),
            ]
            return TutorSolution(
                problem=problem,
                operation="solve",
                variable=variable,
                steps=steps,
                final_answer=str(tot),
                final_answer_latex=str(tot),
                raw_result=tot,
                success=True,
            )

        return None

    def _solve_equation(self, eq: Union[sp.Eq, sp.Expr], var: sp.Symbol, problem: str) -> TutorSolution:
        """Step-by-step solver for algebraic equations."""
        if not isinstance(eq, sp.Eq):
            eq = sp.Eq(eq, 0)

        lhs, rhs = eq.lhs, eq.rhs
        standard_expr = sp.simplify(lhs - rhs)
        degree = sp.degree(standard_expr, gen=var) if standard_expr.is_polynomial(var) else None

        steps: List[Step] = []
        step_num = 1

        # Step 1: Stating problem
        steps.append(
            Step(
                step_number=step_num,
                title="Identify the Equation",
                description=f"Start with the given equation: `{sp.sstr(eq)}`.",
                expression=sp.sstr(eq),
                latex=sp.latex(eq),
                rule_applied="Problem Statement",
            )
        )
        step_num += 1

        # Step 2: Simplify both sides if brackets can be expanded
        exp_lhs = sp.expand(lhs)
        exp_rhs = sp.expand(rhs)
        if exp_lhs != lhs or exp_rhs != rhs:
            simp_eq = sp.Eq(exp_lhs, exp_rhs)
            steps.append(
                Step(
                    step_number=step_num,
                    title="Simplify Both Sides",
                    description=f"Expand brackets and combine like terms on both sides: `{sp.sstr(exp_lhs)} = {sp.sstr(exp_rhs)}`.",
                    expression=f"{sp.sstr(exp_lhs)} = {sp.sstr(exp_rhs)}",
                    latex=rf"{sp.latex(exp_lhs)} = {sp.latex(exp_rhs)}",
                    rule_applied="Distributive Law & Simplification",
                )
            )
            step_num += 1

        # Step 3: Standard form
        if rhs != 0:
            steps.append(
                Step(
                    step_number=step_num,
                    title="Rearrange to Standard Form",
                    description=f"Subtract `{sp.sstr(rhs)}` from both sides so all terms are on the left-hand side.",
                    expression=f"{sp.sstr(standard_expr)} = 0",
                    latex=f"{sp.latex(standard_expr)} = 0",
                    rule_applied="Additive Inverse Property",
                )
            )
            step_num += 1

        roots = sp.solve(eq, var)

        if degree == 1:
            # Linear equation: a*x + b = 0
            poly = sp.Poly(standard_expr, var)
            coeffs = poly.all_coeffs()
            a, b = coeffs[0], coeffs[1]

            steps.append(
                Step(
                    step_number=step_num,
                    title="Isolate the Variable Term",
                    description=f"Subtract the constant term `{sp.sstr(b)}` from both sides.",
                    expression=f"{sp.sstr(a * var)} = {sp.sstr(-b)}",
                    latex=f"{sp.latex(a * var)} = {sp.latex(-b)}",
                    rule_applied="Subtraction Property of Equality",
                )
            )
            step_num += 1

            if a != 1:
                sol_val = -b / a
                steps.append(
                    Step(
                        step_number=step_num,
                        title="Divide by the Coefficient",
                        description=f"Divide both sides by the coefficient of `{var}` (which is `{sp.sstr(a)}`).",
                        expression=f"{var} = {sp.sstr(sol_val)}",
                        latex=f"{sp.latex(var)} = {sp.latex(sol_val)}",
                        rule_applied="Division Property of Equality",
                    )
                )
                step_num += 1

        elif degree == 2:
            # Quadratic equation: a*x^2 + b*x + c = 0
            poly = sp.Poly(standard_expr, var)
            coeffs = poly.all_coeffs()
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            discriminant = b**2 - 4 * a * c

            steps.append(
                Step(
                    step_number=step_num,
                    title="Identify Coefficients and Discriminant",
                    description=(
                        f"In the quadratic form $ax^2 + bx + c = 0$, we have $a = {sp.sstr(a)}$, "
                        f"$b = {sp.sstr(b)}$, and $c = {sp.sstr(c)}$.\n"
                        f"Compute the discriminant $\\Delta = b^2 - 4ac = ({sp.sstr(b)})^2 - 4({sp.sstr(a)})({sp.sstr(c)}) = {sp.sstr(discriminant)}$."
                    ),
                    expression=f"Delta = {sp.sstr(discriminant)}",
                    latex=rf"\Delta = {sp.latex(discriminant)}",
                    rule_applied="Quadratic Discriminant Definition",
                )
            )
            step_num += 1

            # Factor if possible over rationals
            factored = sp.factor(standard_expr)
            if factored != standard_expr and not factored.is_Number:
                steps.append(
                    Step(
                        step_number=step_num,
                        title="Factor the Quadratic Expression",
                        description=f"Factor the quadratic polynomial into linear factors: `{sp.sstr(factored)} = 0`.",
                        expression=f"{sp.sstr(factored)} = 0",
                        latex=f"{sp.latex(factored)} = 0",
                        rule_applied="Factoring / Zero Product Property",
                    )
                )
                step_num += 1
            else:
                steps.append(
                    Step(
                        step_number=step_num,
                        title="Apply the Quadratic Formula",
                        description=(
                            rf"Use the quadratic formula: $x = \frac{{-b \pm \sqrt{{\Delta}}}}{{2a}} = "
                            rf"\frac{{-({sp.sstr(b)}) \pm \sqrt{{{sp.sstr(discriminant)}}}}}{{2({sp.sstr(a)})}}$."
                        ),
                        expression=f"x = (-({b}) +- sqrt({discriminant})) / (2*{a})",
                        latex=rf"{sp.latex(var)} = \frac{{-{sp.latex(b)} \pm \sqrt{{{sp.latex(discriminant)}}}}}{{2 \cdot {sp.latex(a)}}}",
                        rule_applied="Quadratic Formula",
                    )
                )
                step_num += 1

        else:
            # Generic polynomial or transcendental equation
            factored = sp.factor(standard_expr)
            if factored != standard_expr:
                steps.append(
                    Step(
                        step_number=step_num,
                        title="Factor Expression",
                        description=f"Factor the expression to find roots: `{sp.sstr(factored)} = 0`.",
                        expression=f"{sp.sstr(factored)} = 0",
                        latex=f"{sp.latex(factored)} = 0",
                        rule_applied="Polynomial Factoring",
                    )
                )
                step_num += 1

        ans_strs = []
        ans_latex_parts = []
        for r in roots:
            if getattr(r, "is_rational", False) and not getattr(r, "is_integer", False):
                flt = float(r)
                flt_str = f"{flt:.4g}"
                ans_strs.append(f"{var} = {sp.sstr(r)} = {flt_str}")
                ans_latex_parts.append(rf"{sp.latex(var)} = {sp.latex(r)} = {flt_str}")
            else:
                ans_strs.append(f"{var} = {sp.sstr(r)}")
                ans_latex_parts.append(rf"{sp.latex(var)} = {sp.latex(r)}")

        if not ans_strs:
            ans_strs = ["No solution found"]
            ans_latex = r"\text{No solution}"
        else:
            ans_latex = ", ".join(ans_latex_parts)

        steps.append(
            Step(
                step_number=step_num,
                title="Final Solution",
                description=f"The resulting solution(s) for `{var}` are: {', '.join(ans_strs)}.",
                expression=", ".join(ans_strs),
                latex=ans_latex,
                rule_applied="Solution Set",
            )
        )

        return TutorSolution(
            problem=problem,
            operation="solve",
            variable=str(var),
            steps=steps,
            final_answer=", ".join(ans_strs),
            final_answer_latex=ans_latex,
            raw_result=roots,
            original_expr=standard_expr,
            success=True,
        )

    def _solve_derivative(self, expr: sp.Expr, var: sp.Symbol, problem: str) -> TutorSolution:
        """Step-by-step differentiation with term-by-term calculus rules."""
        steps: List[Step] = []
        step_num = 1

        steps.append(
            Step(
                step_number=step_num,
                title="Identify Function to Differentiate",
                description=f"We need to compute the derivative of $f({var}) = {sp.sstr(expr)}$ with respect to `{var}`.",
                expression=f"d/d{var}({sp.sstr(expr)})",
                latex=rf"\frac{{d}}{{d{sp.latex(var)}}}\left[{sp.latex(expr)}\right]",
                rule_applied="Derivative Definition",
            )
        )
        step_num += 1

        expanded = sp.expand(expr)
        terms = sp.Add.make_args(expanded)

        if len(terms) > 1:
            term_derivatives = []
            term_details = []
            for term in terms:
                d_term = sp.diff(term, var)
                term_derivatives.append(d_term)
                # Determine rule
                coeff, power = term.as_coeff_exponent(var)
                if power == 0:
                    rule = "Constant Rule: d/dx[c] = 0"
                elif power == 1:
                    rule = f"Linear Rule: d/dx[c*x] = c ({coeff})"
                else:
                    rule = f"Power Rule: d/dx[x^{power}] = {power}*x^{power - 1}"
                term_details.append(f"- Term `{sp.sstr(term)}`: derivative is `{sp.sstr(d_term)}` ({rule})")

            steps.append(
                Step(
                    step_number=step_num,
                    title="Apply Sum and Power Rules Term-by-Term",
                    description=(
                        f"By linearity of the derivative, differentiate each term separately:\n"
                        + "\n".join(term_details)
                    ),
                    expression=" + ".join(f"({sp.sstr(dt)})" for dt in term_derivatives),
                    latex=" + ".join(rf"\left({sp.latex(dt)}\right)" for dt in term_derivatives),
                    rule_applied="Sum / Difference & Power Rule",
                )
            )
            step_num += 1

        result = sp.diff(expr, var)

        final_ans_str = f"dy/d{var} = {sp.sstr(result)}"
        final_ans_latex = rf"\frac{{dy}}{{d{sp.latex(var)}}} = {sp.latex(result)}"

        steps.append(
            Step(
                step_number=step_num,
                title="Final Derivative",
                description=f"The derivative of the function is: `{final_ans_str}`.",
                expression=final_ans_str,
                latex=final_ans_latex,
                rule_applied="Final Result",
            )
        )

        return TutorSolution(
            problem=problem,
            operation="diff",
            variable=str(var),
            steps=steps,
            final_answer=final_ans_str,
            final_answer_latex=final_ans_latex,
            raw_result=result,
            original_expr=expr,
            success=True,
        )

    def _solve_integral(
        self,
        expr: sp.Expr,
        var: sp.Symbol,
        problem: str,
        limits: Optional[Tuple[Any, Any]] = None,
    ) -> TutorSolution:
        """Step-by-Step integration with indefinite and definite support."""
        steps: List[Step] = []
        step_num = 1

        if limits is not None:
            a_val, b_val = limits
            # Convert float to int if integer value (e.g. 0.0 -> 0, 2.0 -> 2)
            try:
                a_sym = sp.Integer(int(a_val)) if float(a_val).is_integer() else sp.sympify(a_val)
                b_sym = sp.Integer(int(b_val)) if float(b_val).is_integer() else sp.sympify(b_val)
            except Exception:
                a_sym = sp.sympify(a_val)
                b_sym = sp.sympify(b_val)

            steps.append(
                Step(
                    step_number=step_num,
                    title="Identify Definite Integral",
                    description=f"Compute the definite integral $\\int_{{{sp.sstr(a_sym)}}}^{{{sp.sstr(b_sym)}}} \\left({sp.sstr(expr)}\\right) d{var}$.",
                    expression=f"integrate({sp.sstr(expr)}, ({var}, {sp.sstr(a_sym)}, {sp.sstr(b_sym)}))",
                    latex=rf"\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} \left({sp.latex(expr)}\right) d{sp.latex(var)}",
                    rule_applied="Definite Integral Definition",
                )
            )
            step_num += 1

            # Antiderivative
            antiderivative = sp.integrate(expr, var)
            steps.append(
                Step(
                    step_number=step_num,
                    title="Find Antiderivative F(x)",
                    description=f"Find the indefinite integral $F({var}) = {sp.sstr(antiderivative)}$.",
                    expression=f"F({var}) = {sp.sstr(antiderivative)}",
                    latex=rf"F({sp.latex(var)}) = {sp.latex(antiderivative)}",
                    rule_applied="Fundamental Theorem of Calculus",
                )
            )
            step_num += 1

            # Definite evaluation
            def_val = sp.integrate(expr, (var, a_sym, b_sym))
            val_str = sp.sstr(def_val)
            val_latex = sp.latex(def_val)
            if getattr(def_val, "is_rational", False) and not getattr(def_val, "is_integer", False):
                flt_val = float(def_val)
                final_ans_str = f"{val_str} = {flt_val:.4g}"
                final_ans_latex = rf"{val_latex} \approx {flt_val:.4g}"
            else:
                final_ans_str = val_str
                final_ans_latex = val_latex

            steps.append(
                Step(
                    step_number=step_num,
                    title="Evaluate at Limits [F(b) - F(a)]",
                    description=f"Substitute upper and lower limits: $F({sp.sstr(b_sym)}) - F({sp.sstr(a_sym)}) = {final_ans_str}$.",
                    expression=final_ans_str,
                    latex=final_ans_latex,
                    rule_applied="Limits Substitution",
                )
            )

            return TutorSolution(
                problem=problem,
                operation="integrate",
                variable=str(var),
                steps=steps,
                final_answer=final_ans_str,
                final_answer_latex=final_ans_latex,
                raw_result=def_val,
                original_expr=expr,
                limits=(float(a_sym), float(b_sym)),
                success=True,
            )

        # Indefinite integral
        steps.append(
            Step(
                step_number=step_num,
                title="Identify Integrand",
                description=f"We need to find the indefinite integral $\\int \\left({sp.sstr(expr)}\\right) d{var}$.",
                expression=f"int({sp.sstr(expr)}, {var})",
                latex=rf"\int \left({sp.latex(expr)}\right) d{sp.latex(var)}",
                rule_applied="Indefinite Integral Definition",
            )
        )
        step_num += 1

        expanded = sp.expand(expr)
        terms = sp.Add.make_args(expanded)

        if len(terms) > 1:
            term_integrals = []
            term_details = []
            for term in terms:
                int_term = sp.integrate(term, var)
                term_integrals.append(int_term)
                coeff, power = term.as_coeff_exponent(var)
                if power == 0:
                    rule = f"Constant Rule: int({coeff}) dx = {coeff}*x"
                elif power == -1:
                    rule = "Logarithmic Rule: int(1/x) dx = ln|x|"
                else:
                    rule = f"Power Rule: int(x^{power}) dx = (x^{power + 1})/{power + 1}"
                term_details.append(f"- Term `{sp.sstr(term)}`: integral is `{sp.sstr(int_term)}` ({rule})")

            steps.append(
                Step(
                    step_number=step_num,
                    title="Integrate Term-by-Term (Linearity)",
                    description=(
                        f"Integrate each term using standard integration formulas:\n"
                        + "\n".join(term_details)
                    ),
                    expression=" + ".join(f"({sp.sstr(it)})" for it in term_integrals),
                    latex=" + ".join(rf"\left({sp.latex(it)}\right)" for it in term_integrals),
                    rule_applied="Linearity & Power Rule for Integration",
                )
            )
            step_num += 1

        antiderivative = sp.integrate(expr, var)
        final_str = f"{sp.sstr(antiderivative)} + C"
        final_latex = rf"{sp.latex(antiderivative)} + C"

        steps.append(
            Step(
                step_number=step_num,
                title="Add Constant of Integration",
                description="Combine the terms and append the constant of integration $+ C$.",
                expression=final_str,
                latex=final_latex,
                rule_applied="Indefinite Integral Constant (+ C)",
            )
        )

        return TutorSolution(
            problem=problem,
            operation="integrate",
            variable=str(var),
            steps=steps,
            final_answer=final_str,
            final_answer_latex=final_latex,
            raw_result=antiderivative,
            original_expr=expr,
            limits=None,
            success=True,
        )

    def _solve_expand(self, expr: sp.Expr, var: sp.Symbol, problem: str) -> TutorSolution:
        """Step-by-step expansion of polynomial products and powers."""
        steps: List[Step] = []
        step_num = 1

        steps.append(
            Step(
                step_number=step_num,
                title="Identify Expression to Expand",
                description=f"Start with the unexpanded expression: `{sp.sstr(expr)}`.",
                expression=sp.sstr(expr),
                latex=sp.latex(expr),
                rule_applied="Problem Statement",
            )
        )
        step_num += 1

        expanded = sp.expand(expr)
        steps.append(
            Step(
                step_number=step_num,
                title="Apply Distributive Law / FOIL",
                description=f"Multiply factors and expand all brackets: `{sp.sstr(expanded)}`.",
                expression=sp.sstr(expanded),
                latex=sp.latex(expanded),
                rule_applied="Distributive Law",
            )
        )
        step_num += 1

        return TutorSolution(
            problem=problem,
            operation="expand",
            variable=str(var),
            steps=steps,
            final_answer=sp.sstr(expanded),
            final_answer_latex=sp.latex(expanded),
            raw_result=expanded,
            original_expr=expr,
            success=True,
        )

    def _solve_simplify(self, expr: sp.Expr, var: sp.Symbol, problem: str) -> TutorSolution:
        """Step-by-step simplification of algebraic expressions."""
        steps: List[Step] = []
        step_num = 1

        steps.append(
            Step(
                step_number=step_num,
                title="Given Expression",
                description=f"Analyze the expression: `{sp.sstr(expr)}`.",
                expression=sp.sstr(expr),
                latex=sp.latex(expr),
                rule_applied="Problem Statement",
            )
        )
        step_num += 1

        numer, denom = expr.as_numer_denom()
        if denom != 1:
            factored_numer = sp.factor(numer)
            factored_denom = sp.factor(denom)
            if factored_numer != numer or factored_denom != denom:
                steps.append(
                    Step(
                        step_number=step_num,
                        title="Factor Numerator and Denominator",
                        description=(
                            f"Factor both parts of the rational expression:\n"
                            f"- Numerator: `{sp.sstr(numer)}` = `{sp.sstr(factored_numer)}`\n"
                            f"- Denominator: `{sp.sstr(denom)}` = `{sp.sstr(factored_denom)}`"
                        ),
                        expression=f"({sp.sstr(factored_numer)}) / ({sp.sstr(factored_denom)})",
                        latex=rf"\frac{{{sp.latex(factored_numer)}}}{{{sp.latex(factored_denom)}}}",
                        rule_applied="Factoring Rational Expressions",
                    )
                )
                step_num += 1

        simplified = sp.simplify(expr)
        steps.append(
            Step(
                step_number=step_num,
                title="Cancel Common Factors / Simplify",
                description=f"Cancel common factors and combine like terms to obtain the simplest form: `{sp.sstr(simplified)}`.",
                expression=sp.sstr(simplified),
                latex=sp.latex(simplified),
                rule_applied="Algebraic Cancellation",
            )
        )

        return TutorSolution(
            problem=problem,
            operation="simplify",
            variable=str(var),
            steps=steps,
            final_answer=sp.sstr(simplified),
            final_answer_latex=sp.latex(simplified),
            raw_result=simplified,
            success=True,
        )

    def _solve_factor(self, expr: sp.Expr, var: sp.Symbol, problem: str) -> TutorSolution:
        """Step-by-step factoring."""
        steps: List[Step] = []
        step_num = 1

        steps.append(
            Step(
                step_number=step_num,
                title="Given Expression",
                description=f"Expression to factor: `{sp.sstr(expr)}`.",
                expression=sp.sstr(expr),
                latex=sp.latex(expr),
                rule_applied="Problem Statement",
            )
        )
        step_num += 1

        factored = sp.factor(expr)
        steps.append(
            Step(
                step_number=step_num,
                title="Factored Form",
                description=f"Factored into irreducible factors over rationals: `{sp.sstr(factored)}`.",
                expression=sp.sstr(factored),
                latex=sp.latex(factored),
                rule_applied="Polynomial Factoring",
            )
        )

        return TutorSolution(
            problem=problem,
            operation="factor",
            variable=str(var),
            steps=steps,
            final_answer=sp.sstr(factored),
            final_answer_latex=sp.latex(factored),
            raw_result=factored,
            success=True,
        )
