"""Symbolic Step Verifier and Diagnostic Error Classifier.

Compares student-submitted intermediate steps against ground truth using SymPy.
Determines whether a step is mathematically valid, detects specific common errors
(sign errors, arithmetic slips, incorrect operations), and returns diagnostic feedback
without prematurely giving away the final answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tutor.core.solver import SymPySolver

logger = logging.getLogger(__name__)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class StepFeedback:
    """Feedback returned after verifying a student step."""
    is_valid: bool
    step_number: int
    status: str  # 'correct', 'valid_alternative', 'sign_error', 'arithmetic_error', 'operation_error', 'incorrect', 'parse_error'
    message: str
    hint: Optional[str] = None
    student_expression: str = ""
    is_final_solution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "step_number": self.step_number,
            "status": self.status,
            "message": self.message,
            "hint": self.hint,
            "student_expression": self.student_expression,
            "is_final_solution": self.is_final_solution,
        }


class StepVerifier:
    """Verifies student steps symbolically against mathematical problem statements."""

    def __init__(self, solver: Optional[SymPySolver] = None) -> None:
        self.solver = solver or SymPySolver()

    def check_step(
        self,
        problem: str,
        previous_steps: Optional[List[str]],
        student_step: str,
        variable: str = "x",
        operation: str = "auto",
    ) -> StepFeedback:
        """Check whether the student's step is mathematically sound.

        Args:
            problem: The initial problem statement (e.g. '3*x + 7 = 22').
            previous_steps: List of previously accepted student steps (can be empty).
            student_step: The new step submitted by the student.
            variable: The variable in the problem (default 'x').
            operation: Operation type if known.

        Returns:
            StepFeedback containing validity, diagnostic status, and helpful hints.
        """
        step_idx = (len(previous_steps) if previous_steps else 0) + 1

        if not student_step or not student_step.strip():
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="parse_error",
                message="Step is empty. Please enter your mathematical step.",
                student_expression="",
            )

        # 1. Parse ground truth problem
        try:
            prob_parsed, var_sym, prob_is_eq = self.solver.parse(problem, variable=variable)
        except Exception as e:
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="parse_error",
                message=f"Could not parse the original problem: {e}",
                student_expression=student_step,
            )

        # 2. Parse student step
        try:
            step_parsed, _, step_is_eq = self.solver.parse(student_step, variable=variable)
        except Exception as e:
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="parse_error",
                message=(
                    f"Could not parse '{student_step}' as a valid mathematical expression. "
                    "Make sure to use standard symbols like `*` for multiplication and `^` or `**` for powers."
                ),
                student_expression=student_step,
            )

        # 3. Determine base reference (either last accepted step or original problem)
        last_step_str = previous_steps[-1] if (previous_steps and len(previous_steps) > 0) else problem
        try:
            last_parsed, _, last_is_eq = self.solver.parse(last_step_str, variable=variable)
        except Exception:
            last_parsed, last_is_eq = prob_parsed, prob_is_eq

        # 4. Route check based on Equation vs Expression
        if prob_is_eq:
            return self._verify_equation_step(
                prob_eq=prob_parsed,
                last_ref=last_parsed,
                student_obj=step_parsed,
                student_is_eq=step_is_eq,
                var=var_sym,
                student_raw=student_step,
                step_idx=step_idx,
            )
        else:
            return self._verify_expression_step(
                prob_expr=prob_parsed,
                last_ref=last_parsed,
                student_obj=step_parsed,
                student_is_eq=step_is_eq,
                var=var_sym,
                student_raw=student_step,
                step_idx=step_idx,
                operation=operation,
                problem_raw=problem,
            )

    def _verify_equation_step(
        self,
        prob_eq: sp.Eq,
        last_ref: Any,
        student_obj: Any,
        student_is_eq: bool,
        var: sp.Symbol,
        student_raw: str,
        step_idx: int,
    ) -> StepFeedback:
        """Verify steps for an algebraic equation."""
        # Calculate ground truth roots
        prob_roots = set(sp.solve(prob_eq, var))

        if not student_is_eq:
            # Student submitted a single expression instead of an equation.
            # Could they have just submitted the root value, e.g. "5"?
            if isinstance(student_obj, (sp.Number, sp.Expr)):
                if student_obj in prob_roots or any(sp.simplify(student_obj - r) == 0 for r in prob_roots):
                    return StepFeedback(
                        is_valid=True,
                        step_number=step_idx,
                        status="correct",
                        message=f"Spot on! `{student_raw}` is indeed the correct solution.",
                        student_expression=student_raw,
                        is_final_solution=True,
                    )
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="incorrect",
                message=(
                    f"Your step `{student_raw}` is an expression rather than an equation. "
                    f"Please include both sides (e.g. `{var} = ...` or `... = ...`)."
                ),
                hint=f"Write your step as an equality with `{var}`.",
                student_expression=student_raw,
            )

        student_eq = student_obj
        student_diff = sp.simplify(student_eq.lhs - student_eq.rhs)

        # Check solution set equivalence
        try:
            student_roots = set(sp.solve(student_eq, var))
        except Exception:
            student_roots = None

        # Exact algebraic equivalence: is student_diff proportional to original standard equation?
        orig_diff = sp.simplify(prob_eq.lhs - prob_eq.rhs)

        # Direct equivalence test
        is_equivalent = False
        if student_roots is not None and student_roots == prob_roots:
            is_equivalent = True
        elif sp.simplify(orig_diff - student_diff) == 0:
            is_equivalent = True
        elif orig_diff != 0 and student_diff != 0:
            ratio = sp.simplify(student_diff / orig_diff)
            if ratio.is_number and ratio != 0:
                is_equivalent = True

        if is_equivalent:
            # Check if student reached the final solution: var = root
            is_final = False
            if student_roots is not None and len(student_roots) > 0:
                if (student_eq.lhs == var and student_eq.rhs in student_roots) or \
                   (student_eq.rhs == var and student_eq.lhs in student_roots):
                    is_final = True

            msg = "Correct step! That is mathematically equivalent."
            if is_final:
                msg = f"Awesome work! You successfully solved for `{var}`."

            return StepFeedback(
                is_valid=True,
                step_number=step_idx,
                status="correct",
                message=msg,
                student_expression=student_raw,
                is_final_solution=is_final,
            )

        # The step is NOT equivalent: diagnose WHY
        # 1. Sign error diagnosis: Did they add instead of subtract (or vice versa)?
        # Compare student_diff against sign flipped terms
        if isinstance(last_ref, sp.Eq):
            ref_lhs, ref_rhs = last_ref.lhs, last_ref.rhs
        else:
            ref_lhs, ref_rhs = prob_eq.lhs, prob_eq.rhs

        # Common error 1: Moving a term across equals without flipping its sign
        # e.g. 3x + 7 = 22 -> 3x = 22 + 7 = 29
        # Check if student_eq has opposite sign for constant
        poly_orig = sp.Poly(orig_diff, var) if orig_diff.is_polynomial(var) else None
        poly_stud = sp.Poly(student_diff, var) if student_diff.is_polynomial(var) else None

        if poly_orig and poly_stud and poly_orig.degree() == poly_stud.degree():
            orig_coeffs = poly_orig.all_coeffs()
            stud_coeffs = poly_stud.all_coeffs()
            # If variable coefficient matches, but constant is inverted or added
            if len(orig_coeffs) == 2 and len(stud_coeffs) == 2:
                # Linear: a*x + b
                if orig_coeffs[0] == stud_coeffs[0]:
                    # Variable coeff matches, constant differs
                    orig_const = orig_coeffs[1]
                    stud_const = stud_coeffs[1]
                    return StepFeedback(
                        is_valid=False,
                        step_number=step_idx,
                        status="arithmetic_error",
                        message=(
                            "Your variable term is on the right track, but the constant term on the other side "
                            "is incorrect. Double check the arithmetic and sign when moving numbers across the '=' sign."
                        ),
                        hint="Check whether you added or subtracted the constant from both sides.",
                        student_expression=student_raw,
                    )
                elif orig_coeffs[0] == -stud_coeffs[0] and orig_coeffs[1] == -stud_coeffs[1]:
                    # Entire equation multiplied by -1 is valid, but here not matching
                    pass

        # Common error 2: Subtracted coefficient instead of dividing
        # e.g. 3x = 15 -> x = 12
        if student_roots:
            # Check if student subtracted the coeff
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="operation_error",
                message=(
                    f"This step is not equivalent to the previous equation. "
                    f"If you are isolating `{var}`, remember that multiplication is undone by division, "
                    f"and addition is undone by subtraction."
                ),
                hint=f"Inspect the operation linking `{var}` with its coefficient.",
                student_expression=student_raw,
            )

        return StepFeedback(
            is_valid=False,
            step_number=step_idx,
            status="incorrect",
            message=f"This step `{student_raw}` does not maintain the equality of the equation.",
            hint="Try retracing your previous line and applying the same operation to both sides.",
            student_expression=student_raw,
        )

    def _verify_expression_step(
        self,
        prob_expr: sp.Expr,
        last_ref: Any,
        student_obj: Any,
        student_is_eq: bool,
        var: sp.Symbol,
        student_raw: str,
        step_idx: int,
        operation: str,
        problem_raw: str,
    ) -> StepFeedback:
        """Verify steps for expressions, simplification, differentiation, and integration."""
        if student_is_eq:
            # If student entered e.g. "f'(x) = 3*x^2 + 4*x - 5", extract RHS
            student_expr = student_obj.rhs
        else:
            student_expr = student_obj

        # Detect if problem is a calculus operation
        is_diff = operation in ("diff", "differentiate") or any(
            w in problem_raw.lower() for w in ["d/dx", "diff", "derivative"]
        )
        is_int = operation in ("integrate", "integral") or any(
            w in problem_raw.lower() for w in ["integrate", "integral", "int"]
        )

        if is_diff:
            # Target is diff(prob_expr, var)
            expected = sp.diff(prob_expr, var)
            diff_check = sp.simplify(student_expr - expected)

            if diff_check == 0:
                return StepFeedback(
                    is_valid=True,
                    step_number=step_idx,
                    status="correct",
                    message="Excellent! You differentiated the expression accurately.",
                    student_expression=student_raw,
                    is_final_solution=True,
                )

            # Check sign error: student = -expected
            if sp.simplify(student_expr + expected) == 0:
                return StepFeedback(
                    is_valid=False,
                    step_number=step_idx,
                    status="sign_error",
                    message="You have a sign error! Your derivative is off by a factor of -1.",
                    hint="Review the signs on each differentiated term.",
                    student_expression=student_raw,
                )

            # Check if student differentiated power wrong (e.g. multiplied power but forgot to decrease exponent)
            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="incorrect",
                message="This derivative step is incorrect.",
                hint=f"Remember the power rule: d/d{var}[{var}^n] = n*{var}^(n-1). Apply it to each term.",
                student_expression=student_raw,
            )

        elif is_int:
            # Target is integrate(prob_expr, var)
            expected = sp.integrate(prob_expr, var)
            # Remove any "+ C" or Symbol('C') from student_expr if present
            student_clean = student_expr
            c_sym = sp.Symbol("C")
            if c_sym in student_expr.free_symbols:
                student_clean = student_expr.subs(c_sym, 0)

            diff_val = sp.simplify(student_clean - expected)
            if diff_val == 0 or (diff_val.is_number and diff_val.free_symbols == set()):
                return StepFeedback(
                    is_valid=True,
                    step_number=step_idx,
                    status="correct",
                    message="Spot on! Antiderivative is computed correctly.",
                    student_expression=student_raw,
                    is_final_solution=True,
                )

            # Check if derivative of student step equals original integrand
            if sp.simplify(sp.diff(student_clean, var) - prob_expr) == 0:
                return StepFeedback(
                    is_valid=True,
                    step_number=step_idx,
                    status="correct",
                    message="Correct! Differentiating your step yields the original integrand.",
                    student_expression=student_raw,
                    is_final_solution=True,
                )

            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="incorrect",
                message="This integration step is incorrect.",
                hint=f"Remember the integration power rule: int({var}^n) d{var} = ({var}^(n+1))/(n+1) for n != -1.",
                student_expression=student_raw,
            )

        else:
            # General simplification / factoring equivalence
            # Target is prob_expr or simplified prob_expr
            diff_from_orig = sp.simplify(student_expr - prob_expr)

            if diff_from_orig == 0:
                # Student rewrote it in an equivalent algebraic form
                final_simplified = sp.simplify(prob_expr)
                is_final = bool(sp.sstr(student_expr) == sp.sstr(final_simplified))
                return StepFeedback(
                    is_valid=True,
                    step_number=step_idx,
                    status="correct",
                    message="Valid algebraic step! Your expression is equivalent to the original.",
                    student_expression=student_raw,
                    is_final_solution=is_final,
                )

            # Check sign error
            if sp.simplify(student_expr + prob_expr) == 0:
                return StepFeedback(
                    is_valid=False,
                    step_number=step_idx,
                    status="sign_error",
                    message="Sign error detected: your expression is multiplied by -1.",
                    hint="Check signs when expanding, factoring, or canceling terms.",
                    student_expression=student_raw,
                )

            # Check if off by constant
            diff_deriv = sp.diff(student_expr - prob_expr, var)
            if diff_deriv == 0:
                return StepFeedback(
                    is_valid=False,
                    step_number=step_idx,
                    status="arithmetic_error",
                    message=f"The variable terms involving `{var}` are correct, but the constant term is off.",
                    hint="Check your constant arithmetic.",
                    student_expression=student_raw,
                )

            return StepFeedback(
                is_valid=False,
                step_number=step_idx,
                status="incorrect",
                message="This expression is not algebraically equivalent to the problem.",
                hint="Check factoring or distribution rules.",
                student_expression=student_raw,
            )
