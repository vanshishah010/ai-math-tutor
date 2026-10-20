"""Interactive Practice Problem and Quiz Generator.

Generates pedagogically sound, difficulty-tiered practice problems similar
to a student's recently solved mathematical problem. Supports:
- Linear equations
- Quadratic equations
- Polynomial factoring
- Algebraic expansion
- Rational simplification
- Symbolic differentiation
- Symbolic integration
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp
from tutor.core.parser import MathParser

try:
    from tutor.core.parser import format_math_display, to_latex_equation
except ImportError:
    def format_math_display(expr: Any) -> str:
        return str(expr) if expr is not None else ""

    def to_latex_equation(expr: Any) -> str:
        if expr is None:
            return ""
        try:
            return sp.latex(sp.sympify(str(expr)))
        except Exception:
            return str(expr)

from tutor.core.solver import SymPySolver


@dataclass
class PracticeProblem:
    """Structure representing a generated practice quiz problem."""
    id: int
    problem: str
    operation: str
    difficulty: str
    variable: str
    expected_answer: str
    expected_answer_latex: str
    hint: str
    problem_type: str
    problem_latex: str = ""


class PracticeGenerator:
    """Intelligent generator for similar practice math problems with adjustable difficulty."""

    DIFFICULTIES = ["Easy", "Medium", "Hard"]

    def __init__(self, solver: Optional[SymPySolver] = None) -> None:
        self.solver = solver or SymPySolver()

    def generate(
        self,
        problem: str,
        operation: str = "solve",
        variable: str = "x",
        difficulty: str = "Medium",
        count: int = 3,
        seed: Optional[int] = None,
    ) -> List[PracticeProblem]:
        """Generate `count` practice problems similar to `problem` at specified difficulty."""
        rng = random.Random(seed)
        difficulty = difficulty.capitalize() if difficulty.capitalize() in self.DIFFICULTIES else "Medium"
        var_sym = sp.Symbol(variable.strip() or "x")
        var_str = str(var_sym)

        problem_type = self._detect_problem_type(problem, operation, var_sym)
        problems: List[PracticeProblem] = []
        seen_problems = {problem.strip().replace(" ", "")}

        attempts = 0
        max_attempts = count * 20

        while len(problems) < count and attempts < max_attempts:
            attempts += 1
            prob_str, hint = self._generate_single_problem(problem_type, difficulty, var_str, rng)
            norm_key = prob_str.strip().replace(" ", "")
            if norm_key in seen_problems:
                continue

            # Verify that SymPy can solve this problem cleanly
            sol = self.solver.solve(prob_str, operation=operation, variable=var_str)
            if sol.success and sol.final_answer and sol.final_answer != "Could not parse problem.":
                seen_problems.add(norm_key)
                clean_prob = format_math_display(prob_str)
                prob_latex = to_latex_equation(getattr(sol, "parsed_object", None) or prob_str)
                problems.append(
                    PracticeProblem(
                        id=len(problems) + 1,
                        problem=clean_prob,
                        operation=sol.operation,
                        difficulty=difficulty,
                        variable=var_str,
                        expected_answer=sol.final_answer,
                        expected_answer_latex=sol.final_answer_latex or sol.final_answer,
                        hint=format_math_display(hint),
                        problem_type=problem_type,
                        problem_latex=prob_latex,
                    )
                )

        # Fallback if specific generation produced fewer than count
        while len(problems) < count:
            fallback_prob, hint = self._fallback_problem(problem_type, difficulty, var_str, len(problems) + 1)
            sol = self.solver.solve(fallback_prob, operation=operation, variable=var_str)
            clean_fallback = format_math_display(fallback_prob)
            prob_latex = to_latex_equation(getattr(sol, "parsed_object", None) or fallback_prob)
            problems.append(
                PracticeProblem(
                    id=len(problems) + 1,
                    problem=clean_fallback,
                    operation=operation,
                    difficulty=difficulty,
                    variable=var_str,
                    expected_answer=sol.final_answer if sol.success else "x = 0",
                    expected_answer_latex=sol.final_answer_latex if sol.success else "x = 0",
                    hint=format_math_display(hint),
                    problem_type=problem_type,
                    problem_latex=prob_latex,
                )
            )

        return problems

    def _detect_problem_type(self, problem: str, operation: str, var_sym: sp.Symbol) -> str:
        """Classify problem category to generate matching practice templates."""
        op = operation.lower() if operation else "solve"
        if op in ("diff", "differentiation"):
            return "differentiation"
        if op in ("integrate", "integration"):
            return "integration"
        if op in ("factor", "factorisation"):
            return "factorisation"
        if op in ("expand", "expansion"):
            return "expansion"
        if op in ("simplify", "simplification"):
            return "simplification"

        # Check equations
        try:
            parsed = MathParser.parse(problem, default_variable=str(var_sym))
            if parsed.success and parsed.is_equation and parsed.parsed_object is not None:
                eq = parsed.parsed_object
                lhs_rhs = eq.lhs - eq.rhs
                poly = sp.Poly(lhs_rhs, var_sym)
                deg = poly.degree()
                if deg == 1:
                    return "linear_equation"
                elif deg == 2:
                    return "quadratic_equation"
                elif deg > 2:
                    return "polynomial_equation"
        except Exception:
            pass

        if "=" in problem:
            if "^2" in problem or "**2" in problem:
                return "quadratic_equation"
            return "linear_equation"

        return "quadratic_equation"

    def _generate_single_problem(
        self, problem_type: str, difficulty: str, var: str, rng: random.Random
    ) -> Tuple[str, str]:
        """Generate a single raw problem string and hint based on type and difficulty."""
        if problem_type == "quadratic_equation":
            return self._gen_quadratic(difficulty, var, rng)
        elif problem_type == "linear_equation":
            return self._gen_linear(difficulty, var, rng)
        elif problem_type == "factorisation":
            return self._gen_factor(difficulty, var, rng)
        elif problem_type == "expansion":
            return self._gen_expand(difficulty, var, rng)
        elif problem_type == "differentiation":
            return self._gen_diff(difficulty, var, rng)
        elif problem_type == "integration":
            return self._gen_integrate(difficulty, var, rng)
        elif problem_type == "simplification":
            return self._gen_simplify(difficulty, var, rng)
        else:
            return self._gen_quadratic(difficulty, var, rng)

    def _gen_quadratic(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate quadratic equations: (x - r1)(x - r2) = 0 or a*x^2 + b*x + c = 0."""
        x = sp.Symbol(var)
        if difficulty == "Easy":
            # Nice small positive/negative integer roots, monic (a = 1)
            roots = rng.sample([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], 2)
            expr = sp.expand((x - roots[0]) * (x - roots[1]))
            hint = f"Try factoring into ({var} - p)({var} - q) = 0 by finding two numbers that multiply to {expr.coeff(x, 0)} and add to {expr.coeff(x, 1)}."
            prob_str = f"{expr} = 0"
        elif difficulty == "Medium":
            # Monic with non-zero RHS or simple non-monic (a = 2 or 3)
            mode = rng.choice(["non_zero_rhs", "non_monic"])
            if mode == "non_zero_rhs":
                roots = rng.sample([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7], 2)
                expr = sp.expand((x - roots[0]) * (x - roots[1]))
                k = rng.randint(2, 12)
                lhs = sp.expand(expr + k)
                prob_str = f"{lhs} = {k}"
                hint = f"First subtract {k} from both sides to set the quadratic equal to 0, then factor or apply the quadratic formula."
            else:
                a = rng.choice([2, 3])
                r1 = rng.choice([-3, -2, -1, 1, 2, 3])
                r2 = rng.choice([-4, -3, -1, 1, 2, 4])
                expr = sp.expand((a * x - r1) * (x - r2))
                prob_str = f"{expr} = 0"
                hint = f"Identify coefficients a={a}, b={expr.coeff(x, 1)}, and c={expr.coeff(x, 0)} and apply the quadratic formula x = (-b ± √(b² - 4ac)) / (2a)."
        else:  # Hard
            # Non-monic with larger coefficients or completing-the-square structure
            a = rng.choice([2, 3, 4, 5])
            r1 = rng.choice([-5, -3, -2, 1, 2, 3, 5])
            r2 = rng.choice([-4, -2, -1, 1, 2, 3, 4])
            c_shift = rng.randint(1, 8)
            poly = sp.expand((a * x + r1) * (x - r2))
            prob_str = f"{sp.expand(poly + c_shift)} = {c_shift}"
            hint = "Rearrange all terms to one side. If factoring is difficult, use the quadratic formula with precision."

        return prob_str, hint

    def _gen_linear(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate linear equations: ax + b = c or with parentheses."""
        x = sp.Symbol(var)
        if difficulty == "Easy":
            a = rng.choice([2, 3, 4, 5, 6, 7])
            root = rng.randint(-8, 12)
            b = rng.randint(-15, 15)
            c = a * root + b
            prob_str = f"{a}*{var} + {b} = {c}".replace("+ -", "- ")
            hint = f"Isolate the {var} term by subtracting {b} from both sides, then divide by {a}."
        elif difficulty == "Medium":
            # Variables on both sides
            a1 = rng.choice([3, 4, 5, 6])
            a2 = rng.choice([1, 2])
            root = rng.randint(-7, 10)
            b1 = rng.randint(1, 12)
            # a1*root + b1 = a2*root + b2  =>  b2 = (a1 - a2)*root + b1
            b2 = (a1 - a2) * root + b1
            prob_str = f"{a1}*{var} + {b1} = {a2}*{var} + {b2}".replace("+ -", "- ")
            hint = f"Collect all {var} terms on the left side and all constant terms on the right side."
        else:  # Hard
            # Parentheses and multiple operations
            m = rng.choice([2, 3])
            k = rng.choice([2, 4])
            root = rng.randint(-5, 8)
            c1 = rng.randint(1, 5)
            c2 = rng.randint(2, 6)
            lhs_val = m * (root - c1) + c2
            prob_str = f"{m}*({var} - {c1}) + {c2} = {lhs_val}".replace("- -", "+ ")
            hint = f"Expand the brackets {m}*({var} - {c1}) first, combine like terms, and isolate {var}."
        return prob_str, hint

    def _gen_factor(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate factoring exercises."""
        x = sp.Symbol(var)
        if difficulty == "Easy":
            # Difference of squares or basic monic factoring
            mode = rng.choice(["diff_sq", "monic"])
            if mode == "diff_sq":
                k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
                prob_str = f"{var}^2 - {k*k}"
                hint = f"Recognize this as a difference of squares: a² - b² = (a - b)(a + b) with b = {k}."
            else:
                r1, r2 = rng.sample([1, 2, 3, 4, 5], 2)
                expr = sp.expand((x + r1) * (x + r2))
                prob_str = str(expr).replace("**", "^")
                hint = f"Find two positive factors of {r1*r2} that sum to {r1+r2}."
        elif difficulty == "Medium":
            r1 = rng.choice([-5, -4, -3, -2, -1])
            r2 = rng.choice([1, 2, 3, 4, 5])
            expr = sp.expand((x - r1) * (x - r2))
            prob_str = str(expr).replace("**", "^")
            hint = "Look for two numbers with opposite signs whose product is negative."
        else:  # Hard
            a = rng.choice([2, 3, 4])
            r1 = rng.choice([-4, -3, -1, 1, 2, 3])
            r2 = rng.choice([-3, -2, 1, 2, 4])
            expr = sp.expand((a * x + r1) * (x + r2))
            prob_str = str(expr).replace("**", "^")
            hint = f"Since the leading coefficient is {a}, use grouping: find factors of a*c that sum to b."
        return prob_str, hint

    def _gen_expand(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate expansion exercises."""
        if difficulty == "Easy":
            a = rng.randint(2, 6)
            b = rng.randint(1, 8)
            prob_str = f"({var} + {a})*({var} + {b})"
            hint = "Use the FOIL method: First, Outside, Inside, Last."
        elif difficulty == "Medium":
            a = rng.randint(2, 5)
            b = rng.randint(2, 7)
            prob_str = f"({a}*{var} - {b})^2"
            hint = "Apply the binomial expansion formula (a - b)² = a² - 2ab + b²."
        else:  # Hard
            a = rng.randint(2, 4)
            b = rng.randint(1, 4)
            c = rng.randint(1, 3)
            prob_str = f"({a}*{var} + {b})*({var}^2 - {c}*{var} + 2)"
            hint = "Distribute each term of the binomial across all three terms of the trinomial."
        return prob_str, hint

    def _gen_diff(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate differentiation exercises."""
        if difficulty == "Easy":
            a = rng.randint(2, 6)
            b = rng.randint(2, 8)
            n = rng.choice([3, 4])
            prob_str = f"{a}*{var}^{n} - {b}*{var} + {rng.randint(1, 9)}"
            hint = "Apply the power rule: d/dx [x^n] = n*x^(n-1) to each term."
        elif difficulty == "Medium":
            c = rng.choice([2, 3, 4])
            prob_str = f"{var}^{c} * sin({var})"
            hint = "Apply the product rule: (u * v)' = u' * v + u * v'."
        else:  # Hard
            k = rng.choice([2, 3])
            prob_str = f"exp({k}*{var}) * cos({var})"
            hint = "Combine the chain rule with the product rule."
        return prob_str, hint

    def _gen_integrate(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate integration exercises."""
        if difficulty == "Easy":
            a = rng.choice([3, 6, 9])
            b = rng.choice([2, 4, 6])
            prob_str = f"{a}*{var}^2 + {b}*{var}"
            hint = "Apply the reverse power rule: ∫ x^n dx = (x^(n+1))/(n+1) + C."
        elif difficulty == "Medium":
            prob_str = f"cos({var}) - 3*{var}^2"
            hint = "Integrate term by term: ∫ cos(x) dx = sin(x) and ∫ x² dx = x³/3."
        else:  # Hard
            u1, u2 = 0, rng.choice([1, 2])
            prob_str = f"{var}^2 from {u1} to {u2}"
            hint = f"Find the antiderivative x³/3, then evaluate F({u2}) - F({u1})."
        return prob_str, hint

    def _gen_simplify(self, difficulty: str, var: str, rng: random.Random) -> Tuple[str, str]:
        """Generate algebraic simplification exercises."""
        if difficulty == "Easy":
            k = rng.choice([2, 3, 4, 5])
            prob_str = f"({var}^2 - {k*k})/({var} - {k})"
            hint = f"Factor the numerator as a difference of squares ({var} - {k})({var} + {k}) and cancel ({var} - {k})."
        elif difficulty == "Medium":
            r1, r2 = rng.sample([1, 2, 3, 4], 2)
            num = f"{var}^2 + {r1+r2}*{var} + {r1*r2}"
            prob_str = f"({num})/({var} + {r1})"
            hint = f"Factor the quadratic numerator into ({var} + {r1})({var} + {r2}) and simplify."
        else:  # Hard
            k = rng.choice([2, 3])
            prob_str = f"({k}*{var}^2 + {k*3}*{var})/({var}^2 - 9)"
            hint = f"Factor out common factor {k}*{var} on top, factor difference of squares on bottom, and cancel common factors."
        return prob_str, hint

    def _fallback_problem(self, problem_type: str, difficulty: str, var: str, idx: int) -> Tuple[str, str]:
        """Reliable static fallbacks for edge cases."""
        fallbacks = {
            "quadratic_equation": [
                (f"{var}^2 - 7*{var} + 12 = 0", "Factor into two binomials whose constants multiply to 12."),
                (f"{var}^2 + 3*{var} - 10 = 0", "Find numbers with opposite signs multiplying to -10."),
                (f"2*{var}^2 - 8 = 0", "Divide by 2 to get x² - 4 = 0, then take square roots."),
            ],
            "linear_equation": [
                (f"3*{var} + 5 = 20", "Subtract 5 then divide by 3."),
                (f"5*{var} - 7 = 3*{var} + 9", "Collect x terms on one side."),
                (f"2*({var} + 4) = 18", "Distribute or divide both sides by 2."),
            ],
            "factorisation": [
                (f"{var}^2 - 16", "Difference of squares."),
                (f"{var}^2 + 5*{var} + 6", "Factor into (x + 2)(x + 3)."),
                (f"2*{var}^2 + 5*{var} + 2", "Factor non-monic trinomial."),
            ],
        }
        items = fallbacks.get(problem_type, fallbacks["quadratic_equation"])
        return items[(idx - 1) % len(items)]

    def check_answer(self, student_answer: str, expected_answer: str, variable: str = "x") -> Tuple[bool, str]:
        """Check student's proposed answer against expected answer using symbolic verification."""
        if not student_answer or not student_answer.strip():
            return False, "Please enter an answer before checking."

        from tutor.core.verifier import StepVerifier
        verifier = StepVerifier(solver=self.solver)

        # 1. Clean and normalize
        s_ans = student_answer.strip()
        e_ans = expected_answer.strip()

        # 2. Direct string equality (case-insensitive)
        if s_ans.lower() == e_ans.lower():
            return True, "Correct! Your answer is exactly correct."

        # 3. Use symbolic solver verification
        try:
            var_sym = sp.Symbol(variable)
            s_parsed = MathParser.sanitize_math_string(s_ans)
            e_parsed = MathParser.sanitize_math_string(e_ans)

            s_clean = s_parsed.replace(f"{variable}=", "").replace(f"{variable} =", "").strip()
            e_clean = e_parsed.replace(f"{variable}=", "").replace(f"{variable} =", "").strip()

            s_expr = sp.sympify(s_clean)
            e_expr = sp.sympify(e_clean)

            if sp.simplify(s_expr - e_expr) == 0:
                return True, "Correct! Your answer is mathematically equivalent and verified."
        except Exception:
            pass

        # 4. Multi-root list check (e.g. "x = 2, x = 3" vs "3, 2")
        try:
            import re
            s_nums = sorted([float(n) for n in re.findall(r"[-+]?\d+(?:\.\d+)?", s_ans)])
            e_nums = sorted([float(n) for n in re.findall(r"[-+]?\d+(?:\.\d+)?", e_ans)])
            if s_nums and s_nums == e_nums:
                return True, "Correct! You found all valid solutions."
        except Exception:
            pass

        return False, f"Not quite. Expected: {expected_answer}. Try reviewing your steps."
