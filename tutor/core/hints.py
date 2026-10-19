"""Progressive Hint Ladder Engine.

Generates 4-tier progressive hints:
- Level 1: Conceptual Strategy (Big picture / goal)
- Level 2: Next Algebraic Action (Targeted rule or operation)
- Level 3: Concrete Intermediate Step (Equation blueprint / checkpoint)
- Level 4: Full Solution Reveal (Complete step-by-step breakdown)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import sympy as sp

from tutor.core.solver import SymPySolver, TutorSolution

logger = logging.getLogger(__name__)


@dataclass
class TutorHint:
    """Represents a progressive hint at a specific tier."""
    problem: str
    level: int
    level_name: str
    hint_text: str
    latex: Optional[str] = None
    next_level_available: bool = True
    solution: Optional[TutorSolution] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "level": self.level,
            "level_name": self.level_name,
            "hint_text": self.hint_text,
            "latex": self.latex,
            "next_level_available": self.next_level_available,
        }


class HintGenerator:
    """Generates tiered progressive hints for mathematical problem solving."""

    LEVEL_NAMES = {
        1: "Conceptual Strategy",
        2: "Next Algebraic Action",
        3: "Concrete Intermediate Step",
        4: "Full Solution",
    }

    def __init__(self, solver: Optional[SymPySolver] = None) -> None:
        self.solver = solver or SymPySolver()

    def get_hint(
        self,
        problem: str,
        level: int = 1,
        variable: str = "x",
        operation: str = "auto",
    ) -> TutorHint:
        """Produce a progressive hint for the given problem at the specified level.

        Args:
            problem: The math problem string.
            level: Tier from 1 to 4 (defaults to 1; values > 4 are clamped to 4).
            variable: The variable in the problem (default 'x').
            operation: Operation type if known.

        Returns:
            TutorHint object.
        """
        clamped_level = max(1, min(4, level))
        level_name = self.LEVEL_NAMES[clamped_level]
        next_avail = clamped_level < 4

        # Obtain ground-truth solution steps
        sol = self.solver.solve(problem, operation=operation, variable=variable)
        if not sol.success:
            return TutorHint(
                problem=problem,
                level=clamped_level,
                level_name=level_name,
                hint_text=f"Could not parse problem to generate hints: {sol.error_message}",
                next_level_available=False,
            )

        op = sol.operation

        if clamped_level == 1:
            hint_text, latex = self._level_1(sol, variable)
        elif clamped_level == 2:
            hint_text, latex = self._level_2(sol, variable)
        elif clamped_level == 3:
            hint_text, latex = self._level_3(sol, variable)
        else:  # Level 4
            hint_text, latex = self._level_4(sol)

        return TutorHint(
            problem=problem,
            level=clamped_level,
            level_name=level_name,
            hint_text=hint_text,
            latex=latex,
            next_level_available=next_avail,
            solution=sol if clamped_level == 4 else None,
        )

    def _level_1(self, sol: TutorSolution, var: str) -> tuple[str, Optional[str]]:
        """Level 1: Big-picture conceptual strategy."""
        op = sol.operation
        if op == "solve":
            return (
                f"**Big Picture:** Your goal is to find the value of `{var}` that satisfies the equation. "
                f"Remember the golden rule of algebra: whatever operation you perform on one side, "
                f"you must also perform on the other side to keep the equation balanced.",
                None,
            )
        elif op == "diff":
            return (
                f"**Big Picture:** You are finding the instantaneous rate of change (derivative) with respect to `{var}`. "
                f"Since the derivative is a linear operator, you can differentiate the expression term-by-term.",
                rf"\frac{{d}}{{d{var}}}\left[u + v\right] = \frac{{du}}{{d{var}}} + \frac{{dv}}{{d{var}}}",
            )
        elif op == "integrate":
            return (
                f"**Big Picture:** Integration is the reverse process of differentiation (antiderivative). "
                f"Integrate each term individually and remember to include the constant of integration $+ C$.",
                rf"\int \left(u + v\right) d{var} = \int u\,d{var} + \int v\,d{var}",
            )
        elif op == "factor":
            return (
                f"**Big Picture:** Factoring breaks down an algebraic expression into a product of simpler polynomials. "
                f"Look first for a common factor (GCF) across all terms.",
                None,
            )
        else:  # simplify
            return (
                f"**Big Picture:** To simplify, look for opportunities to combine like terms or "
                f"factor the numerator and denominator so common factors can cancel out.",
                None,
            )

    def _level_2(self, sol: TutorSolution, var: str) -> tuple[str, Optional[str]]:
        """Level 2: Next specific algebraic action or rule."""
        op = sol.operation
        if op == "solve":
            if len(sol.steps) > 2:
                next_step = sol.steps[1]
                return (
                    f"**Next Action:** {next_step.title}.\n{next_step.description}",
                    next_step.latex,
                )
            return (
                f"**Next Action:** Move all terms involving `{var}` to the left side and all constant terms to the right side.",
                None,
            )
        elif op == "diff":
            return (
                f"**Next Action:** Apply the Power Rule for differentiation: for any term $c \\cdot {var}^n$, "
                f"the derivative is $c \\cdot n \\cdot {var}^{{n-1}}$. Constant terms become 0.",
                rf"\frac{{d}}{{d{var}}}\left[{var}^n\right] = n{var}^{{n-1}}",
            )
        elif op == "integrate":
            return (
                f"**Next Action:** Apply the Power Rule for integration: for each term ${var}^n$ ($n \\neq -1$), "
                f"add 1 to the exponent and divide by the new exponent: $\\frac{{{var}^{{n+1}}}}{{n+1}}$.",
                rf"\int {var}^n d{var} = \frac{{{var}^{{n+1}}}}{{n+1}} + C",
            )
        elif op == "factor":
            if len(sol.steps) > 1:
                return (
                    f"**Next Action:** Identify the polynomial type (e.g. trinomial, difference of squares $a^2 - b^2 = (a-b)(a+b)$).",
                    r"a^2 - b^2 = (a - b)(a + b)",
                )
            return ("**Next Action:** Group terms and factor out common expressions.", None)
        else:
            return (
                "**Next Action:** Check whether the numerator or denominator can be factored to reveal common factors.",
                None,
            )

    def _level_3(self, sol: TutorSolution, var: str) -> tuple[str, Optional[str]]:
        """Level 3: Concrete intermediate checkpoint."""
        if len(sol.steps) >= 3:
            mid_step = sol.steps[-2]
            return (
                f"**Intermediate Checkpoint:** After performing the initial steps, you should reach this checkpoint:\n"
                f"- **Step {mid_step.step_number}:** {mid_step.title}\n"
                f"- **Target form:** `{mid_step.expression}`\n"
                f"From here, carry out the final simplification or division.",
                mid_step.latex,
            )
        elif sol.steps:
            s = sol.steps[0]
            return (
                f"**Intermediate Checkpoint:** Re-express as `{s.expression}`.",
                s.latex,
            )
        return ("**Intermediate Checkpoint:** Almost there! Carry out the final simplification.", None)

    def _level_4(self, sol: TutorSolution) -> tuple[str, Optional[str]]:
        """Level 4: Full solution reveal."""
        lines = ["**Full Step-by-Step Solution:**\n"]
        for s in sol.steps:
            lines.append(f"**Step {s.step_number} ({s.title}):** {s.description}")
            lines.append(f"$$\n{s.latex}\n$$\n")
        lines.append(f"**Final Answer:** `{sol.final_answer}`")
        return ("\n".join(lines), sol.final_answer_latex)
