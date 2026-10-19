"""Main MathTutor Coordinator Class.

Provides the unified, high-level API for:
1. Step-by-step solving: `MathTutor.solve(problem, operation, variable)`
2. Progressive hints: `MathTutor.hint(problem, level, variable)`
3. Step verification & diagnostic feedback: `MathTutor.check_step(problem, previous_steps, student_step, variable)`
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from tutor.core.hints import HintGenerator, TutorHint
from tutor.core.solver import SymPySolver, TutorSolution
from tutor.core.verifier import StepFeedback, StepVerifier

logger = logging.getLogger(__name__)


class MathTutor:
    """Unified AI Step-by-Step Math Tutor.

    Combines ground-truth symbolic computation (SymPy), step equivalence checking,
    and progressive hint ladders with optional neural natural language explanations.
    """

    def __init__(
        self,
        default_variable: str = "x",
        enable_llm: bool = False,
        adapter_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        """Initialize the tutor.

        Args:
            default_variable: Default algebraic variable symbol (default 'x').
            enable_llm: Whether to attempt loading the fine-tuned small LLM (lazy-loaded).
            adapter_path: Optional path to LoRA adapter weights (None uses base model).
            device: 'auto', 'cpu', or 'cuda'.
        """
        self.default_variable = default_variable
        self.enable_llm = enable_llm
        self.adapter_path = adapter_path
        self.device = device

        self.solver = SymPySolver(default_var=default_variable)
        self.verifier = StepVerifier(solver=self.solver)
        self.hint_generator = HintGenerator(solver=self.solver)
        self._explainer = None  # Lazy-loaded on demand if enable_llm is True

    def solve(
        self,
        problem: str,
        operation: str = "auto",
        variable: Optional[str] = None,
        limits: Optional[Tuple[float, float]] = None,
    ) -> TutorSolution:
        """Produce a complete, granular, step-by-step solution for a math problem.

        Args:
            problem: Mathematical problem string (e.g. '3*x + 7 = 22', 'x^2 - 5*x + 6 = 0').
            operation: One of 'auto', 'solve', 'simplify', 'factor', 'expand', 'diff', 'integrate'.
            variable: Variable symbol (defaults to tutor's default_variable).
            limits: Optional limits for definite integration (a, b).

        Returns:
            TutorSolution containing structured pedagogical steps and final answer.
        """
        var = variable.strip() if (variable and variable.strip()) else self.default_variable
        logger.info("Solving problem: '%s' (op=%s, var=%s)", problem, operation, var)
        solution = self.solver.solve(problem=problem, operation=operation, variable=var, limits=limits)

        # If LLM explainer is enabled and available, we can augment descriptions
        if self.enable_llm and solution.success:
            try:
                explainer = self._get_explainer()
                if explainer and explainer.is_available():
                    solution = explainer.enrich_solution(solution)
            except Exception as e:
                logger.warning("LLM enrichment failed; using SymPy rule-based explanations: %s", e)

        return solution

    def hint(
        self,
        problem: str,
        level: int = 1,
        variable: Optional[str] = None,
        operation: str = "auto",
    ) -> TutorHint:
        """Generate a progressive hint for a problem at a specific level (1 to 4).

        Args:
            problem: Mathematical problem string.
            level: Hint level (1: Conceptual, 2: Next Action, 3: Intermediate Checkpoint, 4: Full Solution).
            variable: Variable symbol.
            operation: Operation type if known.

        Returns:
            TutorHint containing the tiered advice and LaTeX formatting.
        """
        var = variable.strip() if (variable and variable.strip()) else self.default_variable
        logger.info("Generating Level %d hint for problem: '%s'", level, problem)
        return self.hint_generator.get_hint(
            problem=problem,
            level=level,
            variable=var,
            operation=operation,
        )

    def check_step(
        self,
        problem: str,
        previous_steps: Optional[List[str]],
        student_step: str,
        variable: Optional[str] = None,
        operation: str = "auto",
    ) -> StepFeedback:
        """Verify whether a student's intermediate step is mathematically sound.

        Args:
            problem: Original math problem string.
            previous_steps: Ordered list of previous accepted steps (can be empty or None).
            student_step: The step entered by the student.
            variable: Variable symbol.
            operation: Operation type if known.

        Returns:
            StepFeedback indicating validity, diagnostic status, and helpful guidance.
        """
        var = variable.strip() if (variable and variable.strip()) else self.default_variable
        logger.info("Checking student step: '%s' for problem: '%s'", student_step, problem)
        return self.verifier.check_step(
            problem=problem,
            previous_steps=previous_steps,
            student_step=student_step,
            variable=var,
            operation=operation,
        )

    def _get_explainer(self):
        """Lazy loader for neural explainer model."""
        if self._explainer is None:
            from tutor.core.explainer import LLMExplainer
            self._explainer = LLMExplainer(adapter_path=self.adapter_path, device=self.device)
        return self._explainer

    def solve_direct(self, problem: str) -> str:
        """Prompt the neural LLM directly to solve a problem step-by-step."""
        explainer = self._get_explainer()
        if explainer and explainer.is_available():
            return explainer.generate_solution(problem)
        return ""
