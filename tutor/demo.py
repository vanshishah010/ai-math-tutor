"""Interactive Demonstration Script for AI Step-by-Step Math Tutor.

Showcases 5 sample problem types:
1. Linear Equation: 3*x + 7 = 22
2. Quadratic Equation: x^2 - 5*x + 6 = 0
3. Derivative: x^3 + 2*x^2 - 5*x + 1
4. Indefinite Integral: 3*x^2 + 4*x - 2
5. Algebraic Simplification: (x^2 - 4)/(x - 2)

For each problem, demonstrates:
1. Step-by-Step Solution with full mathematical reasoning.
2. Progressive Hints (Level 1 -> Level 2 -> Level 3).
3. Student submits a WRONG step.
4. Tutor returns diagnostic feedback (identifies mistake without spoiling answer).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tutor.core.tutor import MathTutor


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_demo() -> None:
    tutor = MathTutor(default_variable="x")

    problems = [
        {
            "category": "1. Linear Equation",
            "problem": "3*x + 7 = 22",
            "operation": "solve",
            "wrong_step": "3*x = 29",
            "student_mistake_desc": "Student added 7 to the RHS instead of subtracting (sign error)",
        },
        {
            "category": "2. Quadratic Equation",
            "problem": "x^2 - 5*x + 6 = 0",
            "operation": "solve",
            "wrong_step": "x^2 = 5*x + 6",
            "student_mistake_desc": "Student flipped the sign on 6 incorrectly when moving terms",
        },
        {
            "category": "3. Calculus (Derivative)",
            "problem": "x^3 + 2*x^2 - 5*x + 1",
            "operation": "diff",
            "wrong_step": "-(3*x^2 + 4*x - 5)",
            "student_mistake_desc": "Student has an overall sign flip (-1) on the derivative",
        },
        {
            "category": "4. Calculus (Indefinite Integral)",
            "problem": "3*x^2 + 4*x - 2",
            "operation": "integrate",
            "wrong_step": "3*x^3 + 4*x^2 - 2*x",
            "student_mistake_desc": "Student increased exponents but forgot to divide by the new power",
        },
        {
            "category": "5. Algebraic Simplification",
            "problem": "(x^2 - 4)/(x - 2)",
            "operation": "simplify",
            "wrong_step": "x - 2",
            "student_mistake_desc": "Student canceled incorrectly and chose wrong sign factor",
        },
    ]

    for p in problems:
        print_banner(f"Problem Type: {p['category']}")
        print(f"📌 Problem: {p['problem']} (Operation: {p['operation']})")

        # 1. Full Step-by-Step Solution
        print("\n--- 📖 Step-by-Step Solution ---")
        sol = tutor.solve(p["problem"], operation=p["operation"], variable="x")
        for s in sol.steps:
            print(f"  Step {s.step_number} [{s.title}]: {s.description}")
            if s.expression:
                print(f"    -> Expression: {s.expression}")
        print(f"  🎯 Final Answer: {sol.final_answer}")

        # 2. Progressive Hints
        print("\n--- 💡 Progressive Hint Ladder ---")
        for lvl in range(1, 4):
            hint = tutor.hint(p["problem"], level=lvl, variable="x", operation=p["operation"])
            print(f"  [Hint Level {hint.level} - {hint.level_name}]:")
            print(f"    {hint.hint_text}")

        # 3. Student Submits Wrong Step -> Tutor Feedback
        print("\n--- 🧑‍🎓 Student Submits a Step ---")
        print(f"  Student's submitted step: '{p['wrong_step']}'")
        print(f"  (Context / Intentional Mistake: {p['student_mistake_desc']})")

        feedback = tutor.check_step(
            problem=p["problem"],
            previous_steps=[],
            student_step=p["wrong_step"],
            variable="x",
            operation=p["operation"],
        )

        print("\n--- 🧑‍🏫 Tutor Diagnostic Feedback ---")
        status_icon = "✅" if feedback.is_valid else "❌"
        print(f"  Status: {status_icon} Valid: {feedback.is_valid} | Code: {feedback.status}")
        print(f"  Feedback: {feedback.message}")
        if feedback.hint:
            print(f"  Diagnostic Hint: {feedback.hint}")


if __name__ == "__main__":
    run_demo()
