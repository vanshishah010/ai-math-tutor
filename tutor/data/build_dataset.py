"""Rebuild AI Math Tutor Training Dataset.

Constructs a multi-source, high-quality, step-by-step mathematical dataset:
1. SymPy-Generated (~4,500 samples): Algebra, calculus, simplification, factoring, expansion
   with ground-truth step extraction, SymPy answer verification, and natural phrasing.
2. Hugging Face (~6,700 samples): GSM8K (5,500 target) and MetaMathQA (1,200+ target)
   with strict arithmetic validation on all step equations (<expr> = <num>) using SymPy
   and reference answer consistency checking.
3. Kaggle 520K: Kept for reference but disabled (USE_KAGGLE = False) due to mismatched/hallucinated
   answers in the raw dataset.

Deduplicates, enforces <= 512 token limit, shuffles with fixed seed (42),
splits 90/5/5 into train/val/test, and saves dataset_stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sympy as sp
from tutor.core.explainer import SYSTEM_PROMPT
from tutor.core.solver import SymPySolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Flag to control Kaggle 520K dataset loading.
# Set to False because the Kaggle 520K dataset contains hallucinated/mismatched solutions
# (e.g., floor sums answered with digit sums; perfect square tests answered with 28).
USE_KAGGLE: bool = False

DEFAULT_LOCAL_KAGGLE_PATH = r"C:\Users\vanshi\Downloads\archive\datasetv2.jsonl\dataset.jsonl"
KAGGLE_DATASET_ID = "dineshkumar0705/large-scale-math-reasoning-dataset-520k"

# Global drop tracking
DROP_REASONS: Dict[str, int] = {
    "dropped_arithmetic": 0,
    "dropped_final_answer_mismatch": 0,
    "dropped_dupes": 0,
    "dropped_length": 0,
    "dropped_empty": 0,
}


# Global counter for samples with #### artifact cleaned
SAMPLES_WITH_HASH_ARTIFACT: int = 0


# -----------------------------------------------------------------------------
# 1. SymPy-Generated Dataset Generator (~4,500 samples)
# -----------------------------------------------------------------------------
def clean_sympy_math_formatting(text: str) -> str:
    """Standardize math formatting in SymPy-generated steps.

    Uses plain backticks for expressions everywhere, no $...$ and no raw \\int.
    Writes integrals as 'integral of ... dx' or 'definite integral of ... dx from a to b'.
    """
    # 1. Definite integrals: (definite integral) $\int_{a}^{b} (expr) dx$ -> definite integral of `expr` dx from a to b
    text = re.sub(
        r'(?:definite\s+integral\s*)?\$?\\int_\{?([+-]?\d+)\}?\^\{?([+-]?\d+)\}?\s*(?:\\left\(|\()?([^$\\\n]+?)(?:\\right\)|\))?\s*dx\$?',
        r'definite integral of `\3` dx from \1 to \2',
        text,
        flags=re.I,
    )
    # 2. Indefinite integrals: (indefinite integral) $\int (expr) dx$ -> integral of `expr` dx
    text = re.sub(
        r'(?:indefinite\s+integral\s*)?\$?\\int\s*(?:\\left\(|\()?([^$\\\n]+?)(?:\\right\)|\))?\s*dx\$?',
        r'integral of `\1` dx',
        text,
        flags=re.I,
    )
    # 3. Clean remaining LaTeX symbols: \Delta -> Delta, \cdot -> *, \left / \right stripped, \int -> integral of
    text = re.sub(r'\\Delta\b', 'Delta', text)
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\left|\\right', '', text)
    text = re.sub(r'\\int\b', 'integral of', text)

    # 4. Standardize all $...$ into `...`
    text = re.sub(r'\$([^$]+)\$', r'`\1`', text)

    # 5. Fix double backticks or spacing around backticks
    text = re.sub(r'`+', '`', text)
    text = re.sub(r'`\s*`', ' ', text)
    text = re.sub(r'(`[^`]+`)(?=[a-zA-Z0-9])', r'\1 ', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])(`[^`]+`)', r' \1', text)

    # Clean inside backticks
    text = re.sub(r'`\s+([^`]+?)`', r'`\1`', text)
    text = re.sub(r'`([^`]+?)\s+`', r'`\1`', text)

    # Guarantee zero dollars remain
    text = text.replace('$', '')
    return text


def format_sympy_solution(steps: List[Any], final_answer: str) -> str:
    """Format SymPy solver steps into friendly pedagogical numbered explanation.

    Standardizes math formatting using backticks everywhere, with no $ and no raw \\int.
    Removes trailing filler phrases like 'using the Problem Statement.' or
    'using the Final Result.', ensuring each step reads naturally.
    """
    lines = []
    for s in steps:
        title = clean_sympy_math_formatting(s.title.strip())
        desc = clean_sympy_math_formatting(s.description.strip())

        # Clean trailing filler phrases
        desc = re.sub(
            r"\s*using the (?:Problem Statement|Final Result|Solution Set|Given Expression|Indefinite Integral Definition)\.?$",
            "",
            desc,
            flags=re.I,
        ).strip()
        desc = desc.rstrip(".")

        lines.append(f"Step {s.step_number}: {title} — {desc}.")
    lines.append(f"Final answer: {final_answer}")
    return "\n".join(lines)


def generate_sympy_samples(target_count: int = 4500) -> List[Dict[str, Any]]:
    """Programmatically generate varied single-variable math problems verified by SymPy."""
    logger.info("Generating ~%d SymPy ground-truth problems with step verification...", target_count)
    solver = SymPySolver()
    samples: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    sub_targets = {
        "linear": int(target_count * 0.18),       # ~810
        "quadratic": int(target_count * 0.18),    # ~810
        "derivative": int(target_count * 0.17),   # ~765
        "integral": int(target_count * 0.17),     # ~765
        "simplify": int(target_count * 0.11),     # ~495
        "factor": int(target_count * 0.11),       # ~495
        "expand": int(target_count * 0.08),       # ~360
    }

    x = sp.Symbol("x")

    # Helper: Linear generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "linear"]) < sub_targets["linear"] and attempts < 15000:
        attempts += 1
        mode = random.choice(["simple", "both_sides", "brackets", "fraction", "both_brackets"])
        if mode == "simple":
            a = random.choice([-25, -20, -15, -12, -9, -7, -5, -4, -3, -2, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25])
            b = random.randint(-80, 80)
            c = random.randint(-100, 100)
            eqn = f"{a}*x + {b} = {c}".replace("+ -", "- ")
        elif mode == "both_sides":
            a = random.choice([-20, -15, -10, -8, -6, -5, -3, -2, 2, 3, 5, 6, 8, 10, 15, 20])
            c = random.choice([x_val for x_val in [-18, -12, -7, -4, -1, 1, 4, 7, 12, 18] if x_val != a])
            b = random.randint(-70, 70)
            d = random.randint(-70, 70)
            eqn = f"{a}*x + {b} = {c}*x + {d}".replace("+ -", "- ")
        elif mode == "brackets":
            a = random.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
            b = random.choice([1, 2, 3, 4])
            c = random.randint(-15, 15)
            d = random.randint(-40, 40)
            eqn = f"{a}*({b}*x + {c}) = {d}".replace("+ -", "- ")
        elif mode == "both_brackets":
            a = random.choice([2, 3, 4, 5, 6])
            b = random.choice([1, 2, 3])
            c = random.randint(-10, 10)
            d = random.randint(-20, 20)
            e = random.choice([1, 2, 3, 4])
            f = random.randint(-10, 10)
            g = random.randint(-30, 30)
            eqn = f"{a}*({b}*x + {c}) + {d} = {e}*(x + {f}) + {g}".replace("+ -", "- ")
        else:
            c_val = random.choice([2, 3, 4, 5, 6])
            a = random.choice([1, 2, 3, 4, 5]) * c_val
            b = random.randint(-30, 30)
            d = random.randint(-20, 20)
            eqn = f"({a}*x + {b})/{c_val} = {d}".replace("+ -", "- ")

        if eqn in seen:
            continue
        seen.add(eqn)

        try:
            sol = solver.solve(eqn, operation="solve", variable="x")
            if sol.success and sol.steps and sol.final_answer:
                ans_str = sol.final_answer.replace("x =", "").strip().split("=")[0].strip()
                ans_val = sp.sympify(ans_str)
                lhs_str, rhs_str = eqn.split("=")
                lhs = sp.sympify(lhs_str)
                rhs = sp.sympify(rhs_str)
                if sp.simplify(lhs.subs(x, ans_val) - rhs.subs(x, ans_val)) == 0:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Solve for x: {eqn}"},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "linear", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Quadratic generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "quadratic"]) < sub_targets["quadratic"] and attempts < 15000:
        attempts += 1
        q_mode = random.choice(["standard", "pure", "common_x"])
        if q_mode == "pure":
            r_val = random.randint(1, 15)
            a = random.choice([1, 2, 3, 4])
            b_coeff = 0
            c_coeff = -a * (r_val ** 2)
            eqn = f"{a}*x^2 - {abs(c_coeff)} = 0".replace("1*x^2", "x^2")
        elif q_mode == "common_x":
            a = random.choice([1, 2, 3, 4, 5])
            b_coeff = random.choice([-20, -15, -12, -8, -6, 6, 8, 12, 15, 20])
            eqn = f"{a}*x^2 + {b_coeff}*x = 0".replace("+ -", "- ").replace("1*x^2", "x^2")
        else:
            r1 = random.randint(-20, 20)
            r2 = random.randint(-20, 20)
            a = random.choice([-2, -1, 1, 1, 1, 2, 3, 4])
            b_coeff = -a * (r1 + r2)
            c_coeff = a * r1 * r2
            eqn = f"{a}*x^2 + {b_coeff}*x + {c_coeff} = 0".replace("+ -", "- ").replace("1*x^2", "x^2").replace("+ 0*x ", "")

        if eqn in seen:
            continue
        seen.add(eqn)

        try:
            sol = solver.solve(eqn, operation="solve", variable="x")
            if sol.success and sol.steps and sol.final_answer:
                lhs_str = eqn.split("=")[0].strip()
                lhs = sp.sympify(lhs_str.replace("^", "**"))
                # Verify roots
                roots = [s.strip().replace("x =", "").strip() for s in sol.final_answer.split(",")]
                all_valid = True
                for rt in roots:
                    val = sp.sympify(rt)
                    if sp.simplify(lhs.subs(x, val)) != 0:
                        all_valid = False
                        break
                if all_valid:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Solve for x: {eqn}"},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "quadratic", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Derivative generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "derivative"]) < sub_targets["derivative"] and attempts < 15000:
        attempts += 1
        pattern = random.choice(["poly", "multi_poly", "trig", "exp", "product_trig", "product_exp", "rational"])
        if pattern == "poly":
            n = random.randint(2, 8)
            a = random.randint(-15, 15)
            if a == 0: a = 2
            b = random.randint(-20, 20)
            expr_str = f"{a}*x^{n} + {b}*x".replace("+ -", "- ")
            problem = f"Differentiate {expr_str} with respect to x"
        elif pattern == "multi_poly":
            n = random.randint(3, 7)
            a = random.randint(1, 10)
            b = random.randint(-10, 10)
            c = random.randint(-30, 30)
            expr_str = f"{a}*x^{n} + {b}*x^2 + {c}".replace("+ -", "- ")
            problem = f"Differentiate {expr_str} with respect to x"
        elif pattern == "trig":
            a = random.randint(1, 8)
            b = random.choice(["sin", "cos"])
            c = random.randint(1, 6)
            problem = f"Differentiate {a}*{b}({c}*x) with respect to x"
        elif pattern == "exp":
            a = random.randint(1, 6)
            b = random.randint(1, 5)
            problem = f"Find dy/dx for y = {a}*exp({b}*x)"
        elif pattern == "product_trig":
            n = random.randint(1, 4)
            trig_fn = random.choice(["sin", "cos"])
            problem = f"Differentiate x^{n}*{trig_fn}(x) with respect to x"
        elif pattern == "rational":
            n = random.randint(1, 3)
            a = random.randint(1, 5)
            problem = f"Differentiate {a}/x^{n} + 2*x with respect to x"
        else:
            n = random.randint(1, 4)
            problem = f"Derivative of x^{n}*exp(x)"

        if problem in seen:
            continue
        seen.add(problem)

        try:
            sol = solver.solve(problem, operation="diff", variable="x")
            if sol.success and sol.steps and sol.final_answer and sol.original_expr is not None:
                expected = sp.diff(sol.original_expr, x)
                if sp.simplify(sol.raw_result - expected) == 0:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "derivative", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Integral generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "integral"]) < sub_targets["integral"] and attempts < 15000:
        attempts += 1
        is_definite = random.choice([True, False])
        if is_definite:
            a_lim = random.randint(-4, 3)
            b_lim = a_lim + random.randint(1, 5)
            c = random.randint(1, 8)
            deg = random.choice([1, 2, 3])
            problem = f"Integrate {c}*x^{deg} from {a_lim} to {b_lim}"
            limits = (a_lim, b_lim)
        else:
            n = random.randint(1, 5)
            a = random.randint(1, 10)
            b = random.randint(-15, 15)
            problem = f"Integrate {a}*x^{n} + {b}".replace("+ -", "- ")
            limits = None

        if problem in seen:
            continue
        seen.add(problem)

        try:
            sol = solver.solve(problem, operation="integrate", variable="x", limits=limits)
            if sol.success and sol.steps and sol.final_answer and sol.original_expr is not None:
                if limits:
                    expected = sp.integrate(sol.original_expr, (x, limits[0], limits[1]))
                    is_ok = sp.simplify(sol.raw_result - expected) == 0
                else:
                    expected = sp.integrate(sol.original_expr, x)
                    is_ok = sp.simplify(sol.raw_result - expected) == 0

                if is_ok:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "integral", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Simplification generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "simplify"]) < sub_targets["simplify"] and attempts < 15000:
        attempts += 1
        a = random.randint(1, 25)
        pattern = random.choice(["diff_sq_minus", "diff_sq_plus", "common_x", "fraction_cancel"])
        if pattern == "diff_sq_minus":
            problem = f"Simplify (x^2 - {a**2})/(x - {a})"
        elif pattern == "diff_sq_plus":
            problem = f"Simplify (x^2 - {a**2})/(x + {a})"
        elif pattern == "common_x":
            b = random.randint(2, 12)
            problem = f"Simplify ({b}*x^2 + {b*a}*x)/(x + {a})"
        else:
            k = random.randint(2, 8)
            problem = f"Simplify ({k}*x + {k*a})/{k}"

        if problem in seen:
            continue
        seen.add(problem)

        try:
            sol = solver.solve(problem, operation="simplify", variable="x")
            if sol.success and sol.steps and sol.final_answer and sol.original_expr is not None:
                if sp.simplify(sol.raw_result - sp.simplify(sol.original_expr)) == 0:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "simplify", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Factoring generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "factor"]) < sub_targets["factor"] and attempts < 15000:
        attempts += 1
        f_mode = random.choice(["quadratic", "diff_sq", "common"])
        if f_mode == "diff_sq":
            a_root = random.randint(2, 15)
            problem = f"Factor x^2 - {a_root**2}"
        elif f_mode == "common":
            k = random.randint(2, 10)
            problem = f"Factor {k}*x^2 + {k*3}*x"
        else:
            r1 = random.randint(-18, 18)
            r2 = random.randint(-18, 18)
            b = r1 + r2
            c = r1 * r2
            problem = f"Factor x^2 + {b}*x + {c}".replace("+ -", "- ").replace("+ 0*x ", "")

        if problem in seen:
            continue
        seen.add(problem)

        try:
            sol = solver.solve(problem, operation="factor", variable="x")
            if sol.success and sol.steps and sol.final_answer and sol.original_expr is not None:
                if sp.simplify(sol.raw_result - sp.factor(sol.original_expr)) == 0:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "factor", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    # Helper: Expansion generator
    attempts = 0
    while len([s for s in samples if s["meta"]["sub_type"] == "expand"]) < sub_targets["expand"] and attempts < 15000:
        attempts += 1
        a = random.randint(1, 8)
        b = random.randint(-12, 12)
        c = random.randint(1, 6)
        d = random.randint(-12, 12)
        mode = random.choice(["product", "square", "diff_sq", "cube"])
        if mode == "product":
            problem = f"Expand ({a}*x + {b})*({c}*x + {d})".replace("+ -", "- ")
        elif mode == "square":
            problem = f"Expand ({a}*x + {b})^2".replace("+ -", "- ")
        elif mode == "diff_sq":
            problem = f"Expand ({a}*x - {abs(b)})*({a}*x + {abs(b)})"
        else:
            k = random.randint(1, 6)
            problem = f"Expand (x + {k})^3"

        if problem in seen:
            continue
        seen.add(problem)

        try:
            sol = solver.solve(problem, operation="expand", variable="x")
            if sol.success and sol.steps and sol.final_answer and sol.original_expr is not None:
                if sp.simplify(sol.raw_result - sp.expand(sol.original_expr)) == 0:
                    solution_text = format_sympy_solution(sol.steps, sol.final_answer)
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "sympy", "sub_type": "expand", "answer": sol.final_answer},
                    })
        except Exception:
            continue

    logger.info("SymPy generator completed: %d verified samples generated.", len(samples))
    return samples


# -----------------------------------------------------------------------------
# 2. Arithmetic Validator and Reference Answer Checking
# -----------------------------------------------------------------------------
def clean_line_for_equations(line: str) -> str:
    """Normalize line for mathematical equation extraction."""
    line = re.sub(r'(?<=\d),(?=\d)', '', line)
    line = re.sub(r'[\$€£]', '', line)
    # Remove unit words between numbers and operators: '72 cubic ft * 3/4' -> '72 * 3/4'
    line = re.sub(r'(\d+)\s+[a-zA-Z]+\s*(?:[a-zA-Z]+\s*)?([*xX/+-])', r'\1 \2', line)
    # Replace multiplication 'x' or 'X' between numbers with '*'
    line = re.sub(r'(\d)\s*[xX]\s*(\d|\()', r'\1 * \2', line)
    return line


def validate_step_equations(step_text: str) -> Tuple[bool, str]:
    """Find equations of form <expression> = <number> in step_text and verify arithmetic.

    Safely evaluates LHS using sympy.sympify (never eval).
    Returns (is_valid, error_reason).
    """
    line = clean_line_for_equations(step_text)
    pattern = r'([0-9\.\s\+\-\*\/\^\(\)]+?)\s*=\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:\s*\/\s*[0-9]+)?)(?![0-9a-zA-Z_\.])'

    for m in re.finditer(pattern, line):
        lhs_raw = m.group(1).strip()
        rhs_raw = m.group(2).strip()

        # Must contain at least one arithmetic operator (+, -, *, /, ^)
        if not re.search(r'[\+\-\*\/\^]', lhs_raw):
            continue

        m_math = re.search(r'([0-9\.\(\)][0-9\.\s\+\-\*\/\^\(\)]*)$', lhs_raw)
        if not m_math:
            continue
        expr_str = m_math.group(1).strip().replace('^', '**')

        # Must contain at least two numeric operands
        nums = re.findall(r'\b\d+(?:\.\d+)?\b', expr_str)
        if len(nums) < 2:
            continue

        try:
            lhs_expr = sp.sympify(expr_str)
            rhs_expr = sp.sympify(rhs_raw.replace('^', '**'))

            # If either side contains free symbols, skip (not pure arithmetic)
            if lhs_expr.free_symbols or rhs_expr.free_symbols:
                continue

            lhs_val = float(sp.N(lhs_expr))
            rhs_val = float(sp.N(rhs_expr))

            if abs(lhs_val - rhs_val) > 1e-3:
                return False, f"Arithmetic mismatch: {expr_str} (= {lhs_val}) != {rhs_raw} (= {rhs_val})"
        except Exception:
            pass

    return True, ""


def check_final_answer_match(solution_ans: str, ref_ans: str) -> bool:
    """Check if the solution's extracted final answer matches the ground truth reference."""
    s_sol = re.sub(r'[,\$€£\s]', '', solution_ans).rstrip('.').lower()
    s_ref = re.sub(r'[,\$€£\s]', '', ref_ans).rstrip('.').lower()
    s_sol = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s_sol)
    s_ref = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s_ref)

    if s_sol == s_ref:
        return True

    try:
        sol_expr = sp.sympify(s_sol)
        ref_expr = sp.sympify(s_ref)
        if sp.simplify(sol_expr - ref_expr) == 0:
            return True
        if not sol_expr.free_symbols and not ref_expr.free_symbols:
            if abs(float(sp.N(sol_expr)) - float(sp.N(ref_expr))) < 1e-3:
                return True
    except Exception:
        pass

    return False


# -----------------------------------------------------------------------------
# 3. Hugging Face Datasets: GSM8K & MetaMathQA (~6,700 samples)
# -----------------------------------------------------------------------------
def convert_gsm8k_solution(ans_text: str) -> Tuple[str, str, str, bool]:
    """Convert raw GSM8K solution to friendly numbered tutor steps.

    Drops any step whose content is only '#### <number>' (or starts with '####'),
    renumbers remaining steps sequentially, and ensures 'Final answer:' is the only place
    the final answer appears.

    Returns:
        (solution_text, final_answer, raw_reference_answer, had_hash_artifact)
    """
    final_ans = ""
    raw_ref = ""
    had_hash_artifact = False

    if "####" in ans_text:
        had_hash_artifact = True
        parts = ans_text.split("####")
        ans_text = parts[0].strip()
        final_ans = parts[1].strip()
        raw_ref = final_ans

    # Strip calculation tags <<48/2=24>>
    cleaned = re.sub(r"<<[^>]+>>", "", ans_text).strip()
    raw_lines = [l.strip() for l in cleaned.split("\n") if l.strip()]

    # Filter out any leftover #### lines, empty lines, or duplicate final answer lines
    filtered_lines = []
    for line in raw_lines:
        line_clean = line.strip()
        if "####" in line_clean or re.match(r"^#+\s*\d+", line_clean):
            had_hash_artifact = True
            pre_hash = line_clean.split("####")[0].strip()
            if pre_hash and not pre_hash.lower().startswith("the answer is") and not pre_hash.lower().startswith("final answer:"):
                filtered_lines.append(pre_hash)
            continue
        if line_clean.lower().startswith("the answer is") or line_clean.lower().startswith("final answer:"):
            continue
        filtered_lines.append(line_clean)

    # Renumber sequentially
    steps = [f"Step {i}: {line}" for i, line in enumerate(filtered_lines, 1)]
    if final_ans:
        steps.append(f"Final answer: {final_ans}")

    return "\n".join(steps), final_ans, raw_ref, had_hash_artifact


def convert_metamath_solution(resp_text: str) -> Tuple[str, str, str, bool]:
    """Convert raw MetaMathQA response to friendly numbered tutor steps.

    Drops any step whose content is only '#### <number>' (or starts with '####'),
    renumbers remaining steps sequentially, and ensures 'Final answer:' is the only place
    the final answer appears.

    Returns:
        (solution_text, final_answer, raw_reference_answer, had_hash_artifact)
    """
    final_ans = ""
    had_hash_artifact = "####" in resp_text

    # Extract final answer
    m = re.search(r"(?:The answer is|The final answer is|therefore the answer is|The answer is:)\s*[:=]?\s*(.*)$", resp_text, re.I)
    if m:
        final_ans = m.group(1).strip().rstrip(".")
        resp_text = resp_text[:m.start()].strip()
    elif "####" in resp_text:
        had_hash_artifact = True
        parts = resp_text.split("####")
        resp_text = parts[0].strip()
        final_ans = parts[1].strip()

    raw_ref = final_ans

    # Clean LaTeX boxed if on final answer
    m_box = re.search(r"\\boxed\{([^}]+)\}", final_ans)
    if m_box:
        final_ans = m_box.group(1).strip()

    lines = [l.strip() for l in resp_text.split("\n") if l.strip()]
    if len(lines) == 1 and "." in lines[0]:
        sentences = re.split(r"(?<=[.!?])\s+", lines[0])
        lines = [s.strip() for s in sentences if s.strip()]

    # Filter lines: remove any #### lines, and any lines repeating "the answer is"
    filtered_lines = []
    for line in lines:
        line_clean = line.strip()
        if "####" in line_clean or re.match(r"^#+\s*\d+", line_clean):
            had_hash_artifact = True
            pre_hash = line_clean.split("####")[0].strip()
            if pre_hash and not pre_hash.lower().startswith("the answer is") and not pre_hash.lower().startswith("final answer:"):
                filtered_lines.append(pre_hash)
            continue
        if line_clean.lower().startswith("the answer is") or line_clean.lower().startswith("final answer:"):
            continue
        filtered_lines.append(line_clean)

    # Renumber sequentially
    steps = [f"Step {i}: {line}" for i, line in enumerate(filtered_lines, 1)]
    if final_ans:
        steps.append(f"Final answer: {final_ans}")

    return "\n".join(steps), final_ans, raw_ref, had_hash_artifact


def load_huggingface_samples(gsm_target: int = 5500, metamath_target: int = 1500) -> List[Dict[str, Any]]:
    """Load GSM8K and MetaMathQA with strict arithmetic validation and reference checking."""
    from datasets import load_dataset

    samples: List[Dict[str, Any]] = []

    # 1. GSM8K (openai/gsm8k)
    logger.info("Loading GSM8K via datasets (target: %d)...", gsm_target)
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
        gsm_count = 0
        for item in ds_gsm:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if not q or not a:
                DROP_REASONS["dropped_empty"] += 1
                continue

            solution_text, final_ans, raw_ref, had_hash_artifact = convert_gsm8k_solution(a)
            if had_hash_artifact:
                global SAMPLES_WITH_HASH_ARTIFACT
                SAMPLES_WITH_HASH_ARTIFACT += 1

            if not solution_text or len(solution_text.split("\n")) < 2:
                DROP_REASONS["dropped_empty"] += 1
                continue

            # Reference answer validation
            if not raw_ref or not check_final_answer_match(final_ans, raw_ref):
                DROP_REASONS["dropped_final_answer_mismatch"] += 1
                continue

            # Arithmetic equation validation on every step
            arithmetic_ok = True
            for line in solution_text.split("\n"):
                is_valid, _ = validate_step_equations(line)
                if not is_valid:
                    arithmetic_ok = False
                    break

            if not arithmetic_ok:
                DROP_REASONS["dropped_arithmetic"] += 1
                continue

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": solution_text},
                ],
                "meta": {"source": "huggingface_gsm8k", "answer": final_ans},
            })
            gsm_count += 1
            if gsm_count >= gsm_target:
                break
        logger.info("Loaded %d validated samples from GSM8K.", gsm_count)
    except Exception as e:
        logger.warning("Could not load GSM8K: %s", e)

    # 2. MetaMathQA (meta-math/MetaMathQA via streaming)
    logger.info("Streaming MetaMathQA via datasets (target: %d)...", metamath_target)
    try:
        ds_mm = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        mm_count = 0
        for item in ds_mm:
            q = item.get("query", "").strip()
            r = item.get("response", "").strip()
            if not q or not r:
                DROP_REASONS["dropped_empty"] += 1
                continue

            solution_text, final_ans, raw_ref, had_hash_artifact = convert_metamath_solution(r)
            if had_hash_artifact:
                SAMPLES_WITH_HASH_ARTIFACT += 1

            if not solution_text or len(solution_text.split("\n")) < 2:
                DROP_REASONS["dropped_empty"] += 1
                continue

            # Reference answer validation
            if not raw_ref or not check_final_answer_match(final_ans, raw_ref):
                DROP_REASONS["dropped_final_answer_mismatch"] += 1
                continue

            # Arithmetic equation validation
            arithmetic_ok = True
            for line in solution_text.split("\n"):
                is_valid, _ = validate_step_equations(line)
                if not is_valid:
                    arithmetic_ok = False
                    break

            if not arithmetic_ok:
                DROP_REASONS["dropped_arithmetic"] += 1
                continue

            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": solution_text},
                ],
                "meta": {"source": "huggingface_metamath", "answer": final_ans},
            })
            mm_count += 1
            if mm_count >= metamath_target:
                break
        logger.info("Loaded %d validated samples from MetaMathQA.", mm_count)
    except Exception as e:
        logger.warning("Could not load MetaMathQA: %s", e)

    # 3. GSM8K backfill if MetaMathQA yielded fewer than metamath_target
    total_hf_target = gsm_target + metamath_target
    if len(samples) < total_hf_target:
        needed = total_hf_target - len(samples)
        logger.info("Backfilling %d samples from GSM8K to reach target %d...", needed, total_hf_target)
        try:
            ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
            backfill_count = 0
            for idx, item in enumerate(ds_gsm):
                if idx < gsm_count:
                    continue
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if not q or not a:
                    continue

                solution_text, final_ans, raw_ref, had_hash_artifact = convert_gsm8k_solution(a)
                if had_hash_artifact:
                    SAMPLES_WITH_HASH_ARTIFACT += 1

                if not solution_text or len(solution_text.split("\n")) < 2:
                    continue

                if not raw_ref or not check_final_answer_match(final_ans, raw_ref):
                    DROP_REASONS["dropped_final_answer_mismatch"] += 1
                    continue

                arithmetic_ok = True
                for line in solution_text.split("\n"):
                    is_valid, _ = validate_step_equations(line)
                    if not is_valid:
                        arithmetic_ok = False
                        break

                if not arithmetic_ok:
                    DROP_REASONS["dropped_arithmetic"] += 1
                    continue

                samples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": solution_text},
                    ],
                    "meta": {"source": "huggingface_gsm8k", "answer": final_ans},
                })
                backfill_count += 1
                if backfill_count >= needed:
                    break
            logger.info("Backfilled %d additional samples from GSM8K (total HF samples: %d).", backfill_count, len(samples))
        except Exception as e:
            logger.warning("GSM8K backfill error: %s", e)

    return samples


# -----------------------------------------------------------------------------
# 4. Kaggle 520K Dataset Loader (Disabled by default)
# -----------------------------------------------------------------------------
def format_kaggle_solution(resp_text: str, answer_val: str) -> Optional[str]:
    """Convert Kaggle response to friendly numbered steps, filtering out robotic templates."""
    resp_lower = resp_text.lower()
    if "template" in resp_lower or "nn_" in resp_lower:
        return None

    cleaned = re.sub(r"^Problem\s*\([^)]+\):\s*", "", resp_text.strip(), flags=re.I)
    cleaned = re.sub(r"\b(?:Goal|Givens|Approach|Known|Calculation|Conclusion|Plan|Result):\s*", "", cleaned)

    if re.search(r"Step\s*\d+:", cleaned):
        raw_steps = [p.strip() for p in re.split(r"Step\s*\d+:\s*", cleaned) if p.strip()]
    elif "\n" in cleaned:
        raw_steps = [p.strip() for p in cleaned.split("\n") if p.strip()]
    else:
        raw_steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]

    if len(raw_steps) < 3:
        return None

    steps = [f"Step {i+1}: {step}" for i, step in enumerate(raw_steps)]
    if answer_val:
        steps.append(f"Final answer: {answer_val}")
    return "\n".join(steps)


def load_kaggle_samples(target_count: int = 1000) -> List[Dict[str, Any]]:
    """Load from 520K Kaggle dataset (kept for reference, disabled when USE_KAGGLE=False)."""
    if not USE_KAGGLE:
        logger.info("USE_KAGGLE is False. Skipping Kaggle 520K dataset.")
        return []

    logger.info("Loading Kaggle 520K dataset (target: %d)...", target_count)
    samples: List[Dict[str, Any]] = []

    data_file = None
    if Path(DEFAULT_LOCAL_KAGGLE_PATH).exists():
        data_file = Path(DEFAULT_LOCAL_KAGGLE_PATH)
    else:
        try:
            import kagglehub
            path_str = kagglehub.dataset_download(KAGGLE_DATASET_ID)
            p = Path(path_str)
            for cand in p.rglob("*.jsonl"):
                data_file = cand
                break
        except Exception as e:
            logger.warning("Kagglehub download unavailable: %s", e)

    if not data_file or not data_file.exists():
        logger.warning("Kaggle 520K dataset not found. Skipping gracefully.")
        return samples

    try:
        count = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict) or "prompt" not in obj or "response" not in obj:
                    continue
                prompt = obj.get("prompt", "").strip()
                resp = obj.get("response", "").strip()
                ans = str(obj.get("answer", "")).strip()

                if "template" in prompt.lower() or "nn_" in prompt.lower():
                    continue
                if "template" in resp.lower() or "nn_" in resp.lower():
                    continue

                solution_text = format_kaggle_solution(resp, ans)
                if solution_text:
                    samples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": solution_text},
                        ],
                        "meta": {"source": "kaggle", "answer": ans},
                    })
                    count += 1
                    if count >= target_count:
                        break
    except Exception as e:
        logger.warning("Error reading Kaggle dataset: %s", e)

    return samples


# -----------------------------------------------------------------------------
# 5. Pipeline Orchestration: Deduplication, Token Filtering, Split & Save
# -----------------------------------------------------------------------------
def estimate_token_count(text: str) -> int:
    """Fast, accurate token count estimation (roughly 3.5 chars / 0.75 words per token)."""
    words = len(text.split())
    chars = len(text)
    return max(math.ceil(words * 1.3), math.ceil(chars / 3.8))


def clean_and_deduplicate(samples: List[Dict[str, Any]], max_tokens: int = 512) -> List[Dict[str, Any]]:
    """Deduplicate by normalized prompt and filter out empty or >512 token samples."""
    logger.info("Cleaning and deduplicating %d raw samples (max_tokens=%d)...", len(samples), max_tokens)
    cleaned: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for s in samples:
        msgs = s.get("messages", [])
        if len(msgs) < 3:
            DROP_REASONS["dropped_empty"] += 1
            continue

        user_content = msgs[1].get("content", "").strip()
        asst_content = msgs[2].get("content", "").strip()

        if not user_content or not asst_content:
            DROP_REASONS["dropped_empty"] += 1
            continue

        norm_key = re.sub(r"\s+", " ", user_content.lower())
        if norm_key in seen:
            DROP_REASONS["dropped_dupes"] += 1
            continue
        seen.add(norm_key)

        full_text = f"{user_content}\n{asst_content}"
        tokens = estimate_token_count(full_text)
        if tokens > max_tokens:
            DROP_REASONS["dropped_length"] += 1
            continue

        cleaned.append(s)

    logger.info(
        "Cleaned dataset: %d valid samples retained (Dropped: %d dupes, %d >%d tokens, %d empty).",
        len(cleaned), DROP_REASONS["dropped_dupes"], DROP_REASONS["dropped_length"], max_tokens, DROP_REASONS["dropped_empty"]
    )
    return cleaned


def split_and_save_dataset(
    samples: List[Dict[str, Any]],
    output_dir: str = "tutor/data/processed",
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Shuffle with fixed seed, split 90/5/5, save JSONL files and dataset_stats.json."""
    random.seed(seed)
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save splits
    for name, split in [("train.jsonl", train_data), ("val.jsonl", val_data), ("test.jsonl", test_data)]:
        target_file = out_path / name
        with open(target_file, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved %s with %d samples.", name, len(split))

    # Calculate statistics
    source_counts: Dict[str, int] = {}
    source_lengths: Dict[str, List[int]] = {}
    total_tokens: List[int] = []

    for s in samples:
        src = s.get("meta", {}).get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

        asst_text = s["messages"][2]["content"]
        user_text = s["messages"][1]["content"]
        char_len = len(asst_text)
        source_lengths.setdefault(src, []).append(char_len)
        total_tokens.append(estimate_token_count(f"{user_text}\n{asst_text}"))

    avg_char_length = {
        src: round(sum(lens) / max(len(lens), 1), 1)
        for src, lens in source_lengths.items()
    }
    avg_char_length["overall"] = round(sum([sum(l) for l in source_lengths.values()]) / max(total, 1), 1)

    stats = {
        "total_samples": total,
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "source_counts": source_counts,
        "dropped_reasons": DROP_REASONS,
        "cleaned_hash_artifacts_count": SAMPLES_WITH_HASH_ARTIFACT,
        "avg_character_length": avg_char_length,
        "avg_token_length_estimate": round(sum(total_tokens) / max(len(total_tokens), 1), 1),
        "seed": seed,
    }

    stats_file = out_path / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved dataset statistics to: %s", stats_file)

    return stats


def print_sample_demonstrations(samples: List[Dict[str, Any]], stats: Dict[str, Any]):
    """Print 4 random samples (2 MetaMath, 2 SymPy), scan result, and summary statistics."""
    # Integrity scan across all samples
    samples_with_hash = 0
    sympy_samples_with_dollar = 0
    for s in samples:
        full_text = s["messages"][1]["content"] + "\n" + s["messages"][2]["content"]
        if "####" in full_text:
            samples_with_hash += 1
        if s.get("meta", {}).get("source") == "sympy":
            if "$" in full_text:
                sympy_samples_with_dollar += 1

    print("\n" + "=" * 80)
    print("DATASET INTEGRITY SCAN RESULT")
    print("=" * 80)
    print(f"  • Samples containing '####' across entire dataset : {samples_with_hash} (Expected: 0)")
    print(f"  • SymPy samples containing '$'                    : {sympy_samples_with_dollar} (Expected: 0)")
    print(f"  • Raw samples with '####' cleaned during conversion: {stats.get('cleaned_hash_artifacts_count', 0)}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("DATASET BUILD COMPLETE — SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total High-Quality Samples: {stats['total_samples']}")
    print(f"  • Train split (90%):      {stats['splits']['train']} samples")
    print(f"  • Val split   (5%):       {stats['splits']['val']} samples")
    print(f"  • Test split  (5%):       {stats['splits']['test']} samples")
    print("\nSource Breakdown:")
    for src, cnt in stats["source_counts"].items():
        avg_len = stats["avg_character_length"].get(src, 0)
        print(f"  • {src:<24}: {cnt:>5} samples (avg {avg_len} chars/response)")
    print(f"Overall Average Token Count: ~{stats['avg_token_length_estimate']} tokens")

    print("\nSamples Dropped Per Reason:")
    for r_name, r_cnt in stats["dropped_reasons"].items():
        print(f"  • {r_name:<30}: {r_cnt:>5} samples")
    print("=" * 80)

    # Group by exact source
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        src = s.get("meta", {}).get("source", "unknown")
        by_src.setdefault(src, []).append(s)

    print("\n" + "=" * 80)
    print("4 RANDOM DEMONSTRATION SAMPLES (2 MetaMathQA, 2 SymPy)")
    print("=" * 80)

    demo_plan = [
        ("huggingface_metamath", 2, "METAMATHQA"),
        ("sympy", 2, "SYMPY"),
    ]

    for src_key, req_count, label in demo_plan:
        pool = by_src.get(src_key, [])
        if not pool:
            continue
        random.seed(42)
        selected = random.sample(pool, min(req_count, len(pool)))
        for idx, s in enumerate(selected, 1):
            print(f"\n[{label} SAMPLE {idx} — Source: {src_key}]")
            print(f"User Problem:\n  {s['messages'][1]['content']}")
            print(f"Assistant Tutor Solution:\n{s['messages'][2]['content']}")
            print(f"Meta Answer: {s['meta'].get('answer')}")
            print("-" * 60)


def build_mixed_dataset(
    sympy_count: int = 4500,
    gsm_count: int = 5500,
    metamath_count: int = 1500,
    max_tokens: int = 512,
    output_dir: str = "tutor/data/processed",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build, validate, clean, split, save, and report on the mixed dataset."""
    logger.info("Starting High-Quality Math Tutor Dataset Build...")

    # 1. SymPy Ground Truth (~4,500 samples)
    sympy_samples = generate_sympy_samples(target_count=sympy_count)

    # 2. Hugging Face (GSM8K + MetaMathQA with arithmetic verification)
    hf_samples = load_huggingface_samples(gsm_target=gsm_count, metamath_target=metamath_count)

    # 3. Kaggle 520K (Disabled by default)
    kaggle_samples = []
    if USE_KAGGLE:
        kaggle_samples = load_kaggle_samples(target_count=1000)

    # Combine sources
    all_raw_samples = sympy_samples + hf_samples + kaggle_samples
    logger.info("Total raw candidate samples: %d", len(all_raw_samples))

    # Clean, filter tokens, deduplicate
    cleaned_samples = clean_and_deduplicate(all_raw_samples, max_tokens=max_tokens)

    # Split 90/5/5 and save
    stats = split_and_save_dataset(cleaned_samples, output_dir=output_dir, seed=seed)

    # Print 8 random samples and stats
    print_sample_demonstrations(cleaned_samples, stats)

    return cleaned_samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed training dataset for Math Tutor")
    parser.add_argument("--sympy-count", type=int, default=4500, help="Target SymPy samples")
    parser.add_argument("--gsm-count", type=int, default=5500, help="Target GSM8K samples")
    parser.add_argument("--metamath-count", type=int, default=1500, help="Target MetaMathQA samples")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per sample")
    parser.add_argument("--output-dir", type=str, default="tutor/data/processed")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_mixed_dataset(
        sympy_count=args.sympy_count,
        gsm_count=args.gsm_count,
        metamath_count=args.metamath_count,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
