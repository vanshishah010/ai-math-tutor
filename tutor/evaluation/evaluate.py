"""Evaluation Benchmark for AI Math Tutor.

Compares three modes on test benchmarks using SymPy as the ground-truth oracle:
1. BASE model: Un-finetuned base LLM (zero adapter weights).
2. FINE-TUNED: Base LLM + fine-tuned LoRA adapter weights.
3. SYMPY: Rule-based symbolic computation and verification engine.

Key Features:
- LaTeX balanced brace handling (nested \\frac, \\boxed, \\sqrt).
- Currency, units, thousands commas, and trailing punctuation normalization.
- Implicit multiplication support ("16x^2" == "16*x**2").
- Format compliance metric ("Final answer:" detection).
- Per-problem raw model output text and extracted answers saved to results.csv.
- 95% Wilson score confidence intervals for all accuracy figures in results.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tutor.core.explainer import LLMExplainer
from tutor.core.solver import SymPySolver
from tutor.core.tutor import MathTutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_ADAPTER_DIR = "models/qwen2.5-math-tutor-lora"

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

UNITS_PATTERN = re.compile(
    r"\b(?:"
    r"dollars?|cents?|bucks?|usd|eur|gbp|"
    r"hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|"
    r"miles?|mph|km|kilometers?|meters?|metres?|cm|centimeters?|centimetres?|mm|millimeters?|inches?|in|feet|foot|ft|yards?|"
    r"kg|kilograms?|grams?|liters?|litres?|ml|gallons?|pounds?|lbs?|ounces?|oz|"
    r"cookies?|apples?|candies?|candy|pieces?|trips?|points?|items?|people|persons?|students?|cars?|songs?|books?|batches|"
    r"percent(?:age)?|degrees?|deg"
    r")\b|[%°]",
    flags=re.IGNORECASE,
)


def compute_wilson_ci(k: int, n: int, z: float = 1.95996) -> Tuple[float, float, float]:
    """Calculate 95% Wilson score confidence interval for a binomial proportion.

    Returns:
        (accuracy_pct, lower_bound_pct, upper_bound_pct)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denominator = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2))))
    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return round(p * 100.0, 2), round(lower, 2), round(upper, 2)


def check_format_compliance(text: str) -> bool:
    """Check if model output contains the expected 'Final answer:' token."""
    if not text:
        return False
    return bool(re.search(r"Final\s+answer\s*:", text, flags=re.IGNORECASE))


def unpack_latex_macro(text: str, macro: str) -> str:
    """Unpack \\macro{...} by properly counting balanced nested braces."""
    prefix = f"\\{macro}{{"
    while True:
        idx = text.find(prefix)
        if idx == -1:
            break
        start_brace = idx + len(prefix) - 1
        depth = 0
        end_brace = -1
        for i in range(start_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end_brace = i
                    break
        if end_brace == -1:
            break
        inner = text[start_brace + 1:end_brace]
        text = text[:idx] + inner + text[end_brace + 1:]
    return text


def replace_latex_fractions(text: str) -> str:
    """Convert \\frac{a}{b} and \\dfrac{a}{b} into ((a)/(b)), supporting arbitrary nested braces."""
    for macro in ["frac", "dfrac"]:
        prefix = f"\\{macro}{{"
        while True:
            idx = text.find(prefix)
            if idx == -1:
                break
            # Find numerator
            depth = 0
            first_start = idx + len(prefix) - 1
            first_end = -1
            for i in range(first_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        first_end = i
                        break
            if first_end == -1:
                break
            num = text[first_start + 1:first_end]

            # Find denominator starting at next '{'
            rest_idx = first_end + 1
            while rest_idx < len(text) and text[rest_idx].isspace():
                rest_idx += 1
            if rest_idx >= len(text) or text[rest_idx] != "{":
                break
            second_start = rest_idx
            depth = 0
            second_end = -1
            for i in range(second_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        second_end = i
                        break
            if second_end == -1:
                break
            den = text[second_start + 1:second_end]
            text = text[:idx] + f"(({num})/({den}))" + text[second_end + 1:]
    return text


def replace_latex_sqrt(text: str) -> str:
    """Convert \\sqrt{...} into sqrt(...) with balanced braces."""
    prefix = "\\sqrt{"
    while True:
        idx = text.find(prefix)
        if idx == -1:
            break
        start_brace = idx + len(prefix) - 1
        depth = 0
        end_brace = -1
        for i in range(start_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end_brace = i
                    break
        if end_brace == -1:
            break
        inner = text[start_brace + 1:end_brace]
        text = text[:idx] + f"sqrt({inner})" + text[end_brace + 1:]
    return text


def extract_nested_braces(text: str, macro: str) -> Optional[str]:
    """Find first instance of \\macro{...} with balanced braces and return its inner content."""
    prefix = f"\\{macro}{{"
    idx = text.find(prefix)
    if idx == -1:
        return None
    start_brace = idx + len(prefix) - 1
    depth = 0
    for i in range(start_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_brace + 1:i]
    return None


def normalize_math_expression(s: str) -> str:
    """Comprehensive normalization for math strings across all evaluation modes equally.

    - Handles nested LaTeX braces (\\boxed, \\frac, \\sqrt, \\text).
    - Strips currency ($ € £ ¥ ₹), units (dollars, minutes, %, cookies).
    - Removes thousands separators (14,500 -> 14500).
    - Standardizes exponents (^ to **).
    - Strips trailing punctuation and formatting noise.
    """
    if not s:
        return ""

    s = str(s).strip()
    s = s.strip("`*\"'")

    # 1. Unpack boxed, text, and font macros with balanced braces
    for m in ["boxed", "text", "mathrm", "mathbf", "boldsymbol", "operatorname", "underline", "textbf"]:
        s = unpack_latex_macro(s, m)

    # 2. Convert nested LaTeX fractions and square roots
    s = replace_latex_fractions(s)
    s = replace_latex_sqrt(s)

    # 3. Strip $ signs and currency symbols
    s = re.sub(r"[\$€£¥₹]", "", s)

    # 4. Strip units (trailing or standalone words)
    s = UNITS_PATTERN.sub("", s)

    # 5. Strip \left and \right
    s = re.sub(r"\\left|\\right", "", s)

    # 6. Replace symbols
    s = (
        s.replace(r"\cdot", "*")
        .replace(r"\times", "*")
        .replace(r"\div", "/")
        .replace(r"\pi", "pi")
    )

    # 7. Commas inside numbers: 14,500 -> 14500 (preserving comma lists like '6, 19')
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\b|\D))", "", s)

    # 8. Convert exponents: ^{...} to **(...) and ^ to **
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    s = s.replace("^", "**")

    # 9. Clean whitespace and trailing punctuation
    s = s.strip().rstrip(" .;\n\t\r").strip()
    return s


def extract_final_answer(text: str) -> str:
    """Extract final answer string from model output or solver output."""
    if not text:
        return ""

    # 1. Look specifically for "Final answer: <ans>"
    m = re.search(r"Final\s+answer\s*[:=]?\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        ans_line = m.group(1).strip()
        boxed = extract_nested_braces(ans_line, "boxed")
        if boxed:
            return boxed.strip()
        return ans_line.strip()

    # 2. Look for \boxed{...} anywhere in the text
    boxed = extract_nested_braces(text, "boxed")
    if boxed:
        return boxed.strip()

    # 3. Look for "The answer is: <ans>", "The final answer is: <ans>", or "#### <ans>"
    m_ans = re.search(
        r"(?:The answer is|therefore the answer is|The final answer is|####)\s*[:=]?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m_ans:
        ans_line = m_ans.group(1).strip()
        boxed = extract_nested_braces(ans_line, "boxed")
        if boxed:
            return boxed.strip()
        return ans_line.strip()

    # 4. Fallback to trailing numerical or algebraic value
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text)
    if nums:
        return nums[-1].strip()

    # 5. Last non-empty line
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        return lines[-1]

    return ""


def verify_answer(predicted: str, expected: str) -> bool:
    """Verify numeric or symbolic mathematical equivalence using SymPy.

    Applies equal normalization to predicted and expected values:
    - Normalizes LaTeX, units, commas, $ signs, and powers.
    - Resolves implicit multiplication (e.g. 16x^2 == 16*x**2).
    - Resolves variable assignments and chained equalities (e.g. 'x = 71/3 = 23.67' vs '71/3').
    - Compares multi-root lists with set equivalence.
    - Evaluates symbolic difference simplify(p - e) == 0.
    - Evaluates numeric float tolerance (|p - e| < 1e-4).
    """
    if not predicted or not expected:
        return False

    p = normalize_math_expression(predicted)
    e = normalize_math_expression(expected)

    if not p or not e:
        return False

    # 1. Direct normalized string match
    if p.lower() == e.lower():
        return True

    # 2. Variable assignments and chained equality like 'x = 71/3 = 23.67' vs '71/3'
    if "=" in p or "=" in e:
        p_candidates = [
            part.strip()
            for part in p.split("=")
            if part.strip() and not (len(part.strip()) == 1 and part.strip().isalpha())
        ]
        e_candidates = [
            part.strip()
            for part in e.split("=")
            if part.strip() and not (len(part.strip()) == 1 and part.strip().isalpha())
        ]
        if not p_candidates:
            p_candidates = [p]
        if not e_candidates:
            e_candidates = [e]
        if len(p_candidates) > 1 or len(e_candidates) > 1:
            for pc in p_candidates:
                for ec in e_candidates:
                    if verify_answer(pc, ec):
                        return True

    p_no_var = re.sub(r"^[a-zA-Z]\s*=\s*", "", p).strip()
    e_no_var = re.sub(r"^[a-zA-Z]\s*=\s*", "", e).strip()
    if p_no_var.lower() == e_no_var.lower():
        return True

    # 3. Multiple roots / list of values (e.g., 'x = 6, x = 19' vs '6, 19' or '19, 6')
    if ("," in p or "," in e) and not re.search(r"\d,\d{3}", p + e):
        p_items = [
            normalize_math_expression(re.sub(r"^[a-zA-Z]\s*=\s*", "", x).strip())
            for x in p.split(",")
            if x.strip()
        ]
        e_items = [
            normalize_math_expression(re.sub(r"^[a-zA-Z]\s*=\s*", "", x).strip())
            for x in e.split(",")
            if x.strip()
        ]
        if len(p_items) == len(e_items) and len(p_items) > 1:
            matched = [False] * len(p_items)
            for ei in e_items:
                found = False
                for j, pi in enumerate(p_items):
                    if not matched[j] and verify_answer(pi, ei):
                        matched[j] = True
                        found = True
                        break
                if not found:
                    break
            if all(matched):
                return True

    # 4. SymPy symbolic equivalence with implicit multiplication support (16x^2 == 16*x**2)
    try:
        p_sym = parse_expr(p_no_var, transformations=TRANSFORMATIONS)
        e_sym = parse_expr(e_no_var, transformations=TRANSFORMATIONS)
        if sp.simplify(p_sym - e_sym) == 0:
            return True
    except Exception:
        pass

    # 5. Floating point numeric equivalence
    try:
        p_sym = parse_expr(p_no_var, transformations=TRANSFORMATIONS)
        e_sym = parse_expr(e_no_var, transformations=TRANSFORMATIONS)
        p_val = float(sp.N(p_sym))
        e_val = float(sp.N(e_sym))
        if abs(p_val - e_val) < 1e-4:
            return True
    except Exception:
        pass

    return False


class Evaluator:
    """Benchmark evaluation coordinator across BASE, FINE-TUNED, and SYMPY modes."""

    def __init__(
        self,
        test_file: str = "tutor/data/processed/test.jsonl",
        adapter_path: Optional[str] = DEFAULT_ADAPTER_DIR,
        device: str = "auto",
        eval_gsm8k_official: bool = False,
    ) -> None:
        self.test_file = Path(test_file)
        self.adapter_path = adapter_path
        self.device = device
        self.eval_gsm8k_official = eval_gsm8k_official

        self.solver = SymPySolver()
        # Explainer instances (initialized lazily)
        self._base_explainer: Optional[LLMExplainer] = None
        self._finetuned_explainer: Optional[LLMExplainer] = None

    def get_base_explainer(self) -> LLMExplainer:
        """Get or initialize pure base LLM explainer (no adapter)."""
        if self._base_explainer is None:
            self._base_explainer = LLMExplainer(adapter_path=None, device=self.device)
        return self._base_explainer

    def get_finetuned_explainer(self) -> LLMExplainer:
        """Get or initialize fine-tuned LLM explainer (with LoRA adapter)."""
        if self._finetuned_explainer is None:
            self._finetuned_explainer = LLMExplainer(
                adapter_path=self.adapter_path, device=self.device
            )
        return self._finetuned_explainer

    def load_test_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load test dataset from processed JSONL or official GSM8K benchmark."""
        samples: List[Dict[str, Any]] = []

        if self.eval_gsm8k_official:
            logger.info("Loading official openai/gsm8k test split (first 300 problems)...")
            try:
                from datasets import load_dataset

                ds = load_dataset("openai/gsm8k", "main", split="test")
                for idx, item in enumerate(ds):
                    if idx >= 300:
                        break
                    q = item.get("question", "").strip()
                    ans = item.get("answer", "").strip()
                    ref = ans.split("####")[1].strip() if "####" in ans else ans
                    samples.append(
                        {
                            "problem": q,
                            "expected_answer": ref,
                            "source": "gsm8k_official",
                        }
                    )
                logger.info(
                    "Loaded %d official held-out GSM8K benchmark problems.", len(samples)
                )
            except Exception as e:
                logger.error("Failed to load official GSM8K benchmark: %s", e)
        else:
            if not self.test_file.exists():
                raise FileNotFoundError(
                    f"Test file not found at {self.test_file}. "
                    f"Please run `python tutor/data/build_dataset.py` first."
                )

            with open(self.test_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    messages = item.get("messages", [])
                    user_prob = ""
                    for msg in messages:
                        if msg["role"] == "user":
                            user_prob = msg["content"]
                            break

                    ref_ans = item.get("meta", {}).get("answer", "")
                    src = item.get("meta", {}).get("source", "unknown")

                    if user_prob:
                        samples.append(
                            {
                                "problem": user_prob,
                                "expected_answer": str(ref_ans).strip(),
                                "source": src,
                            }
                        )

        if limit is not None and limit > 0:
            samples = samples[:limit]

        logger.info("Total evaluation problems selected: %d", len(samples))
        return samples

    def evaluate_sample_sympy(self, problem: str) -> Tuple[str, str]:
        """Solve problem using SymPy symbolic solver. Returns (predicted_ans, raw_output)."""
        clean_p = problem.strip()
        clean_p = re.sub(
            r"^(?:Solve\s+for\s+[a-zA-Z]:\s*)", "", clean_p, flags=re.I
        ).strip()
        try:
            sol = self.solver.solve(clean_p)
            if sol.success and sol.final_answer:
                steps_txt = "\n".join(
                    f"Step {s.step_number}: {s.title} — {s.description}"
                    for s in sol.steps
                )
                raw_out = (
                    f"{steps_txt}\nFinal answer: {sol.final_answer}"
                    if steps_txt
                    else f"Final answer: {sol.final_answer}"
                )
                return sol.final_answer, raw_out
            else:
                return "", "Could not parse or solve problem."
        except Exception as e:
            return "", f"SymPy error: {e}"

    def evaluate_sample_base_llm(self, problem: str) -> Tuple[str, str]:
        """Solve problem using Base LLM directly. Returns (predicted_ans, raw_generation)."""
        explainer = self.get_base_explainer()
        raw_text = explainer.generate_solution(problem)
        pred = extract_final_answer(raw_text)
        return pred, raw_text

    def evaluate_sample_finetuned_llm(self, problem: str) -> Tuple[str, str]:
        """Solve problem using Fine-Tuned LoRA LLM directly. Returns (predicted_ans, raw_generation)."""
        explainer = self.get_finetuned_explainer()
        raw_text = explainer.generate_solution(problem)
        pred = extract_final_answer(raw_text)
        return pred, raw_text

    def run_benchmark(
        self,
        mode: str = "all",
        limit: Optional[int] = None,
        output_csv: str = "tutor/evaluation/results.csv",
        output_md: str = "tutor/evaluation/results.md",
    ) -> Dict[str, Any]:
        """Execute evaluation across selected modes and save CSV / MD reports."""
        samples = self.load_test_samples(limit=limit)
        total = len(samples)
        if total == 0:
            logger.warning("No samples found to evaluate.")
            return {}

        active_modes = []
        if mode == "all":
            active_modes = ["sympy", "base", "fine-tuned"]
        else:
            active_modes = [mode.lower()]

        logger.info(
            "Starting evaluation benchmark on %d problems across modes: %s",
            total,
            active_modes,
        )

        # Metrics per mode:
        # {m: {"correct": int, "compliant": int, "by_source": {src: {"correct": int, "compliant": int, "total": int}}}}
        metrics: Dict[str, Dict[str, Any]] = {
            m: {"correct": 0, "compliant": 0, "by_source": {}} for m in active_modes
        }

        # Store all per-problem evaluation records for CSV output
        all_eval_records: List[Dict[str, Any]] = []

        for idx, s in enumerate(samples, 1):
            prob = s["problem"]
            expected = s["expected_answer"]
            src = s["source"]

            rec: Dict[str, Any] = {
                "id": idx,
                "problem": prob,
                "expected": expected,
                "source": src,
                "predictions": {},
                "raw_outputs": {},
                "is_correct": {},
                "format_compliant": {},
            }

            for m in active_modes:
                src_stat = metrics[m]["by_source"].setdefault(
                    src, {"correct": 0, "compliant": 0, "total": 0}
                )
                src_stat["total"] += 1

                pred = ""
                raw_out = ""

                if m == "sympy":
                    pred, raw_out = self.evaluate_sample_sympy(prob)
                elif m == "base":
                    pred, raw_out = self.evaluate_sample_base_llm(prob)
                elif m == "fine-tuned":
                    pred, raw_out = self.evaluate_sample_finetuned_llm(prob)

                # Format compliance check
                is_compliant = check_format_compliance(raw_out)
                if is_compliant:
                    metrics[m]["compliant"] += 1
                    src_stat["compliant"] += 1

                # Answer accuracy check (applying equal normalization)
                is_ok = verify_answer(pred, expected)
                if is_ok:
                    metrics[m]["correct"] += 1
                    src_stat["correct"] += 1

                rec["predictions"][m] = pred
                rec["raw_outputs"][m] = raw_out
                rec["is_correct"][m] = is_ok
                rec["format_compliant"][m] = is_compliant

            all_eval_records.append(rec)

            if idx % 20 == 0 or idx == total:
                logger.info("Progress: %d / %d problems evaluated.", idx, total)

        # Compile Summary Results with 95% Wilson Confidence Intervals
        summary: Dict[str, Any] = {
            "total_samples": total,
            "modes": {},
            "all_eval_records": all_eval_records,
            "side_by_side_examples": all_eval_records[:5],
        }

        for m in active_modes:
            corr = metrics[m]["correct"]
            compl = metrics[m]["compliant"]
            acc, acc_ci_low, acc_ci_high = compute_wilson_ci(corr, total)
            compl_pct, compl_ci_low, compl_ci_high = compute_wilson_ci(compl, total)

            src_breakdown = {}
            for src, stat in metrics[m]["by_source"].items():
                s_tot = stat["total"]
                s_cor = stat["correct"]
                s_cmp = stat["compliant"]
                s_acc, s_ci_low, s_ci_high = compute_wilson_ci(s_cor, s_tot)
                s_cmp_pct, _, _ = compute_wilson_ci(s_cmp, s_tot)
                src_breakdown[src] = {
                    "correct": s_cor,
                    "compliant": s_cmp,
                    "total": s_tot,
                    "accuracy": s_acc,
                    "accuracy_ci": (s_ci_low, s_ci_high),
                    "compliance": s_cmp_pct,
                }

            summary["modes"][m] = {
                "correct": corr,
                "compliant": compl,
                "total": total,
                "accuracy": acc,
                "accuracy_ci": (acc_ci_low, acc_ci_high),
                "compliance": compl_pct,
                "compliance_ci": (compl_ci_low, compl_ci_high),
                "by_source": src_breakdown,
            }

        # Save to CSV and Markdown
        self.save_reports(summary, output_csv=output_csv, output_md=output_md)
        self.print_summary(summary)

        return summary

    def save_reports(
        self, summary: Dict[str, Any], output_csv: str, output_md: str
    ) -> None:
        """Save results tables to CSV (with per-problem raw output) and formatted Markdown report."""
        out_csv_path = Path(output_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path = Path(output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)

        modes_data = summary.get("modes", {})
        active_modes = list(modes_data.keys())
        total = summary.get("total_samples", 0)
        records = summary.get("all_eval_records", [])

        # Collect distinct problem sources
        all_sources = set()
        for m_data in modes_data.values():
            all_sources.update(m_data.get("by_source", {}).keys())
        sorted_sources = sorted(list(all_sources))

        # 1. Write per-problem records to output_csv
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Problem ID", "Source", "Problem", "Expected Answer"]
            for m in active_modes:
                m_name = m.upper()
                header.extend(
                    [
                        f"{m_name} Extracted Answer",
                        f"{m_name} Raw Output",
                        f"{m_name} Correct",
                        f"{m_name} Format Compliant",
                    ]
                )
            writer.writerow(header)

            for rec in records:
                row = [
                    rec["id"],
                    rec["source"],
                    rec["problem"],
                    rec["expected"],
                ]
                for m in active_modes:
                    row.extend(
                        [
                            rec["predictions"].get(m, ""),
                            rec["raw_outputs"].get(m, ""),
                            rec["is_correct"].get(m, False),
                            rec["format_compliant"].get(m, False),
                        ]
                    )
                writer.writerow(row)
        logger.info(
            "Saved %d per-problem evaluation rows (with raw output) to CSV: %s",
            len(records),
            out_csv_path,
        )

        # Also save aggregate summary CSV
        summary_csv_path = out_csv_path.with_name(out_csv_path.stem + "_summary.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            sum_header = [
                "Mode",
                "Total",
                "Correct",
                "Overall Accuracy (%)",
                "95% CI Lower (%)",
                "95% CI Upper (%)",
                "Format Compliance (%)",
            ] + [f"{src} Acc (%)" for src in sorted_sources]
            writer.writerow(sum_header)

            for m, m_data in modes_data.items():
                ci_low, ci_high = m_data.get("accuracy_ci", (0.0, 0.0))
                row = [
                    m.upper(),
                    m_data["total"],
                    m_data["correct"],
                    f"{m_data['accuracy']:.2f}",
                    f"{ci_low:.2f}",
                    f"{ci_high:.2f}",
                    f"{m_data['compliance']:.2f}",
                ]
                for src in sorted_sources:
                    src_info = m_data.get("by_source", {}).get(src, {})
                    row.append(f"{src_info.get('accuracy', 0.0):.2f}")
                writer.writerow(row)
        logger.info("Saved aggregate evaluation summary CSV to: %s", summary_csv_path)

        # 2. Write Markdown Report with 95% Wilson Confidence Intervals
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("# AI Math Tutor Evaluation Report\n\n")
            f.write(f"**Total Test Problems Evaluated:** {total}\n\n")

            f.write("## Overall Benchmark Comparison\n\n")
            table_header = (
                "| Mode | Correct / Total | Overall Accuracy (95% CI) | Format Compliance (%) | "
                + " | ".join(f"{s} Acc (95% CI)" for s in sorted_sources)
                + " |\n"
            )
            table_sep = (
                "| :--- | :---: | :---: | :---: | "
                + " | ".join(":---:" for _ in sorted_sources)
                + " |\n"
            )
            f.write(table_header)
            f.write(table_sep)

            for m, m_data in modes_data.items():
                ci_low, ci_high = m_data.get("accuracy_ci", (0.0, 0.0))
                acc_ci_str = f"**{m_data['accuracy']:.2f}%** [{ci_low:.1f}%, {ci_high:.1f}%]"
                compl_str = f"{m_data['compliance']:.2f}% ({m_data['compliant']}/{m_data['total']})"

                src_cols = []
                for s in sorted_sources:
                    s_info = m_data.get("by_source", {}).get(s, {})
                    s_acc = s_info.get("accuracy", 0.0)
                    s_low, s_high = s_info.get("accuracy_ci", (0.0, 0.0))
                    src_cols.append(f"{s_acc:.1f}% [{s_low:.1f}%, {s_high:.1f}%]")

                f.write(
                    f"| **{m.upper()}** | {m_data['correct']} / {m_data['total']} | {acc_ci_str} | {compl_str} | "
                    + " | ".join(src_cols)
                    + " |\n"
                )

            f.write("\n*Note: All confidence intervals are 95% Wilson score intervals.*\n\n")

            f.write("## Side-by-Side Qualitative Examples (5 Selected)\n\n")
            for ex in summary.get("side_by_side_examples", []):
                f.write(f"### Example {ex['id']} — [{ex['source']}]\n\n")
                f.write(f"**Problem:**\n> {ex['problem']}\n\n")
                f.write(f"**Expected Ground Truth Answer:** `{ex['expected']}`\n\n")
                f.write(
                    "| Evaluation Mode | Model Predicted Answer | Format Compliant? | SymPy Verified? |\n"
                )
                f.write("| :--- | :--- | :---: | :---: |\n")
                for m in active_modes:
                    pred = ex["predictions"].get(m, "")
                    ok_str = (
                        "PASSED (Correct)"
                        if ex["is_correct"].get(m, False)
                        else "FAILED (Incorrect)"
                    )
                    compl_str = "Yes" if ex["format_compliant"].get(m, False) else "No"
                    f.write(
                        f"| {m.upper()} | `{pred or '(no answer extracted)'}` | {compl_str} | {ok_str} |\n"
                    )

                for m in active_modes:
                    raw_text = ex["raw_outputs"].get(m, "")
                    if raw_text:
                        f.write(
                            f"\n<details>\n<summary>View {m.upper()} Raw Output</summary>\n\n```\n{raw_text}\n```\n</details>\n"
                        )
                f.write("\n---\n\n")

        logger.info("Saved evaluation Markdown report to: %s", out_md_path)

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print readable console summary."""
        print("\n" + "=" * 80)
        print("EVALUATION BENCHMARK RESULTS")
        print("=" * 80)
        total = summary.get("total_samples", 0)
        print(f"Total Problems Evaluated: {total}")

        modes_data = summary.get("modes", {})
        for m, data in modes_data.items():
            ci_low, ci_high = data.get("accuracy_ci", (0.0, 0.0))
            print(f"\n[{m.upper()} MODE]")
            print(
                f"  • Overall Accuracy   : {data['accuracy']:.2f}% ({data['correct']}/{data['total']}) [95% CI: {ci_low:.1f}%, {ci_high:.1f}%]"
            )
            print(
                f"  • Format Compliance  : {data['compliance']:.2f}% ({data['compliant']}/{data['total']})"
            )
            print("  • Accuracy by Source :")
            for src, s_stat in data.get("by_source", {}).items():
                s_low, s_high = s_stat.get("accuracy_ci", (0.0, 0.0))
                print(
                    f"     - {src:<24}: {s_stat['accuracy']:>5.1f}% ({s_stat['correct']}/{s_stat['total']}) [95% CI: {s_low:.1f}%, {s_high:.1f}%]"
                )

        print("\n" + "=" * 80)
        print("SIDE-BY-SIDE QUALITATIVE COMPARISONS")
        print("=" * 80)
        for ex in summary.get("side_by_side_examples", []):
            print(f"\n[Problem {ex['id']} — {ex['source']}]")
            print(f"Problem : {ex['problem']}")
            print(f"Expected: {ex['expected']}")
            for m in summary.get("modes", {}).keys():
                pred = ex["predictions"].get(m, "")
                mark = "CORRECT" if ex["is_correct"].get(m, False) else "INCORRECT"
                fc = "YES" if ex["format_compliant"].get(m, False) else "NO"
                print(
                    f"  • {m.upper():<12}: '{pred}' [{mark}] (Format Compliant: {fc})"
                )
            print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AI Math Tutor on Benchmark Test Set"
    )
    parser.add_argument(
        "--test-file", type=str, default="tutor/data/processed/test.jsonl"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "base", "fine-tuned", "sympy"],
        default="all",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Max test samples to evaluate"
    )
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--gsm8k-official",
        action="store_true",
        help="Evaluate on first 300 problems of official GSM8K test split",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-csv", type=str, default="tutor/evaluation/results.csv"
    )
    parser.add_argument(
        "--output-md", type=str, default="tutor/evaluation/results.md"
    )

    args = parser.parse_args()
    evaluator = Evaluator(
        test_file=args.test_file,
        adapter_path=args.adapter_path,
        device=args.device,
        eval_gsm8k_official=args.gsm8k_official,
    )
    evaluator.run_benchmark(
        mode=args.mode,
        limit=args.limit,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
