"""Reusable and robust mathematical input parser.

Handles:
- Natural-language query intent extraction (e.g. 'Differentiate y = x^3 sin(x) with respect to x')
- Implicit multiplication (e.g. '3(2x - 4)', '2x + 3', 'x(x + 2)', 'x^3 sin(x)')
- Exponent conversion ('^' to '**')
- LaTeX and backslash sanitization
- Distinguishing equations ('=') from algebraic expressions
- Clean variable extraction and validation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

logger = logging.getLogger(__name__)

# Standard SymPy transformations with implicit multiplication
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class ParsedMathQuery:
    """Structured result of parsing a user's mathematical query."""
    raw_query: str
    cleaned_expression: str
    detected_operation: Optional[str] = None
    variable: str = "x"
    is_equation: bool = False
    parsed_object: Optional[Union[sp.Expr, sp.Eq]] = None
    limits: Optional[Tuple[float, float]] = None
    limits_symbolic: Optional[Tuple[sp.Expr, sp.Expr]] = None
    success: bool = True
    error_message: Optional[str] = None


class MathParser:
    """Comprehensive mathematical input parser."""

    COMMON_SYMBOLS = ["x", "y", "z", "t", "u", "v", "a", "b", "c", "n", "k", "m", "r", "theta"]

    @classmethod
    def get_symbol_dict(cls, variable: str = "x") -> Dict[str, Any]:
        """Build dictionary of allowed mathematical symbols and functions."""
        v_sym = sp.Symbol(variable.strip() or "x")
        d: Dict[str, Any] = {
            str(v_sym): v_sym,
            "x": sp.Symbol("x"),
            "y": sp.Symbol("y"),
            "z": sp.Symbol("z"),
            "t": sp.Symbol("t"),
            "u": sp.Symbol("u"),
            "v": sp.Symbol("v"),
            "a": sp.Symbol("a"),
            "b": sp.Symbol("b"),
            "c": sp.Symbol("c"),
            "n": sp.Symbol("n"),
            "k": sp.Symbol("k"),
            "pi": sp.pi,
            "E": sp.E,
            "e": sp.E,
            "I": sp.I,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sec": sp.sec,
            "csc": sp.csc,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sqrt": sp.sqrt,
            "cbrt": sp.cbrt,
            "Abs": sp.Abs,
            "abs": sp.Abs,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
        }
        return d

    # Mapping of unicode superscript characters to standard ASCII digits/operators
    SUPERSCRIPT_MAP = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
        "⁺": "+", "⁻": "-", "ⁿ": "n", "ᵐ": "m", "ᵏ": "k",
        "⁽": "(", "⁾": ")",
    }

    @classmethod
    def sanitize_math_string(cls, raw: str) -> str:
        """Sanitize raw text: strip LaTeX commands, backslashes, unicode symbols, and superscripts."""
        s = raw.strip()
        # Remove any LaTeX backslashes or line continuations
        s = s.replace(r"\left", "").replace(r"\right", "")
        s = s.replace(r"\cdot", "*").replace(r"\times", "*")
        s = s.replace(r"\sin", "sin").replace(r"\cos", "cos").replace(r"\tan", "tan")
        s = s.replace(r"\exp", "exp").replace(r"\log", "log").replace(r"\ln", "ln")
        s = s.replace(r"\sqrt", "sqrt").replace(r"\pi", "pi")

        # Convert simple \frac{a}{b} -> (a)/(b)
        frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
        while frac_pattern.search(s):
            s = frac_pattern.sub(r"((\1)/(\2))", s)

        # Strip remaining backslashes to prevent 'unexpected character after line continuation'
        s = s.replace("\\", "")

        # Unicode math replacements
        s = s.replace("–", "-").replace("—", "-").replace("−", "-")
        s = s.replace("×", "*").replace("·", "*").replace("÷", "/")
        s = s.replace("√", "sqrt")

        # Convert unicode superscripts (e.g. x², x³, x⁴) to explicit exponent notation **(...)
        def _sub_superscripts(m):
            converted = "".join(cls.SUPERSCRIPT_MAP.get(c, c) for c in m.group(0))
            return f"**({converted})"
        s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵐᵏ⁽⁾]+", _sub_superscripts, s)

        s = s.replace("^", "**")

        # Remove extra whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def extract_intent(cls, raw: str, default_var: str = "x") -> Tuple[Optional[str], str, Optional[Tuple[sp.Expr, sp.Expr]], str]:
        """Extract operation intent, variable, definite integral limits, and pure expression.

        Returns:
            (operation, variable, limits, pure_expression)
        """
        s = cls.sanitize_math_string(raw)
        detected_op: Optional[str] = None
        var: str = default_var
        limits: Optional[Tuple[sp.Expr, sp.Expr]] = None

        # 1. Check for definite integral limits: 'from a to b' or 'between a and b'
        lim_m = re.search(r"\b(?:from|between)\s+([-\d\.\/]+)\s+(?:to|and)\s+([-\d\.\/]+)\b", s, flags=re.I)
        if lim_m:
            try:
                a_expr = parse_expr(lim_m.group(1).strip(), transformations=TRANSFORMATIONS)
                b_expr = parse_expr(lim_m.group(2).strip(), transformations=TRANSFORMATIONS)
                limits = (a_expr, b_expr)
                detected_op = "integrate"
                s = s[:lim_m.start()] + " " + s[lim_m.end():]
            except Exception:
                pass

        # 2. Check for differentiation variable: e.g. 'dy/dx', 'd/dx', 'with respect to x', 'wrt x'
        d_var_m = re.search(r"\bd[a-zA-Z]?/d([a-zA-Z])\b", s, flags=re.I)
        if d_var_m:
            var = d_var_m.group(1).lower()
            detected_op = "diff"

        wrt_m = re.search(r"\b(?:with\s+respect\s+to|w\.?r\.?t\.?)\s+([a-zA-Z])\b", s, flags=re.I)
        if wrt_m:
            var = wrt_m.group(1).lower()
            s = s[:wrt_m.start()] + " " + s[wrt_m.end():]

        # 3. Check for 'solve ... for <var>'
        for_var_m = re.search(r"\bfor\s+([a-zA-Z])\b", s, flags=re.I)
        if for_var_m and not s.strip().lower().startswith("find"):
            var = for_var_m.group(1).lower()
            s = s[:for_var_m.start()] + " " + s[for_var_m.end():]

        # 4. Check operation keywords
        if not detected_op:
            if re.search(r"\b(?:differentiate|derivative|diff|dy/dx|d/dx)\b", s, flags=re.I):
                detected_op = "diff"
            elif re.search(r"\b(?:integrate|integral|antiderivative)\b", s, flags=re.I):
                detected_op = "integrate"
            elif re.search(r"\b(?:expand)\b", s, flags=re.I):
                detected_op = "expand"
            elif re.search(r"\b(?:factor|factorise|factorize)\b", s, flags=re.I):
                detected_op = "factor"
            elif re.search(r"\b(?:simplify)\b", s, flags=re.I):
                detected_op = "simplify"
            elif re.search(r"\b(?:plot|graph)\b", s, flags=re.I):
                detected_op = "graph"
            elif re.search(r"\b(?:solve)\b", s, flags=re.I):
                detected_op = "solve"

        # 5. Clean natural-language command phrases from the expression
        s = re.sub(
            r"\b(?:find\s+)?(?:the\s+)?(?:derivative\s+of|differentiate|diff|find\s+dy/dx\s+for|dy/dx|d/dx)\b",
            "",
            s,
            flags=re.I,
        )
        s = re.sub(
            r"\b(?:find\s+)?(?:the\s+)?(?:integral\s+of|integrate|antiderivative\s+of)\b",
            "",
            s,
            flags=re.I,
        )
        s = re.sub(r"\b(?:simplify|factorise|factorize|factor|expand|solve|plot|graph)\b", "", s, flags=re.I)

        s = s.strip()

        # 6. If operation is diff/integrate/simplify/expand/factor/graph, and there is 'y = ...' or 'f(x) = ...', strip LHS
        if detected_op in ("diff", "integrate", "simplify", "expand", "factor", "graph"):
            eq_m = re.match(r"^\s*(?:[yY]|[fF]\s*\([a-zA-Z]\))\s*=\s*(.*)$", s)
            if eq_m:
                s = eq_m.group(1).strip()

        # Also strip any leading 'for y = ...'
        s = re.sub(r"^\s*for\s+[a-zA-Z]\s*=\s*", "", s, flags=re.I)

        return detected_op, var, limits, s.strip()

    @classmethod
    def apply_implicit_multiplication_patterns(cls, s: str) -> str:
        """Add multiplication asterisks for edge cases before SymPy parsing."""
        # e.g., '3(2x - 4)' -> '3*(2x - 4)'
        s = re.sub(r"(\d)\s*\(", r"\1*(", s)
        # e.g., ')(' -> ')*('
        s = re.sub(r"\)\s*\(", r")*(", s)
        # e.g., ')x' or ')2'
        s = re.sub(r"\)\s*([a-zA-Z0-9])", r")*\1", s)

        # e.g., 'x(x+2)' -> 'x*(x+2)' while keeping 'sin(x)', 'cos(x)', etc.
        known_functions = {
            "sin", "cos", "tan", "cot", "sec", "csc",
            "asin", "acos", "atan", "exp", "log", "ln",
            "sqrt", "cbrt", "abs", "sinh", "cosh", "tanh"
        }

        def _insert_mult(m: re.Match) -> str:
            word = m.group(1)
            if word.lower() in known_functions:
                return f"{word}("
            # If word is single variable like 'x' or 'y'
            return f"{word}*("

        s = re.sub(r"\b([a-zA-Z]+)\s*\(", _insert_mult, s)
        return s

    @classmethod
    def parse(
        cls,
        raw_query: str,
        default_operation: str = "auto",
        default_variable: str = "x",
    ) -> ParsedMathQuery:
        """Parse raw user mathematical query into SymPy expression or equation.

        Args:
            raw_query: User input string.
            default_operation: Operation if not detected ('auto', 'solve', etc.).
            default_variable: Variable symbol name.

        Returns:
            ParsedMathQuery object.
        """
        if not raw_query or not raw_query.strip():
            return ParsedMathQuery(
                raw_query=raw_query,
                cleaned_expression="",
                success=False,
                error_message="Input is empty. Please enter a mathematical expression or equation.",
            )

        detected_op, var, limits, clean_expr = cls.extract_intent(raw_query, default_var=default_variable)
        op = detected_op or default_operation

        if not clean_expr:
            return ParsedMathQuery(
                raw_query=raw_query,
                cleaned_expression="",
                detected_operation=op,
                variable=var,
                success=False,
                error_message="Could not extract a mathematical expression from the input.",
            )

        # Apply implicit multiplication formatting
        prepared_expr = cls.apply_implicit_multiplication_patterns(clean_expr)

        # Build symbol dictionary for the target variable and common symbols
        local_dict = cls.get_symbol_dict(variable=var)

        try:
            # Check if this is an equation with '='
            if "=" in prepared_expr and op not in ("diff", "integrate", "simplify", "expand", "factor", "graph"):
                parts = prepared_expr.split("=")
                if len(parts) != 2:
                    raise ValueError(f"Invalid equation containing multiple '=' symbols: '{clean_expr}'")
                lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
                lhs = parse_expr(lhs_str, local_dict=local_dict, transformations=TRANSFORMATIONS)
                rhs = parse_expr(rhs_str, local_dict=local_dict, transformations=TRANSFORMATIONS)
                eq = sp.Eq(lhs, rhs)

                # Determine active variable from equation's free symbols if var not in equation
                eq_free_syms = [s.name for s in eq.free_symbols]
                if var not in eq_free_syms and eq_free_syms:
                    var = eq_free_syms[0]

                return ParsedMathQuery(
                    raw_query=raw_query,
                    cleaned_expression=clean_expr,
                    detected_operation="solve" if op == "auto" else op,
                    variable=var,
                    is_equation=True,
                    parsed_object=eq,
                    limits=None,
                    success=True,
                )
            else:
                # Parse as single algebraic expression
                expr_to_parse = prepared_expr
                if "=" in expr_to_parse:
                    # If an unexpected '=' is still present in non-solve mode, take the RHS
                    expr_to_parse = expr_to_parse.split("=")[-1].strip()

                expr = parse_expr(expr_to_parse, local_dict=local_dict, transformations=TRANSFORMATIONS)

                # Determine variable from free symbols if default not found
                free_sym_names = [s.name for s in expr.free_symbols]
                if var not in free_sym_names and free_sym_names:
                    # Pick most common math variable if present
                    candidates = ["x", "t", "y", "u", "z", "n"]
                    found_cand = next((c for c in candidates if c in free_sym_names), free_sym_names[0])
                    var = found_cand

                # Determine operation if still 'auto'
                final_op = op
                if final_op == "auto":
                    final_op = "simplify"

                # Numeric limits conversion if present
                limits_numeric = None
                if limits:
                    try:
                        limits_numeric = (float(sp.N(limits[0])), float(sp.N(limits[1])))
                    except Exception:
                        limits_numeric = None

                return ParsedMathQuery(
                    raw_query=raw_query,
                    cleaned_expression=clean_expr,
                    detected_operation=final_op,
                    variable=var,
                    is_equation=False,
                    parsed_object=expr,
                    limits=limits_numeric,
                    limits_symbolic=limits,
                    success=True,
                )

        except Exception as e:
            logger.warning("MathParser failed on '%s': %s", raw_query, e)
            return ParsedMathQuery(
                raw_query=raw_query,
                cleaned_expression=clean_expr,
                detected_operation=op,
                variable=var,
                success=False,
                error_message=f"Unable to parse mathematical expression: {e}",
            )

    @classmethod
    def parse_expression(cls, expr_str: str, variable: Union[str, sp.Symbol] = "x") -> Optional[sp.Expr]:
        """Convenience helper to parse an expression string directly into a SymPy Expr."""
        var_name = variable.name if isinstance(variable, sp.Symbol) else str(variable)
        res = cls.parse(expr_str, default_variable=var_name)
        if res.success and isinstance(res.parsed_object, sp.Expr):
            return res.parsed_object
        return None


UNICODE_SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "n": "ⁿ", "m": "ᵐ", "k": "ᵏ",
    "(": "⁽", ")": "⁾",
}


def to_superscript(exp_str: str) -> str:
    """Convert exponent string into unicode superscript characters."""
    exp_str = exp_str.strip().strip("()")
    if all(c in UNICODE_SUPERSCRIPTS for c in exp_str):
        return "".join(UNICODE_SUPERSCRIPTS[c] for c in exp_str)
    return f"^{exp_str}"


def format_math_display(expr: Any) -> str:
    """Format mathematical expression/equation into clean human-readable notation.

    Eliminates raw SymPy artifacts:
    - Unwraps 'Eq(lhs, rhs)' into 'lhs = rhs'
    - Converts raw powers '**2' or '^2' to floating superscripts '²'
    - Strips explicit '*' multiplication between coefficients, variables, and parentheses
    """
    if expr is None:
        return ""
    if isinstance(expr, sp.Eq):
        return f"{format_math_display(expr.lhs)} = {format_math_display(expr.rhs)}"

    s = sp.sstr(expr) if not isinstance(expr, str) else expr.strip()

    # Unwrap Eq(...) string representation
    eq_match = re.match(r"^Eq\((.*),\s*(.*)\)$", s)
    if eq_match:
        return f"{format_math_display(eq_match.group(1))} = {format_math_display(eq_match.group(2))}"

    # Convert powers **(...) or **n to superscripts
    s = re.sub(r"\*\*\(([^)]+)\)", lambda m: to_superscript(m.group(1)), s)
    s = re.sub(r"\*\*([0-9a-zA-Z+-]+)", lambda m: to_superscript(m.group(1)), s)
    s = re.sub(r"\^([0-9a-zA-Z+-]+)", lambda m: to_superscript(m.group(1)), s)

    # Clean multiplication signs:
    # 1) Between parentheses: )(
    s = re.sub(r"\)\s*\*\s*\(", ")(", s)
    # 2) Number * letter or Number * (
    s = re.sub(r"(\b\d+)\s*\*\s*([a-zA-Z\(])", r"\1\2", s)
    # 3) Letter * (
    s = re.sub(r"([a-zA-Z])\s*\*\s*\(", r"\1(", s)
    # 4) ) * letter
    s = re.sub(r"\)\s*\*\s*([a-zA-Z])", r")\1", s)
    # 5) variable * function or variable (e.g. x*sin(x) -> x sin(x), x*y -> xy)
    s = re.sub(r"([a-zA-Z0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*\*\s*([a-zA-Z]+(?:\([^)]*\))?)", r"\1 \2", s)
    # 6) Any remaining * between expressions
    s = re.sub(r"\s*\*\s*", " · ", s)

    # Clean formatting quirks
    s = re.sub(r"\+\s*-", "- ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_to_html_math(expr: Any) -> str:
    """Format mathematical expression into HTML with true floating <sup> tags.

    Used in HTML cards so exponents visually float above the baseline consistently.
    """
    clean = format_math_display(expr)

    rev_supers = {v: k for k, v in UNICODE_SUPERSCRIPTS.items()}

    def _sub_html_super(m):
        inner = "".join(rev_supers.get(c, c) for c in m.group(0))
        return f'<sup style="font-size: 0.72em; vertical-align: super; line-height: 0;">{inner}</sup>'

    html_str = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵐᵏ⁽⁾]+", _sub_html_super, clean)
    html_str = re.sub(
        r"\^([0-9a-zA-Z+-]+)",
        r'<sup style="font-size: 0.72em; vertical-align: super; line-height: 0;">\1</sup>',
        html_str,
    )
    return html_str


def to_latex_equation(expr: Any) -> str:
    """Convert an expression, equation, or equation string to clean LaTeX without Eq(...) wrapper."""
    if expr is None:
        return ""
    if isinstance(expr, sp.Eq):
        return rf"{sp.latex(expr.lhs)} = {sp.latex(expr.rhs)}"
    if isinstance(expr, sp.Expr):
        return sp.latex(expr)

    s = str(expr).strip()
    if s.startswith("Eq(") and s.endswith(")"):
        eq_match = re.match(r"^Eq\((.*),\s*(.*)\)$", s)
        if eq_match:
            try:
                lhs = sp.sympify(MathParser.sanitize_math_string(eq_match.group(1)))
                rhs = sp.sympify(MathParser.sanitize_math_string(eq_match.group(2)))
                return rf"{sp.latex(lhs)} = {sp.latex(rhs)}"
            except Exception:
                pass
    if "=" in s:
        parts = s.split("=", 1)
        try:
            lhs = sp.sympify(MathParser.sanitize_math_string(parts[0]))
            rhs = sp.sympify(MathParser.sanitize_math_string(parts[1]))
            return rf"{sp.latex(lhs)} = {sp.latex(rhs)}"
        except Exception:
            pass
    try:
        sym = sp.sympify(MathParser.sanitize_math_string(s))
        return sp.latex(sym)
    except Exception:
        return s
