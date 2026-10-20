"""MathLab AI — Professional Academic Mathematics Workspace.

Provides:
- 🧮 Equation Solving (Linear, quadratic, polynomial, rational)
- 🔍 Algebraic Simplification (Rational reduction, trigonometric)
- 🧩 Polynomial Factorisation (Factorization over rationals)
- 📐 Algebraic Expansion (Distributive polynomial products)
- 📈 Symbolic Differentiation (Analytical dy/dx with dual-curve plots)
- 📉 Symbolic Integration (Indefinite antiderivatives & definite shaded-area plots)
- 📊 Function Graphing (Real analysis, continuous bounds, root detection)
- 🕒 Calculation History (Session audit trail with one-click reload)
- 📷 100% Offline OCR (Local image formula recognition without APIs)
- ✍️ Pedagogical Step Verifier ("Check My Step" with diagnostic hints)
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import sympy as sp
from PIL import Image

# Ensure tutor package is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tutor.core.graphing import MathPlotter
from tutor.core.ocr import LocalMathOCR
from tutor.core.parser import MathParser

# Import math display helpers with safe fallbacks if tutor/core/parser.py on Colab is not yet updated
try:
    from tutor.core.parser import format_math_display, format_to_html_math, to_latex_equation
except ImportError:
    def format_math_display(expr: Any) -> str:
        if expr is None:
            return ""
        s = str(expr).strip()
        if s.startswith("Eq(") and s.endswith(")"):
            s = s[3:-1].replace(",", " =", 1)
        s = re.sub(r"(\b\d+)\s*\*\s*([a-zA-Z\(])", r"\1\2", s)
        s = s.replace("**", "^").replace("*", " · ")
        return s

    def format_to_html_math(expr: Any) -> str:
        s = format_math_display(expr)
        return re.sub(
            r"\^([0-9a-zA-Z+-]+)",
            r'<sup style="font-size: 0.72em; vertical-align: super; line-height: 0;">\1</sup>',
            s,
        )

    def to_latex_equation(expr: Any) -> str:
        if expr is None:
            return ""
        try:
            return sp.latex(sp.sympify(str(expr)))
        except Exception:
            return str(expr)

from tutor.core.solver import SymPySolver, TutorSolution
from tutor.core.tutor import MathTutor
from tutor.core.verifier import StepFeedback

# -----------------------------------------------------------------------------
# Page Configuration & Professional Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MathLab AI — Academic Mathematics Studio",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic, high-contrast, professional mathematical UI
st.markdown(
    """
    <style>
    /* Global academic theme */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1080px;
    }
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    h1, h2, h3, h4 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0f172a;
        font-weight: 700;
        letter-spacing: -0.015em;
    }

    /* Academic Header Card */
    .lab-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .lab-header-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .lab-category {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: #2563eb;
        background: #eff6ff;
        padding: 3px 9px;
        border-radius: 4px;
        border: 1px solid #dbeafe;
    }
    .lab-status {
        font-size: 0.75rem;
        color: #059669;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .lab-status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
    }
    .lab-header h1 {
        margin: 4px 0 6px 0;
        font-size: 1.75rem;
        color: #0f172a;
    }
    .lab-header p {
        margin: 0;
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Input Card Container */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .input-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #334155;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Card */
    .result-container {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #1d4ed8;
        border-radius: 10px;
        padding: 22px 26px;
        margin: 20px 0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        color: #475569;
        margin-bottom: 8px;
    }
    .result-value {
        font-size: 1.35rem;
        color: #0f172a;
        font-family: "JetBrains Mono", Consolas, "Courier New", monospace;
        font-weight: 600;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    /* Empty state */
    .empty-state {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 36px 24px;
        text-align: center;
        color: #64748b;
        margin: 20px 0;
        font-size: 0.95rem;
    }

    /* Step Derivation Card */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .step-badge {
        display: inline-block;
        background: #1e3a8a;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }
    .rule-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }

    /* History Entry */
    .history-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .op-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }

    /* Button enhancements */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Workspace Metadata & Preloaded Functions Definition
# -----------------------------------------------------------------------------
WORKSPACE_CONFIG: Dict[str, Dict[str, Any]] = {
    "🧮 Equation Solving": {
        "title": "Equation Solving",
        "category": "ALGEBRA & POLYNOMIAL ROOTS",
        "icon": "🧮",
        "desc": "Solve linear, quadratic, polynomial, and rational equations symbolically with step-by-step algebraic equivalence.",
        "key": "input_solve",
        "default": "3(2x - 4) + 5 = 2(x + 7) + 9",
        "op": "solve",
        "placeholder": "e.g. 3(2x - 4) + 5 = 2(x + 7) + 9 or x^2 - 5x + 6 = 0",
        "btn_label": "⚡ Solve Equation",
        "examples": [
            "3(2x - 4) + 5 = 2(x + 7) + 9",
            "x^2 - 5x + 6 = 0",
            "2x + 3 = 11",
            "x^2 - 4 = 0",
            "2(x + 3) - 4 = 3(x - 1) + 5",
        ],
    },
    "🔍 Simplification": {
        "title": "Algebraic Simplification",
        "category": "RATIONAL & ALGEBRAIC REDUCTION",
        "icon": "🔍",
        "desc": "Simplify algebraic and rational expressions by canceling common factors and collecting like terms.",
        "key": "input_simp",
        "default": "(x^2 - 1)/(x - 1)",
        "op": "simplify",
        "placeholder": "e.g. (x^2 - 1)/(x - 1) or (2x + 4)/2",
        "btn_label": "🔍 Simplify Expression",
        "examples": [
            "(x^2 - 1)/(x - 1)",
            "(x^2 - 4)/(x - 2)",
            "(2x + 4)/2",
            "(x^3 - x)/(x - 1)",
            "(x^2 + 2x + 1)/(x + 1)",
        ],
    },
    "🧩 Factorisation": {
        "title": "Polynomial Factorisation",
        "category": "POLYNOMIAL DECOMPOSITION",
        "icon": "🧩",
        "desc": "Decompose polynomials into irreducible linear and quadratic factors over rational numbers.",
        "key": "input_factor",
        "default": "x^2 + 5x + 6",
        "op": "factor",
        "placeholder": "e.g. x^2 + 5x + 6 or x^3 - 3x^2 + 3x - 1",
        "btn_label": "🧩 Factor Polynomial",
        "examples": [
            "x^2 + 5x + 6",
            "x^2 - 4",
            "2x^2 + 7x + 3",
            "x^3 - 3x^2 + 3x - 1",
            "x^4 - 16",
        ],
    },
    "📐 Expansion": {
        "title": "Algebraic Expansion",
        "category": "POLYNOMIAL MULTIPLICATION",
        "icon": "📐",
        "desc": "Expand products of binomials, polynomials, and powers into standard polynomial canonical form.",
        "key": "input_expand",
        "default": "(x + 2)(x + 3)",
        "op": "expand",
        "placeholder": "e.g. (x + 2)(x + 3) or (2x - 1)(3x + 4)",
        "btn_label": "📐 Expand Expression",
        "examples": [
            "(x + 2)(x + 3)",
            "(2x - 1)(3x + 4)",
            "(x + 1)^3",
            "(x - 2)(x + 2)",
            "(2x + 3)^2",
        ],
    },
    "📈 Differentiation": {
        "title": "Symbolic Differentiation",
        "category": "DIFFERENTIAL CALCULUS",
        "icon": "📈",
        "desc": "Compute exact analytical derivatives dy/dx using calculus product, quotient, and chain rules with dual-curve plots.",
        "key": "input_diff",
        "default": "x^3*sin(x)",
        "op": "diff",
        "placeholder": "e.g. x^3*sin(x) or Differentiate y = x^2 + 3x with respect to x",
        "btn_label": "📈 Differentiate & Plot",
        "examples": [
            "x^3*sin(x)",
            "Differentiate y = x^3 sin(x) with respect to x",
            "Find dy/dx for y = x^2 + 3x",
            "sin(x)*exp(x)",
            "x^2 + 3x - 5",
            "x*ln(x)",
        ],
    },
    "📉 Integration": {
        "title": "Symbolic Integration",
        "category": "INTEGRAL CALCULUS",
        "icon": "📉",
        "desc": "Compute indefinite integrals (antiderivatives) or definite integrals with shaded signed-area plots.",
        "key": "input_integ",
        "default": "x^2",
        "op": "integrate",
        "placeholder": "e.g. x^2 or Integrate x^2 from 0 to 2",
        "btn_label": "📉 Integrate & Plot",
        "examples": [
            "x^2",
            "3x^2 + 4x - 2",
            "sin(x)",
            "x^2 from 0 to 2",
            "exp(x)*sin(x)",
            "1/(1 + x^2)",
        ],
    },
    "📊 Function Graphing": {
        "title": "Function Graphing",
        "category": "REAL ANALYSIS & PLOTTING",
        "icon": "📊",
        "desc": "Visualise continuous single-variable functions y = f(x), observe asymptotic bounds, and identify real roots.",
        "key": "input_graph",
        "default": "x^2 - 4",
        "op": "graph",
        "placeholder": "e.g. x^2 - 4 or sin(x) + cos(2*x)",
        "btn_label": "📊 Generate Graph",
        "examples": [
            "x^2 - 4",
            "sin(x)",
            "x^3 - 3x",
            "1/(x^2 + 1)",
            "x*exp(-x^2)",
            "tan(x)",
        ],
    },
    "🕒 Calculation History": {
        "title": "Calculation History",
        "category": "AUDIT LOG & PERSISTENCE",
        "icon": "🕒",
        "desc": "Comprehensive session audit trail. Review past symbolic solutions and reload any previous expression into the editor.",
        "key": "history",
        "default": "",
        "op": "history",
        "placeholder": "",
        "btn_label": "",
        "examples": [],
    },
}

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "last_solution" not in st.session_state:
    st.session_state["last_solution"] = None

if "student_steps" not in st.session_state:
    st.session_state["student_steps"] = []

# Cached singleton tutor instances
@st.cache_resource
def get_tutor() -> MathTutor:
    """Singleton instance of rule-based SymPy tutor."""
    return MathTutor(default_variable="x", enable_llm=False)


@st.cache_resource(show_spinner=False)
def get_ai_tutor() -> Optional[MathTutor]:
    """Singleton instance of AI-enhanced tutor with fine-tuned LoRA adapter.

    Raises RuntimeError on failure so that only a successfully loaded model is cached.
    """
    adapter_dir = ROOT_DIR / "models" / "qwen2.5-math-tutor-lora"
    if not adapter_dir.exists():
        adapter_dir = Path("models/qwen2.5-math-tutor-lora")

    tutor_ai = MathTutor(
        default_variable="x",
        enable_llm=True,
        adapter_path=str(adapter_dir),
        device="cuda",
    )
    explainer = tutor_ai._get_explainer()
    if not explainer._ensure_loaded():
        reason = explainer._failure_reason or "Failed to load neural model weights."
        raise RuntimeError(reason)
    return tutor_ai


def resolve_tutor(use_ai: bool) -> Tuple[MathTutor, bool]:
    """Resolve active tutor instance and verified AI mode.

    If AI mode is requested but fails to load (raises or returns None),
    falls back to rule-based SymPy tutor and returns (tutor, False).
    """
    if use_ai:
        try:
            tutor = get_ai_tutor()
            if tutor is None or not getattr(tutor, "enable_llm", False):
                raise RuntimeError("Failed to load neural model weights.")
            return tutor, True
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not initialize AI model ({e}), using the SymPy tutor instead.")
            return get_tutor(), False
    return get_tutor(), False


def check_gpu_and_dependencies() -> Tuple[bool, str]:
    """Verify GPU availability, PyTorch, and adapter artifacts for AI mode."""
    adapter_path = ROOT_DIR / "models" / "qwen2.5-math-tutor-lora"
    if not adapter_path.exists():
        adapter_path = Path("models/qwen2.5-math-tutor-lora")

    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        return False, "Adapter folder 'models/qwen2.5-math-tutor-lora' not found. AI mode needs the fine-tuned adapter, using the SymPy tutor instead."
    try:
        import torch
        import transformers
    except ImportError as e:
        return False, f"Neural libraries not installed ({e}). AI mode needs requirements-train.txt, using the SymPy tutor instead."
    if not torch.cuda.is_available():
        return False, "AI mode needs a GPU, using the SymPy tutor instead."
    return True, ""


def _on_ai_toggle() -> None:
    """Validate GPU availability and dependencies when AI mode is toggled."""
    if st.session_state.get("ai_explanation_toggle", False):
        can_run, reason = check_gpu_and_dependencies()
        if not can_run:
            st.session_state["ai_explanation_toggle"] = False
            st.session_state["ai_fallback_reason"] = reason or "AI mode needs a GPU, using the SymPy tutor instead."


def render_result_badge(is_ai_mode: bool):
    """Render a small label under each answer showing which mode produced it."""
    engine_name = "SymPy + fine-tuned model" if is_ai_mode else "SymPy engine"
    icon = "🤖" if is_ai_mode else "⚙️"
    badge_bg = "#ecfdf5" if is_ai_mode else "#f8fafc"
    badge_color = "#047857" if is_ai_mode else "#475569"
    badge_border = "#a7f3d0" if is_ai_mode else "#cbd5e1"
    st.markdown(
        f"""
        <div style="margin-top: 4px; margin-bottom: 14px; font-size: 0.78rem; color: #64748b; display: flex; align-items: center; gap: 6px;">
            <span>{icon} Engine:</span>
            <span style="background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                {engine_name}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_to_history(operation: str, query: str, result: str, latex: str, engine_mode: str = "SymPy engine"):
    """Record calculation to session history."""
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "operation": operation,
        "input": query,
        "result": result,
        "latex": latex,
        "engine": engine_mode,
    }
    st.session_state["history"].insert(0, entry)


def load_preloaded_expression(expr: str, target_key: str, auto_run: bool = True, target_ws: Optional[str] = None):
    """Populate the expression directly into the target input box and trigger execution.
    
    Designed to work seamlessly as a Streamlit on_click callback so that
    st.session_state[target_key] is updated before the target widget is instantiated.
    """
    clean_expr = format_math_display(expr) if expr else ""
    st.session_state[target_key] = clean_expr
    st.session_state["last_solution"] = None
    st.session_state["student_steps"] = []
    if auto_run:
        st.session_state["auto_solve"] = True
    if target_ws and "selected_workspace_radio" in st.session_state:
        st.session_state["selected_workspace_radio"] = target_ws


def render_practice_arena(tutor: MathTutor, problem: str, operation: str, variable: str, target_key: str):
    """Interactive Quiz & Practice Problem Generator for reinforcing learned math concepts."""
    with st.expander("🎯 Interactive Quiz & Practice Arena (3 Similar Problems)", expanded=True):
        st.markdown(
            f"""
            <div style="font-size: 0.9rem; color: #475569; margin-bottom: 10px;">
                Reinforce what you just learned! Test your skills on 3 generated problems similar to <code>{format_math_display(problem)}</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_diff, col_gen = st.columns([3, 2])
        with col_diff:
            diff = st.radio(
                "Select Difficulty Tier:",
                options=["Easy", "Medium", "Hard"],
                index=1,
                horizontal=True,
                key=f"quiz_diff_{target_key}",
            )
        with col_gen:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            gen_btn = st.button("🎲 Generate New Problems", key=f"quiz_gen_{target_key}", use_container_width=True)

        storage_key = f"quiz_problems_{target_key}"
        curr_diff_key = f"quiz_diff_val_{target_key}"
        diff_changed = st.session_state.get(curr_diff_key) != diff

        # Generate on explicit button click, initial load, or if difficulty changed (do not wipe on problem solve)
        if gen_btn or (storage_key not in st.session_state) or diff_changed:
            st.session_state[curr_diff_key] = diff
            try:
                if hasattr(tutor, "generate_practice"):
                    st.session_state[storage_key] = tutor.generate_practice(
                        problem=problem,
                        operation=operation,
                        difficulty=diff,
                        count=3,
                        variable=variable,
                    )
                else:
                    from tutor.core.practice import PracticeGenerator
                    generator = PracticeGenerator(solver=getattr(tutor, "solver", None))
                    st.session_state[storage_key] = generator.generate(
                        problem=problem,
                        operation=operation,
                        difficulty=diff,
                        count=3,
                        variable=variable,
                    )
            except Exception as e:
                st.warning(f"Could not automatically generate practice problems: {e}")
                # Provide lightweight fallback problems so the quiz arena is never blank
                from tutor.core.practice import PracticeProblem
                fallback_probs = [
                    PracticeProblem(id=1, problem=f"2*{variable} + 5 = 15", operation="solve", difficulty="Easy", variable=variable, expected_answer=f"{variable} = 5", expected_answer_latex=f"{variable} = 5", hint="Subtract 5 from both sides, then divide by 2.", problem_type="linear_eq"),
                    PracticeProblem(id=2, problem=f"3*{variable} - 4 = 11", operation="solve", difficulty="Medium", variable=variable, expected_answer=f"{variable} = 5", expected_answer_latex=f"{variable} = 5", hint="Add 4 to both sides, then divide by 3.", problem_type="linear_eq"),
                    PracticeProblem(id=3, problem=f"5*{variable} + 2 = 3*{variable} + 12", operation="solve", difficulty="Hard", variable=variable, expected_answer=f"{variable} = 5", expected_answer_latex=f"{variable} = 5", hint="Subtract 3x from both sides, then subtract 2.", problem_type="linear_eq"),
                ] if operation == "solve" else [
                    PracticeProblem(id=1, problem=f"({variable} + 2)*({variable} + 3)", operation="expand", difficulty="Easy", variable=variable, expected_answer=f"{variable}**2 + 5*{variable} + 6", expected_answer_latex=f"{variable}^2 + 5{variable} + 6", hint="Use the FOIL method.", problem_type="expansion"),
                    PracticeProblem(id=2, problem=f"2*({variable} + 4)", operation="expand", difficulty="Easy", variable=variable, expected_answer=f"2*{variable} + 8", expected_answer_latex=f"2{variable} + 8", hint="Distribute the 2.", problem_type="expansion"),
                    PracticeProblem(id=3, problem=f"({variable} - 1)*({variable} + 5)", operation="expand", difficulty="Medium", variable=variable, expected_answer=f"{variable}**2 + 4*{variable} - 5", expected_answer_latex=f"{variable}^2 + 4{variable} - 5", hint="Multiply terms pairwise.", problem_type="expansion"),
                ]
                st.session_state[storage_key] = fallback_probs

        practice_items = st.session_state.get(storage_key, [])
        if practice_items:
            for idx, p in enumerate(practice_items, start=1):
                clean_prob_text = format_math_display(p.problem)
                clean_prob_html = format_to_html_math(p.problem)
                sol_state_key = f"show_sol_{target_key}_{idx}_{clean_prob_text}"
                is_showing_sol = st.session_state.get(sol_state_key, False)

                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-top: 10px; margin-bottom: 8px; border-left: 4px solid #3b82f6;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #1d4ed8; background: #eff6ff; padding: 2px 7px; border-radius: 4px;">Problem {idx} &bull; {p.difficulty}</span>
                        <div style="font-size: 1.15rem; font-family: monospace; font-weight: 600; color: #0f172a; margin-top: 6px;">
                            {clean_prob_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_ans, col_chk, col_load = st.columns([3, 1.3, 1.3])
                with col_ans:
                    student_ans = st.text_input(
                        f"Your answer for #{idx}:",
                        placeholder="e.g. x = 2, x = 3",
                        key=f"q_ans_{target_key}_{idx}_{clean_prob_text}",
                        label_visibility="collapsed",
                    )
                with col_chk:
                    check_clicked = st.button(
                        "Check Answer",
                        key=f"btn_chk_{target_key}_{idx}_{clean_prob_text}",
                        use_container_width=True,
                    )
                with col_load:
                    if st.button(
                        "🙈 Hide Steps" if is_showing_sol else "📖 Show Steps",
                        key=f"btn_toggle_sol_{target_key}_{idx}_{clean_prob_text}",
                        use_container_width=True,
                        help="View step-by-step derivation for this problem without losing current quiz",
                    ):
                        st.session_state[sol_state_key] = not is_showing_sol
                        st.rerun()

                if check_clicked:
                    if student_ans.strip():
                        if hasattr(tutor, "check_practice_answer"):
                            is_corr, msg = tutor.check_practice_answer(student_ans, p.expected_answer, variable=variable)
                        else:
                            from tutor.core.practice import PracticeGenerator
                            generator = PracticeGenerator(solver=getattr(tutor, "solver", None))
                            is_corr, msg = generator.check_answer(student_ans, p.expected_answer, variable=variable)
                        if is_corr:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.warning("Please type your answer above before checking.")

                # If student clicked "Show Steps" for this quiz problem:
                if st.session_state.get(sol_state_key, False):
                    with st.container():
                        with st.spinner("Calculating step-by-step derivation..."):
                            step_sol = tutor.solve(p.problem, operation=getattr(p, "operation", operation), variable=variable)

                        if step_sol and step_sol.success:
                            st.markdown(
                                f"""
                                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px; margin-top: 10px; margin-bottom: 8px; border-left: 4px solid #16a34a;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span style="font-size: 0.75rem; font-weight: 700; color: #166534; background: #dcfce7; padding: 2px 7px; border-radius: 4px;">📖 Step-by-Step Solution for Problem {idx}</span>
                                        <span style="font-size: 0.88rem; color: #15803d; font-weight: 600;">Answer: {step_sol.final_answer}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if step_sol.final_answer_latex:
                                st.latex(step_sol.final_answer_latex)

                            for s in step_sol.steps:
                                st.markdown(
                                    f"""
                                    <div class="step-card" style="margin-bottom: 6px; padding: 8px 12px; background: #ffffff;">
                                        <span class="step-badge">Step {s.step_number}</span>
                                        <strong>{s.title}</strong>
                                        <p style="margin: 4px 0; color: #334155; font-size: 0.92rem;">{s.description}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if s.latex:
                                    st.latex(s.latex)
                                if s.rule_applied:
                                    st.caption(f"Rule applied: {s.rule_applied}")
                        else:
                            st.info(f"Expected Answer: {p.expected_answer}")

                with st.expander(f"💡 Need a hint for Problem {idx}?", expanded=False):
                    st.info(format_math_display(p.hint))


# -----------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📐 MathLab Studio")
    st.caption("Academic Computer Algebra & Symbolic Tutor")
    st.markdown("---")

    selected_workspace = st.radio(
        "Mathematical Operations",
        list(WORKSPACE_CONFIG.keys()),
        index=0,
        key="selected_workspace_radio",
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Workspace Configuration")
    default_var = st.text_input(
        "Independent Variable",
        value="x",
        max_chars=4,
        help="Target algebraic or calculus variable symbol (default 'x').",
    )

    st.markdown("---")
    st.markdown("#### 🤖 AI Enhancement")

    st.sidebar.toggle(
        "Use AI explanation (fine-tuned Qwen)",
        value=False,
        key="ai_explanation_toggle",
        on_change=_on_ai_toggle,
        help="Augment SymPy step derivations with fine-tuned Qwen2.5-Math neural explanations. Requires GPU and adapter weights.",
    )

    if st.session_state.pop("ai_fallback_reason", None):
        st.sidebar.warning("⚠️ AI mode needs a GPU, using the SymPy tutor instead.")

    requested_ai = st.session_state.get("ai_explanation_toggle", False)
    if requested_ai:
        with st.spinner("Loading fine-tuned Qwen model onto GPU..."):
            tutor, use_ai = resolve_tutor(requested_ai)
        if use_ai:
            st.sidebar.success("🟢 AI Explainer Online (GPU)")
    else:
        tutor, use_ai = resolve_tutor(False)

    calc_count = len(st.session_state["history"])
    st.markdown(f"**Session Computations:** `{calc_count}`")

    if calc_count > 0:
        if st.button("🗑️ Clear Calculation History", use_container_width=True):
            st.session_state["history"] = []
            st.session_state["last_solution"] = None
            st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ System Status", expanded=False):
        engine_display = "SymPy + fine-tuned model" if use_ai else "SymPy Ground-Truth"
        st.markdown(
            f"""
            - **Engine**: {engine_display}
            - **OCR**: Local CPU ONNX / WinOCR
            - **Mode**: 100% Offline (Zero Cloud APIs)
            - **Precision**: Exact Symbolic & Rational
            """
        )

# -----------------------------------------------------------------------------
# Header Display
# -----------------------------------------------------------------------------
cfg = WORKSPACE_CONFIG[selected_workspace]
ws_key = cfg["key"]

st.markdown(
    f"""
    <div class="lab-header">
        <div class="lab-header-top">
            <span class="lab-category">{cfg["category"]}</span>
            <span class="lab-status"><span class="lab-status-dot"></span> Symbolic Engine Online</span>
        </div>
        <h1>{cfg["icon"]} {cfg["title"]}</h1>
        <p>{cfg["desc"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Reusable Preloaded Function Tray Component
# -----------------------------------------------------------------------------
def render_preloaded_tray(examples: List[str], target_key: str):
    """Render interactive preloaded function chips that populate the input box."""
    if not examples:
        return

    st.markdown(
        """
        <div style="margin-top: 2px; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
                📌 Preloaded Functions — Click to Display in Box & Perform:
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(examples), 5))
    for i, ex in enumerate(examples):
        col_idx = i % len(cols)
        clean_ex = format_math_display(ex)
        with cols[col_idx]:
            st.button(
                clean_ex,
                key=f"btn_preload_{target_key}_{i}",
                use_container_width=True,
                on_click=load_preloaded_expression,
                args=(clean_ex, target_key, True),
            )


# -----------------------------------------------------------------------------
# Reusable Offline OCR Scan Drawer Component
# -----------------------------------------------------------------------------
def render_offline_ocr_drawer(target_key: str):
    """Provide a 100% offline OCR upload drawer without cloud APIs."""
    with st.expander("📷 Scan Equation from Image (100% Offline OCR — No API Key Required)", expanded=False):
        st.caption(
            "Upload a screenshot or photo of a math problem. "
            "Text recognition runs entirely locally on your machine CPU using ONNX/WinOCR."
        )
        up_img = st.file_uploader(
            "Select mathematical image",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"ocr_uploader_{target_key}",
        )
        if up_img is not None:
            try:
                pil_image = Image.open(up_img)
                col_o1, col_o2 = st.columns([1, 1.5])
                with col_o1:
                    st.image(pil_image, caption="Uploaded Input", use_container_width=True)
                with col_o2:
                    with st.spinner("Extracting mathematical notation locally..."):
                        ocr_text, ocr_engine = LocalMathOCR.recognize(pil_image)
                    if ocr_text:
                        st.success(f"**Recognized Text:** `{ocr_text}`")
                        st.caption(f"Engine: {ocr_engine}")
                        st.button(
                            "📥 Insert into Input Box & Perform",
                            key=f"btn_ocr_insert_{target_key}",
                            type="primary",
                            on_click=load_preloaded_expression,
                            args=(ocr_text, target_key, True),
                        )
                    else:
                        st.warning("No clear mathematical text detected. Try uploading a higher-contrast crop.")
            except Exception as e:
                st.error(f"Local OCR processing error: {e}")


# Initialize widget key in session state if not already set
if ws_key != "history" and ws_key not in st.session_state:
    st.session_state[ws_key] = cfg["default"]


# -----------------------------------------------------------------------------
# Workspace 1: Equation Solving
# -----------------------------------------------------------------------------
if selected_workspace == "🧮 Equation Solving":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Algebraic Equation:",
        key=ws_key,
        placeholder=cfg["placeholder"],
        help="Input an equation with '='. Implicit multiplication (e.g. 3(2x-4)) and brackets are supported.",
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        exec_solve = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_solve or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Solving equation symbolically..."):
            sol = tutor.solve(user_query, operation="solve", variable=default_var)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Equation Solving", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "solve":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Exact Computed Solution</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Step-by-Step Derivation & Rules", expanded=True):
                for s in sol.steps:
                    st.markdown(
                        f"""
                        <div class="step-card">
                            <span class="step-badge">Step {s.step_number}</span>
                            <strong>{s.title}</strong>
                            <p style="margin: 6px 0 8px 0; color: #334155;">{s.description}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if s.latex:
                        st.latex(s.latex)
                    if s.rule_applied:
                        st.caption(f"Rule: {s.rule_applied}")

            with st.expander("✍️ Check My Step (Interactive Step Verifier)", expanded=False):
                st.caption("Verify intermediate algebraic work without immediately revealing subsequent steps.")
                student_step = st.text_input("Your proposed step:", placeholder="e.g. 6x - 7 = 2x + 23 or 4x = 30", key="step_solve")
                if st.button("Verify Step", key="btn_check_solve"):
                    if student_step.strip():
                        feedback = tutor.check_step(user_query, st.session_state["student_steps"], student_step, variable=default_var)
                        if feedback.is_valid:
                            st.success(f"✅ {feedback.message}")
                            st.session_state["student_steps"].append(student_step.strip())
                        else:
                            st.error(f"❌ {feedback.message}")
                            if feedback.hint:
                                st.info(f"💡 Hint: {feedback.hint}")

        else:
            st.error(f"Could not solve: {sol.error_message}")
            st.info("Tip: Ensure the equation includes an '=' sign and valid single-variable terms.")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded function above or enter an equation to compute exact symbolic roots.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="solve", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 2: Simplification
# -----------------------------------------------------------------------------
elif selected_workspace == "🔍 Simplification":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Algebraic Expression:",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        exec_simp = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_simp or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Simplifying symbolically..."):
            sol = tutor.solve(user_query, operation="simplify", variable=default_var)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Simplification", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "simplify":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Simplified Canonical Form</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Step-by-Step Derivation", expanded=True):
                for s in sol.steps:
                    st.markdown(f"**Step {s.step_number}: {s.title}**")
                    st.markdown(s.description)
                    if s.latex:
                        st.latex(s.latex)
                    st.markdown("---")

        else:
            st.error(f"Could not simplify: {sol.error_message}")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded expression or enter a formula to reduce and simplify terms.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="simplify", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 3: Factorisation
# -----------------------------------------------------------------------------
elif selected_workspace == "🧩 Factorisation":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Polynomial to Factor:",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        exec_factor = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_factor or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Factoring polynomial..."):
            sol = tutor.solve(user_query, operation="factor", variable=default_var)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Factorisation", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "factor":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Factored Polynomial Form</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Factorisation Steps", expanded=True):
                for s in sol.steps:
                    st.markdown(f"**Step {s.step_number}: {s.title}**")
                    st.markdown(s.description)
                    if s.latex:
                        st.latex(s.latex)

        else:
            st.error(f"Could not factor: {sol.error_message}")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded polynomial or enter terms to factor into linear components.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="factor", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 4: Expansion
# -----------------------------------------------------------------------------
elif selected_workspace == "📐 Expansion":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Factored Expression to Expand:",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        exec_expand = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_expand or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Expanding polynomial products..."):
            sol = tutor.solve(user_query, operation="expand", variable=default_var)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Expansion", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "expand":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Expanded Polynomial Form</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Derivation Steps", expanded=True):
                for s in sol.steps:
                    st.markdown(f"**Step {s.step_number}: {s.title}**")
                    st.markdown(s.description)
                    if s.latex:
                        st.latex(s.latex)

        else:
            st.error(f"Could not expand: {sol.error_message}")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded product or enter binomial terms to expand.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="expand", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 5: Differentiation (with Dual Function & Derivative Graphs)
# -----------------------------------------------------------------------------
elif selected_workspace == "📈 Differentiation":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Function or Differentiation Query:",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    col_diff_btn, col_rng1, col_rng2 = st.columns([2, 1.5, 1.5])
    with col_rng1:
        x_min_diff = st.number_input("Interval Min (x_min)", value=-5.0, step=1.0, key="diff_xmin")
    with col_rng2:
        x_max_diff = st.number_input("Interval Max (x_max)", value=5.0, step=1.0, key="diff_xmax")
    with col_diff_btn:
        st.write("")
        st.write("")
        exec_diff = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_diff or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Computing analytical derivative..."):
            sol = tutor.solve(user_query, operation="diff", variable=default_var)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Differentiation", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "diff":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Computed Derivative (dy/dx)</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Calculus Step Breakdown & Rules", expanded=True):
                for s in sol.steps:
                    st.markdown(f"**Step {s.step_number}: {s.title}**")
                    st.markdown(s.description)
                    if s.latex:
                        st.latex(s.latex)
                    if s.rule_applied:
                        st.caption(f"Rule: {s.rule_applied}")
                    st.markdown("---")

            if sol.original_expr is not None:
                st.markdown("#### 📊 Dual Graph Visualisation: $f(x)$ vs $f'(x)$")
                try:
                    var_sym = sp.Symbol(sol.variable)
                    fig = MathPlotter.plot_differentiation(
                        func_expr=sol.original_expr,
                        deriv_expr=sol.raw_result,
                        var=var_sym,
                        x_range=(float(x_min_diff), float(x_max_diff)),
                    )
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Graph could not be plotted for this function on the selected interval.")
                except Exception as ex:
                    st.info(f"Graph rendering skipped: {ex}")

        else:
            st.error(f"Differentiation failed: {sol.error_message}")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded function above or type an expression to differentiate and render dual curves.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="diff", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 6: Integration (with Indefinite & Definite Area Graphs)
# -----------------------------------------------------------------------------
elif selected_workspace == "📉 Integration":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Integrand Expression:",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    integ_type = st.radio(
        "Integration Mode",
        ["Indefinite Integral (Antiderivative + C)", "Definite Integral (with limits [a, b])"],
        horizontal=True,
    )

    col_btn, col_a, col_b = st.columns([2, 1.5, 1.5])
    limits_val = None
    if integ_type == "Definite Integral (with limits [a, b])":
        with col_a:
            lim_a = st.number_input("Lower limit (a)", value=0.0, step=0.5, key="int_a")
        with col_b:
            lim_b = st.number_input("Upper limit (b)", value=2.0, step=0.5, key="int_b")
        limits_val = (lim_a, lim_b)

    with col_btn:
        st.write("")
        st.write("")
        exec_integ = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_integ or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        with st.spinner("Computing analytical integral..."):
            sol = tutor.solve(user_query, operation="integrate", variable=default_var, limits=limits_val)
            st.session_state["last_solution"] = sol
            if sol.success:
                engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
                add_to_history("Integration", user_query, sol.final_answer, sol.final_answer_latex, engine_mode=engine_str)

    sol = st.session_state.get("last_solution")
    if sol and sol.operation == "integrate":
        if sol.success:
            st.markdown(
                f"""
                <div class="result-container">
                    <div class="result-label">Computed Integral</div>
                    <div class="result-value">{sol.final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(sol.final_answer_latex)
            render_result_badge(use_ai)

            with st.expander("📖 Step-by-Step Integration & Rules", expanded=True):
                for s in sol.steps:
                    st.markdown(f"**Step {s.step_number}: {s.title}**")
                    st.markdown(s.description)
                    if s.latex:
                        st.latex(s.latex)
                    if s.rule_applied:
                        st.caption(f"Rule: {s.rule_applied}")
                    st.markdown("---")

            if sol.original_expr is not None:
                st.markdown("#### 📊 Integral Visualisation (Area Under Curve)")
                try:
                    var_sym = sp.Symbol(sol.variable)
                    antideriv_expr = sp.integrate(sol.original_expr, var_sym)
                    fig = MathPlotter.plot_integration(
                        integrand=sol.original_expr,
                        antiderivative=antideriv_expr,
                        var=var_sym,
                        limits=sol.limits,
                    )
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Graph could not be plotted for this function on the selected interval.")
                except Exception as ex:
                    st.info(f"Graph rendering skipped: {ex}")

        else:
            st.error(f"Integration failed: {sol.error_message}")
    elif not sol:
        st.markdown('<div class="empty-state">Select a preloaded integrand or enter a function to compute antiderivatives or evaluate signed area.</div>', unsafe_allow_html=True)

    render_practice_arena(tutor, user_query or cfg["default"], operation="integrate", variable=default_var, target_key=ws_key)


# -----------------------------------------------------------------------------
# Workspace 7: Function Graphing
# -----------------------------------------------------------------------------
elif selected_workspace == "📊 Function Graphing":
    render_preloaded_tray(cfg["examples"], ws_key)
    render_offline_ocr_drawer(ws_key)

    user_query = st.text_input(
        "Enter Function y = f(x):",
        key=ws_key,
        placeholder=cfg["placeholder"],
    )

    col_btn, col_r1, col_r2 = st.columns([2, 1.5, 1.5])
    with col_r1:
        gx_min = st.number_input("Interval Min (x_min)", value=-6.0, step=1.0, key="gx_min")
    with col_r2:
        gx_max = st.number_input("Interval Max (x_max)", value=6.0, step=1.0, key="gx_max")
    with col_btn:
        st.write("")
        st.write("")
        exec_plot = st.button(cfg["btn_label"], type="primary", use_container_width=True)

    should_execute = exec_plot or st.session_state.pop("auto_solve", False)
    if should_execute and user_query.strip():
        parsed = MathParser.parse(user_query, default_variable=default_var)
        if parsed.success and isinstance(parsed.parsed_object, sp.Expr):
            var_sym = sp.Symbol(parsed.variable)
            fig = MathPlotter.plot_function(
                func_expr=parsed.parsed_object,
                var=var_sym,
                x_range=(float(gx_min), float(gx_max)),
            )
            if fig:
                st.pyplot(fig)
                add_to_history("Function Graphing", user_query, f"Plot [{gx_min}, {gx_max}]", sp.latex(parsed.parsed_object))
            else:
                st.warning("Could not generate a numerical plot for this expression on the specified range.")
        elif not parsed.success:
            st.error(f"Invalid function expression: {parsed.error_message}")
    elif not user_query.strip():
        st.markdown('<div class="empty-state">Select a preloaded function or enter an expression to render its 2D plot and detect real roots.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Workspace 8: Calculation History
# -----------------------------------------------------------------------------
elif selected_workspace == "🕒 Calculation History":
    history_records = st.session_state["history"]

    if not history_records:
        st.markdown('<div class="empty-state">No calculations recorded yet in this session. Computations performed across any workspace will appear here.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"**Total calculations recorded in active session:** `{len(history_records)}`")

        for idx, item in enumerate(history_records):
            with st.container():
                st.markdown(
                    f"""
                    <div class="history-card">
                        <span class="op-badge">{item['operation']}</span>
                        <span class="rule-tag">{item.get('engine', 'SymPy engine')}</span>
                        <span style="color: #64748b; font-size: 0.8rem; font-weight: 500; margin-left: 6px;">{item['timestamp']}</span>
                        <div style="margin-top: 10px; font-weight: 600; color: #1e293b;">Input: <code>{item['input']}</code></div>
                        <div style="margin-top: 6px; color: #1d4ed8; font-family: monospace; font-size: 1.05rem;">Result: <strong>{item['result']}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_h1, _ = st.columns([2, 5])
                with col_h1:
                    # Map operation to corresponding workspace
                    op_to_ws = {
                        "Equation Solving": "🧮 Equation Solving",
                        "Simplification": "🔍 Simplification",
                        "Factorisation": "🧩 Factorisation",
                        "Expansion": "📐 Expansion",
                        "Differentiation": "📈 Differentiation",
                        "Integration": "📉 Integration",
                        "Function Graphing": "📊 Function Graphing",
                    }
                    target_ws = op_to_ws.get(item["operation"])
                    if target_ws and target_ws in WORKSPACE_CONFIG:
                        target_key = WORKSPACE_CONFIG[target_ws]["key"]
                        st.button(
                            f"📥 Reload into {item['operation']}",
                            key=f"hist_reload_{idx}",
                            on_click=load_preloaded_expression,
                            args=(item["input"], target_key, True, target_ws),
                        )
