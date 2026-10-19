"""Mathematical function plotting and graph visualisation.

Generates publication-quality, academic plots using Matplotlib for:
1. Differentiation: original function f(x) and its derivative f'(x) on dual labelled curves.
2. Integration:
   - Indefinite: integrand f(x) and antiderivative F(x).
   - Definite: integrand f(x) with shaded signed area between limits [a, b].
3. General Function Graphing: arbitrary single-variable function y = f(x) with range controls.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import sympy as sp

logger = logging.getLogger(__name__)


def get_matplotlib_figure():
    """Lazily import matplotlib and configure a clean academic style."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    return plt


class MathPlotter:
    """Generates clean, responsive mathematical plots."""

    LINE_COLOR_MAIN = "#1f77b4"       # Deep blue
    LINE_COLOR_DERIV = "#d95f02"      # Orange-red
    LINE_COLOR_INTEG = "#2ca02c"      # Green
    COLOR_FILL = "#a1d99b"            # Soft green for definite integral area
    GRID_COLOR = "#e0e0e0"

    @classmethod
    def _to_symbol(cls, var: Union[sp.Symbol, str]) -> sp.Symbol:
        if isinstance(var, str):
            return sp.Symbol(var)
        return var

    @classmethod
    def _to_expr(cls, expr: Union[sp.Expr, str], var: sp.Symbol) -> sp.Expr:
        if isinstance(expr, str):
            from tutor.core.parser import MathParser
            parsed = MathParser.parse_expression(expr, var)
            if parsed is None:
                parsed_res = MathParser.parse(expr, default_variable=var.name)
                if parsed_res.success and parsed_res.parsed_object is not None:
                    return parsed_res.parsed_object
                raise ValueError(f"Unable to parse expression: {expr}")
            return parsed
        return expr

    @classmethod
    def evaluate_expression_safely(
        cls,
        expr: sp.Expr,
        var: sp.Symbol,
        x_vals: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Numerically evaluate a SymPy expression over numpy array, filtering asymptotes."""
        try:
            # Modules for lambdify
            modules = [
                {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
                 "log": np.log, "sqrt": np.sqrt, "abs": np.abs},
                "numpy"
            ]
            f_num = sp.lambdify(var, expr, modules=modules)
            y_vals = f_num(x_vals)

            # If scalar returned (e.g. constant function like f(x) = 5)
            if not isinstance(y_vals, np.ndarray):
                y_vals = np.full_like(x_vals, float(y_vals), dtype=float)

            # Ensure 1D float array
            y_vals = np.asarray(y_vals, dtype=float)

            # Mask out non-finite or extremely large asymptote values
            mask = np.isfinite(y_vals) & (np.abs(y_vals) < 1e5)
            y_clean = np.where(mask, y_vals, np.nan)
            return y_clean
        except Exception as e:
            logger.debug("Failed numerical evaluation for plot: %s", e)
            return None

    @classmethod
    def plot_differentiation(
        cls,
        func_expr: Union[sp.Expr, str],
        deriv_expr: Optional[Union[sp.Expr, str]] = None,
        var: Union[sp.Symbol, str] = "x",
        x_range: Tuple[float, float] = (-5.0, 5.0),
        title: Optional[str] = None,
    ):
        """Plot both original function f(x) and its derivative f'(x)."""
        var = cls._to_symbol(var)
        func_expr = cls._to_expr(func_expr, var)
        if deriv_expr is None:
            deriv_expr = sp.diff(func_expr, var)
        else:
            deriv_expr = cls._to_expr(deriv_expr, var)

        plt = get_matplotlib_figure()
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)

        x_min, x_max = x_range
        if x_min >= x_max:
            x_min, x_max = -5.0, 5.0
        x_vals = np.linspace(x_min, x_max, 600)

        y_orig = cls.evaluate_expression_safely(func_expr, var, x_vals)
        y_deriv = cls.evaluate_expression_safely(deriv_expr, var, x_vals)

        has_data = False
        var_name = var.name

        if y_orig is not None and not np.all(np.isnan(y_orig)):
            ax.plot(
                x_vals,
                y_orig,
                label=rf"$y = f({var_name}) = {sp.latex(func_expr)}$",
                color=cls.LINE_COLOR_MAIN,
                linewidth=2.2,
            )
            has_data = True

        if y_deriv is not None and not np.all(np.isnan(y_deriv)):
            ax.plot(
                x_vals,
                y_deriv,
                label=rf"$y = f'({var_name}) = {sp.latex(deriv_expr)}$",
                color=cls.LINE_COLOR_DERIV,
                linewidth=2.0,
                linestyle="--",
            )
            has_data = True

        if not has_data:
            plt.close(fig)
            return None

        # Clean academic axes styling
        ax.set_title(title or f"Differentiation: Function and Derivative with respect to ${var_name}$", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel(f"${var_name}$", fontsize=10)
        ax.set_ylabel("$y$", fontsize=10)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.grid(True, color=cls.GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="best")

        # Auto-adjust y limits to avoid crazy asymptotes
        all_valid_y = []
        if y_orig is not None:
            all_valid_y.extend(y_orig[np.isfinite(y_orig)])
        if y_deriv is not None:
            all_valid_y.extend(y_deriv[np.isfinite(y_deriv)])

        if all_valid_y:
            p_low = np.percentile(all_valid_y, 2)
            p_high = np.percentile(all_valid_y, 98)
            pad = max(1.0, (p_high - p_low) * 0.15)
            ax.set_ylim(p_low - pad, p_high + pad)

        fig.tight_layout()
        return fig

    @classmethod
    def plot_integration(
        cls,
        integrand: Union[sp.Expr, str],
        antiderivative: Optional[Union[sp.Expr, str]] = None,
        var: Union[sp.Symbol, str] = "x",
        limits: Optional[Tuple[float, float]] = None,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        title: Optional[str] = None,
    ):
        """Plot integration: antiderivative curves or definite integral shaded area."""
        var = cls._to_symbol(var)
        integrand = cls._to_expr(integrand, var)
        if antiderivative is None and limits is None:
            antiderivative = sp.integrate(integrand, var)
        elif antiderivative is not None:
            antiderivative = cls._to_expr(antiderivative, var)

        plt = get_matplotlib_figure()
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)

        var_name = var.name

        if limits is not None:
            # Definite integral plot
            a, b = float(limits[0]), float(limits[1])
            # Center view around limits
            span = max(abs(b - a), 2.0)
            x_min = min(a, b) - span * 0.4
            x_max = max(a, b) + span * 0.4
            x_vals = np.linspace(x_min, x_max, 600)

            y_integrand = cls.evaluate_expression_safely(integrand, var, x_vals)
            if y_integrand is None or np.all(np.isnan(y_integrand)):
                plt.close(fig)
                return None

            # Plot integrand
            ax.plot(
                x_vals,
                y_integrand,
                label=rf"$y = f({var_name}) = {sp.latex(integrand)}$",
                color=cls.LINE_COLOR_MAIN,
                linewidth=2.2,
            )

            # Shade definite integral region
            x_fill = np.linspace(min(a, b), max(a, b), 300)
            y_fill = cls.evaluate_expression_safely(integrand, var, x_fill)
            if y_fill is not None:
                # Compute exact value for legend
                try:
                    val_exact = sp.integrate(integrand, (var, limits[0], limits[1]))
                    val_str = f" = {val_exact}"
                except Exception:
                    val_str = ""

                ax.fill_between(
                    x_fill,
                    0,
                    y_fill,
                    color=cls.COLOR_FILL,
                    alpha=0.5,
                    label=rf"$\int_{{{a}}}^{{{b}}} f({var_name})\,d{var_name}{val_str}$",
                )

            # Vertical boundary markers
            ax.axvline(a, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"$a = {a}$")
            ax.axvline(b, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"$b = {b}$")

            plot_title = title or rf"Definite Integral: Area Under $f({var_name})$ from ${a}$ to ${b}$"

        else:
            # Indefinite integral plot
            x_min, x_max = x_range
            x_vals = np.linspace(x_min, x_max, 600)

            y_integrand = cls.evaluate_expression_safely(integrand, var, x_vals)
            y_antideriv = cls.evaluate_expression_safely(antiderivative, var, x_vals)

            has_data = False
            if y_integrand is not None and not np.all(np.isnan(y_integrand)):
                ax.plot(
                    x_vals,
                    y_integrand,
                    label=rf"$y = f({var_name}) = {sp.latex(integrand)}$",
                    color=cls.LINE_COLOR_MAIN,
                    linewidth=2.2,
                )
                has_data = True

            if y_antideriv is not None and not np.all(np.isnan(y_antideriv)):
                ax.plot(
                    x_vals,
                    y_antideriv,
                    label=rf"$y = F({var_name}) = \int f({var_name})\,d{var_name}$",
                    color=cls.LINE_COLOR_INTEG,
                    linewidth=2.0,
                    linestyle="--",
                )
                has_data = True

            if not has_data:
                plt.close(fig)
                return None

            plot_title = title or rf"Indefinite Integration: Integrand and Antiderivative"

        ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel(f"${var_name}$", fontsize=10)
        ax.set_ylabel("$y$", fontsize=10)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.grid(True, color=cls.GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="best")
        fig.tight_layout()
        return fig

    @classmethod
    def plot_function(
        cls,
        func_expr: Union[sp.Expr, str],
        var: Union[sp.Symbol, str] = "x",
        x_range: Tuple[float, float] = (-10.0, 10.0),
        title: Optional[str] = None,
    ):
        """Plot a general mathematical function y = f(x)."""
        var = cls._to_symbol(var)
        func_expr = cls._to_expr(func_expr, var)

        plt = get_matplotlib_figure()
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)

        x_min, x_max = x_range
        if x_min >= x_max:
            x_min, x_max = -10.0, 10.0
        x_vals = np.linspace(x_min, x_max, 800)

        y_vals = cls.evaluate_expression_safely(func_expr, var, x_vals)
        if y_vals is None or np.all(np.isnan(y_vals)):
            plt.close(fig)
            return None

        var_name = var.name
        ax.plot(
            x_vals,
            y_vals,
            label=rf"$y = {sp.latex(func_expr)}$",
            color=cls.LINE_COLOR_MAIN,
            linewidth=2.2,
        )

        # Plot real roots within interval if easily computable
        try:
            roots = sp.solve(sp.Eq(func_expr, 0), var)
            real_roots = []
            for r in roots:
                if r.is_real:
                    r_val = float(r)
                    if x_min <= r_val <= x_max:
                        real_roots.append(r_val)
            if real_roots:
                ax.plot(
                    real_roots,
                    [0] * len(real_roots),
                    "o",
                    color="#d95f02",
                    markersize=6,
                    label="Roots $(y = 0)$",
                )
        except Exception:
            pass

        ax.set_title(title or rf"Function Plot: $y = {sp.latex(func_expr)}$", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel(f"${var_name}$", fontsize=10)
        ax.set_ylabel("$y$", fontsize=10)
        ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.grid(True, color=cls.GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=9, loc="best")

        # Trim vertical bounds to avoid asymptote blowup
        valid = y_vals[np.isfinite(y_vals)]
        if len(valid) > 0:
            p_low = np.percentile(valid, 2)
            p_high = np.percentile(valid, 98)
            pad = max(1.0, (p_high - p_low) * 0.15)
            ax.set_ylim(p_low - pad, p_high + pad)

        fig.tight_layout()
        return fig

    # Class aliases for convenience
    plot_derivative = plot_differentiation
    plot_integral = plot_integration
