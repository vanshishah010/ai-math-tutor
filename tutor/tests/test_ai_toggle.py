"""Unit tests for AI explanation mode toggle behavior in app.py."""

from unittest.mock import patch
import streamlit as st
from app import _on_ai_toggle, resolve_tutor


def test_on_ai_toggle_resets_when_no_gpu():
    """Assert that _on_ai_toggle resets toggle to False and sets fallback reason when no GPU/deps."""
    st.session_state["ai_explanation_toggle"] = True
    st.session_state.pop("ai_fallback_reason", None)
    with patch("app.check_gpu_and_dependencies", return_value=(False, "no gpu")):
        _on_ai_toggle()
    assert st.session_state["ai_explanation_toggle"] is False
    assert st.session_state.get("ai_fallback_reason") == "no gpu"


def test_on_ai_toggle_stays_on_when_gpu_available():
    """Assert that _on_ai_toggle leaves toggle as True when GPU and dependencies are available."""
    st.session_state["ai_explanation_toggle"] = True
    st.session_state.pop("ai_fallback_reason", None)
    with patch("app.check_gpu_and_dependencies", return_value=(True, "")):
        _on_ai_toggle()
    assert st.session_state["ai_explanation_toggle"] is True
    assert "ai_fallback_reason" not in st.session_state


def test_resolve_tutor_fallback_on_model_load_failure():
    """Assert that when get_ai_tutor fails, resolve_tutor falls back to SymPy tutor and engine badge/history say SymPy engine."""
    with patch("app.get_ai_tutor", side_effect=RuntimeError("GPU OOM / weights not found")):
        with patch("streamlit.sidebar.warning") as mock_warning:
            tutor, use_ai = resolve_tutor(use_ai=True)
            assert use_ai is False
            assert tutor.enable_llm is False
            mock_warning.assert_called_once()
            assert "Could not initialize AI model" in mock_warning.call_args[0][0]

            # Verify engine badge and history card label reflect the active tutor
            engine_str = "SymPy + fine-tuned model" if use_ai else "SymPy engine"
            assert engine_str == "SymPy engine"

