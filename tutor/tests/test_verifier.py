"""Unit tests for StepVerifier."""

import pytest
from tutor.core.verifier import StepVerifier


@pytest.fixture
def verifier():
    return StepVerifier()


def test_verify_linear_equation_valid_step(verifier):
    feedback = verifier.check_step(
        problem="3*x + 7 = 22",
        previous_steps=[],
        student_step="3*x = 15",
        variable="x",
    )
    assert feedback.is_valid is True
    assert feedback.status == "correct"


def test_verify_linear_equation_final_step(verifier):
    feedback = verifier.check_step(
        problem="3*x + 7 = 22",
        previous_steps=["3*x = 15"],
        student_step="x = 5",
        variable="x",
    )
    assert feedback.is_valid is True
    assert feedback.is_final_solution is True


def test_verify_linear_equation_arithmetic_or_sign_error(verifier):
    # Student adds 7 instead of subtracting 7: 3x = 29
    feedback = verifier.check_step(
        problem="3*x + 7 = 22",
        previous_steps=[],
        student_step="3*x = 29",
        variable="x",
    )
    assert feedback.is_valid is False
    assert feedback.status in ("arithmetic_error", "incorrect")
    assert "constant" in feedback.message.lower() or "arithmetic" in feedback.message.lower()


def test_verify_quadratic_valid_step(verifier):
    feedback = verifier.check_step(
        problem="x^2 - 5*x + 6 = 0",
        previous_steps=[],
        student_step="(x - 2)*(x - 3) = 0",
        variable="x",
    )
    assert feedback.is_valid is True
    assert feedback.status == "correct"


def test_verify_differentiation_valid_step(verifier):
    feedback = verifier.check_step(
        problem="x^3 + 2*x^2 - 5*x + 1",
        previous_steps=[],
        student_step="3*x^2 + 4*x - 5",
        variable="x",
        operation="diff",
    )
    assert feedback.is_valid is True
    assert feedback.status == "correct"


def test_verify_differentiation_sign_error(verifier):
    feedback = verifier.check_step(
        problem="x^3 + 2*x^2 - 5*x + 1",
        previous_steps=[],
        student_step="-(3*x^2 + 4*x - 5)",
        variable="x",
        operation="diff",
    )
    assert feedback.is_valid is False
    assert feedback.status == "sign_error"
    assert "sign" in feedback.message.lower()


def test_verify_simplification_valid_step(verifier):
    feedback = verifier.check_step(
        problem="(x^2 - 4)/(x - 2)",
        previous_steps=[],
        student_step="(x - 2)*(x + 2)/(x - 2)",
        variable="x",
        operation="simplify",
    )
    assert feedback.is_valid is True
    assert feedback.status == "correct"
