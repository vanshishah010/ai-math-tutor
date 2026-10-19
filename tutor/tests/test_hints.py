"""Unit tests for HintGenerator."""

import pytest
from tutor.core.hints import HintGenerator


@pytest.fixture
def generator():
    return HintGenerator()


def test_hint_level_1_conceptual(generator):
    hint = generator.get_hint("3*x + 7 = 22", level=1, variable="x")
    assert hint.level == 1
    assert hint.level_name == "Conceptual Strategy"
    assert "isolate" in hint.hint_text.lower() or "balanced" in hint.hint_text.lower()
    assert hint.next_level_available is True


def test_hint_level_2_action(generator):
    hint = generator.get_hint("3*x + 7 = 22", level=2, variable="x")
    assert hint.level == 2
    assert hint.level_name == "Next Algebraic Action"
    assert len(hint.hint_text) > 10
    assert hint.next_level_available is True


def test_hint_level_3_checkpoint(generator):
    hint = generator.get_hint("3*x + 7 = 22", level=3, variable="x")
    assert hint.level == 3
    assert hint.level_name == "Concrete Intermediate Step"
    assert "checkpoint" in hint.hint_text.lower()
    assert hint.next_level_available is True


def test_hint_level_4_full_solution(generator):
    hint = generator.get_hint("3*x + 7 = 22", level=4, variable="x")
    assert hint.level == 4
    assert hint.level_name == "Full Solution"
    assert "Final Answer" in hint.hint_text
    assert hint.next_level_available is False
    assert hint.solution is not None
