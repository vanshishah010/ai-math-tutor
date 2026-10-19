"""Core modules for SymPy solving, verification, hints, and tutor orchestration."""

from tutor.core.solver import SymPySolver, Step, TutorSolution
from tutor.core.verifier import StepVerifier, StepFeedback
from tutor.core.hints import HintGenerator, TutorHint
from tutor.core.parser import MathParser, ParsedMathQuery
from tutor.core.graphing import MathPlotter
from tutor.core.tutor import MathTutor
from tutor.core.ocr import LocalMathOCR

__all__ = [
    "SymPySolver",
    "Step",
    "TutorSolution",
    "StepVerifier",
    "StepFeedback",
    "HintGenerator",
    "TutorHint",
    "MathParser",
    "ParsedMathQuery",
    "MathPlotter",
    "MathTutor",
    "LocalMathOCR",
]
