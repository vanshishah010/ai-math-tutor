"""AI Step-by-Step Math Tutor Package.

Exposes the primary high-level interface `MathTutor` along with core data models
`TutorSolution`, `TutorHint`, `StepFeedback`, and `Step`.
"""

from tutor.core.solver import Step, TutorSolution
from tutor.core.verifier import StepFeedback
from tutor.core.hints import TutorHint
from tutor.core.tutor import MathTutor

__all__ = [
    "MathTutor",
    "TutorSolution",
    "TutorHint",
    "StepFeedback",
    "Step",
]

__version__ = "1.0.0"
