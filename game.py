from typing import NamedTuple, List
from enum import Enum


class Feedback(str, Enum):
    CORRECT = 1
    INCORRECT = 2
    INVALID = 3


class Round(NamedTuple):
    context: List[str]
    target: int
    message: str
    selection: int
    feedback: Feedback
