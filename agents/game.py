from typing import List, Tuple, Optional, Annotated
from pydantic import BaseModel, Field
from itertools import batched


class Trial(BaseModel):
    target: str
    message: Optional[str] = None
    interpretation: Optional[dict] = None

    def get_target(self) -> str:
        return self.target

    def get_message(self) -> Optional[str]:
        return self.message

    def get_selection(self) -> Optional[str]:
        if self.interpretation:
            return max(self.interpretation, key=self.interpretation.get)
        return None

    def get_correct(self) -> Optional[bool]:
        if self.interpretation:
            return max(self.interpretation, key=self.interpretation.get) == self.target
        return None

    def get_interpretation(self) -> Optional[dict]:
        return self.interpretation


class RepeatedReferenceGame(BaseModel):
    context: Tuple[str, ...]
    trials: Annotated[List[Trial], Field(default_factory=list)]

    def validate_block_structure(self):
        for block in batched(self.trials, len(self.context)):
            targets = [trial.target for trial in block]
            if len(targets) < len(self.context):
                if not len(set(targets)) == len(targets):
                    return False
            else:
                if not set(targets) == set(self.context):
                    return False

        return True
