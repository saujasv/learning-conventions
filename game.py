from typing import NamedTuple, List, Tuple, Optional, Any, Annotated
from enum import Enum
from pydantic import BaseModel, RootModel, model_validator, Field
from collections import defaultdict
from itertools import batched


class Trial(BaseModel):
    target: str
    message: Optional[str] = None
    selection: Optional[str] = None
    correct: Optional[bool] = None


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
