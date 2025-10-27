from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
    SPEAKER_USER_PROMPT_TARGET,
)
from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class AgentArguments:
    agent_type: Literal["listener", "speaker"]
    model_type: Literal["base", "chat"]
    context_presentation: Literal["once", "last_shuffle", "last_no_shuffle"] = "once"
    feedback_label: bool = True
    max_image_size: Optional[int] = None
    chat_template_file: Optional[str] = None
    demonstration_game: Optional[str] = None
    contrastive_decoding: bool = False
    system_prompt_template: Optional[str] = None
    user_prompt: Optional[str] = None
    target_prompt_template: Optional[str] = None
    ensemble: int = 1
