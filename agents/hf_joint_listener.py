from typing import Tuple, List
import torch
from PIL import Image
import random
import sys
from copy import deepcopy
import itertools
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from .chat_listener import ChatListener
from .hf_listeners import ScoringListener
from .hf_speakers import ScoringSpeaker


class JointInferenceListener:
    def __init__(
        self,
        listener_model,
        listener_processor,
        speaker_model,
        speaker_processor,
        listener_lambda=0.5,
        image_base_path: str = "",
        listener_context_presentation="block_shuffle",
        speaker_context_presentation="once",
        listener_feedback_label=False,
        speaker_feedback_label=False,
        speaker_prompt_type="standard",
    ):
        self.listener = ScoringListener(
            listener_model,
            listener_processor,
            image_base_path,
            listener_context_presentation,
            listener_feedback_label,
        )
        self.speaker = ScoringSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_prompt_type,
        )

        self.listener_lambda = listener_lambda

    def score(self, repeated_reference_game):
        listener_outputs = self.listener.score(repeated_reference_game)
        speaker_outputs = {
            referent: self.speaker.score(
                RepeatedReferenceGame(
                    context=repeated_reference_game.context,
                    trials=[
                        *repeated_reference_game.trials[:-1],
                        Trial(
                            target=referent,
                            message=repeated_reference_game.trials[-1].message,
                            selection=referent,
                            correct=True,
                        ),
                    ],
                )
            )
            for referent in repeated_reference_game.context
        }

        speaker_logprobs = torch.tensor(
            [speaker_outputs[r] for r in repeated_reference_game.context]
        )
        listener_logprobs = torch.tensor(
            [listener_outputs[r] for r in repeated_reference_game.context]
        )

        joint_logprobs_unnormalized = (
            listener_logprobs * self.listener_lambda
            + (1 - self.listener_lambda) * speaker_logprobs
        )

        joint_logprobs = (
            joint_logprobs_unnormalized - joint_logprobs_unnormalized.logsumexp(dim=-1)
        )

        return {
            r: p.item() for r, p in zip(repeated_reference_game.context, joint_logprobs)
        }

    def select(self, repeated_reference_game):
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)
        logprobs = self.score(repeated_reference_game)
        return max(logprobs.items(), key=lambda x: x[1])[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
