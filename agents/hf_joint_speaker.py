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
from .hf_speakers import GenerateSpeaker


class JointInferenceSpeaker:
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        speaker_lambda=0.5,
        num_speaker_samples=5,
        listener_context_presentation="block_shuffle",
        speaker_context_presentation="once",
        listener_feedback_label=False,
        speaker_feedback_label=False,
        speaker_prompt_type="standard",
        speaker_generation_config=None,
    ):
        self.listener = ScoringListener(
            model,
            processor,
            image_base_path,
            context_presentation=listener_context_presentation,
            feedback_label=listener_feedback_label,
        )
        self.generate_speaker = GenerateSpeaker(
            model,
            processor,
            image_base_path,
            context_presentation=speaker_context_presentation,
            feedback_label=speaker_feedback_label,
            generation_config=speaker_generation_config,
            prompt_type=speaker_prompt_type,
        )
        self.scoring_speaker = GenerateSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            context_presentation=speaker_context_presentation,
            feedback_label=speaker_feedback_label,
            prompt_type=speaker_prompt_type,
        )

        self.num_speaker_samples = num_speaker_samples
        self.speaker_lambda = speaker_lambda

    def generate(self, repeated_reference_game):
        speaker_samples = list(
            set(
                self.generate_speaker.generate(repeated_reference_game)
                for _ in range(self.num_speaker_samples)
            )
        )
        speaker_logprobs = torch.tensor(
            [
                self.scoring_speaker.score(
                    RepeatedReferenceGame(
                        context=repeated_reference_game.context,
                        trials=[
                            *repeated_reference_game.trials[:-1],
                            Trial(
                                target=repeated_reference_game.trials[-1].target,
                                message=s,
                                selection=repeated_reference_game.trials[-1].target,
                                correct=True,
                            ),
                        ],
                    )
                )
                for s in speaker_samples
            ]
        )

        listener_logprobs = torch.tensor(
            [
                self.listener.score(
                    RepeatedReferenceGame(
                        context=repeated_reference_game.context,
                        trials=[
                            *repeated_reference_game.trials[:-1],
                            Trial(
                                target=repeated_reference_game.trials[-1].target,
                                message=s,
                                selection=repeated_reference_game.trials[-1].target,
                                correct=True,
                            ),
                        ],
                    )
                )[repeated_reference_game.trials[-1].target]
                for s in speaker_samples
            ]
        )

        joint_logprobs_unnormalized = (
            listener_logprobs * self.speaker_lambda
            + (1 - self.speaker_lambda) * speaker_logprobs
        )

        joint_logprobs = (
            joint_logprobs_unnormalized - joint_logprobs_unnormalized.logsumexp(dim=-1)
        )

        return speaker_samples[joint_logprobs.argmax().item()]
