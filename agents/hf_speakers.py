from typing import Tuple, List
import torch
from PIL import Image
import random
from pathlib import Path
from .chat_speaker import ChatSpeaker


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="once",
        feedback_label=False,
        generation_config=None,
        prompt_type="standard",
    ):
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.prompt_type = prompt_type

        self.generation_config = {
            "max_new_tokens": 64,
            "temperature": 0.3,
            "do_sample": True,
            "top_p": 0.9,
            "stop_strings": ["\n"],
        }

        if generation_config is not None:
            self.generation_config.update(generation_config)

        self.text_only_assistant = False

    def generate(self, repeated_reference_game):
        messages = self.construct_prompt_messages(repeated_reference_game)
        formatted_messages = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        processed = self.processor(
            text=formatted_messages,
            images=list(
                itertools.chain.from_iterable(
                    [
                        [
                            Image.open(chunk["image_url"]["url"]).convert("RGB")
                            for chunk in m["content"]
                            if chunk["type"] == "image_url"
                        ]
                        for m in messages
                    ]
                )
            ),
            return_tensors="pt",
        )

        outputs = self.model.generate(
            **processed.to(self.model.device, self.model.dtype),
            **self.generation_config,
            tokenizer=self.processor.tokenizer,
        )

        response = self.processor.batch_decode(
            outputs[:, processed.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip(" \n\t\"'")

        return response

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class JointInferenceSpeaker(ChatSpeaker):
    def __init__(
        self,
        speaker_model,
        speaker_processor,
        listener_model,
        listener_processor,
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
            listener_model,
            listener_processor,
            image_base_path,
            listener_context_presentation,
            listener_feedback_label,
        )
        self.generate_speaker = GenerateSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_generation_config,
            speaker_prompt_type,
        )
        self.scoring_speaker = ScoringSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_prompt_type,
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
