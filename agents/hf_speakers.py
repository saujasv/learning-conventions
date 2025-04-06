from typing import Tuple, List
import torch
from PIL import Image
import random
import itertools
from pathlib import Path
from transformers import PixtralProcessor
import warnings
from transformers.generation.logits_process import TopPLogitsWarper
from .chat_speaker import ChatSpeaker
from .hf_listeners import ScoringListener
from .utils import FIRELogitsWarper, TemperatureDecayLogitsWarper


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        generation_config=None,
        context_presentation="last_shuffle",
        feedback_label=False,
        prompt_type="standard",
        use_length_token=False,
        tangrams=True,
        max_image_size=None,
    ):
        ChatSpeaker.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            prompt_type=prompt_type,
            tangrams=tangrams,
        )

        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.use_length_token = use_length_token
        self.max_image_size = max_image_size

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

    def generate(
        self, repeated_reference_game, num_return_sequences=1, target_length=None
    ):
        messages, _ = self.construct_prompt_messages(repeated_reference_game)
        if target_length:
            if not self.use_length_token:
                warnings.warn("Length value is ignored when use_length_token is False")
                formatted_messages = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"<{target_length}>"}],
                    }
                )
                formatted_messages = self.processor.apply_chat_template(
                    messages, continue_final_message=True
                )
        else:
            formatted_messages = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

        if isinstance(self.processor, PixtralProcessor):
            processed = self.processor(
                text=formatted_messages,
                images=[
                    list(
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
                    )
                    for _ in range(num_return_sequences)
                ],
                return_tensors="pt",
                size=(
                    {"longest_edge": self.max_image_size}
                    if self.max_image_size
                    else None
                ),
            )
        else:
            processed = self.processor(
                text=formatted_messages,
                images=[
                    list(
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
                    )
                ],
                return_tensors="pt",
                size=(
                    {"longest_edge": self.max_image_size}
                    if self.max_image_size
                    else None
                ),
            )

        if self.inference_strategy == "sampling":
            logits_processor = None
        elif self.inference_strategy == "fire":
            logits_processor = [
                TopPLogitsWarper(self.generation_config["top_p"]),
                FIRELogitsWarper(
                    num_return_sequences=num_return_sequences,
                    standard_temperature=self.fire_standard_temperature,
                    fire_temperature=self.fire_temperature,
                ),
            ]
        elif self.inference_strategy == "temperature_decay":
            logits_processor = [
                TopPLogitsWarper(self.generation_config["top_p"]),
                TemperatureDecayLogitsWarper(
                    self.temperature_decay_scale, self.temperature_decay_target
                ),
            ]

        outputs = self.model.generate(
            **processed.to(self.model.device, self.model.dtype),
            **self.generation_config,
            tokenizer=self.processor.tokenizer,
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
        )

        response = [
            x.strip()
            for x in self.processor.batch_decode(
                outputs[:, processed.input_ids.shape[1] :], skip_special_tokens=True
            )
        ]

        return response if num_return_sequences > 1 else response[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class ScoringSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="once",
        feedback_label=False,
        prompt_type="standard",
    ):
        ChatSpeaker.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            prompt_type=prompt_type,
        )
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path

        self.text_only_assistant = False

    @torch.no_grad()
    def score(self, repeated_reference_game):
        messages, _ = self.construct_prompt_messages(
            repeated_reference_game, exclude_feedback_on_last=True
        )

        context = messages[:-1]
        formatted_context = self.processor.apply_chat_template(context)
        processed_context = self.processor(
            text=formatted_context,
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

        formatted_messages = self.processor.apply_chat_template(messages)
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

        labels = processed.input_ids.clone()
        labels[:, : processed_context.input_ids.shape[1]] = -100

        outputs = self.model(
            **processed.to(self.model.device, self.model.dtype),
            labels=labels,
            use_cache=False,
        )

        log_p = -outputs.loss.item() * (
            processed.input_ids.shape[1] - processed_context.input_ids.shape[1]
        )
        return log_p

    def generate(self, repeated_reference_game):
        raise NotImplementedError(
            "ScoringSpeaker can only be used for scoring sequences as a speaker agent"
        )

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
