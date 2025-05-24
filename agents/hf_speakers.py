from typing import Tuple, List
import torch
from PIL import Image
import random
import itertools
from pathlib import Path
from transformers import PixtralProcessor
from accelerate import find_executable_batch_size
import re
from copy import deepcopy
from .chat_speaker import ChatSpeaker
from .hf_listeners import ScoringListener
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
    SPEAKER_USER_PROMPT_TARGET,
)


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        generation_config=None,
        context_presentation="last_shuffle",
        feedback_label=False,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_STANDARD,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS,
        target_prompt_template=SPEAKER_USER_PROMPT_TARGET,
        max_image_size=None,
        chat_template_file=None,
    ):
        ChatSpeaker.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            system_prompt_template=system_prompt_template,
            user_prompt=user_prompt,
            target_prompt_template=target_prompt_template,
        )

        self.model = model
        self.processor = processor
        if chat_template_file:
            with open(chat_template_file, "r") as f:
                self.chat_template = f.read()
        else:
            self.chat_template = None

        self.image_base_path = image_base_path
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

    @find_executable_batch_size(starting_batch_size=16)
    def batch_generate(
        batch_size,
        self,
        repeated_reference_game,
        num_return_sequences=None,
        target_lengths=None,
    ):
        if not num_return_sequences is None:
            target_lengths = [None for _ in range(num_return_sequences)]

        return list(
            itertools.chain.from_iterable(
                [
                    self.generate(
                        repeated_reference_game,
                        target_lengths=batch,
                    )
                    for batch in itertools.batched(target_lengths, batch_size)
                ]
            )
        )

    def generate(
        self, repeated_reference_game, num_return_sequences=None, target_lengths=None
    ):
        messages, image_paths = self.construct_prompt_messages(repeated_reference_game)
        images = [Image.open(img) for img in image_paths]

        prompts = list()
        if target_lengths:
            assert (
                num_return_sequences is None
                or len(target_lengths) == num_return_sequences
            )

            for i, target_length in enumerate(target_lengths):
                prompt_messages = deepcopy(messages)
                if not target_length is None:
                    label = re.match(
                        self.target_prompt_template.substitute(
                            target="([A-Z])", content="a description"
                        ),
                        prompt_messages[-1]["content"][-1]["text"],
                    ).group(1)
                    prompt_messages[-1]["content"][-1]["text"] = (
                        self.target_prompt_template.substitute(
                            target=label, content=f"a {target_length}-word description"
                        )
                    )
                prompts.append(prompt_messages)
        elif num_return_sequences:
            prompts.extend([messages for _ in range(num_return_sequences)])
        else:
            prompts.append(messages)

        formatted_messages = self.processor.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            chat_template=self.chat_template,
        )

        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        processed = self.processor(
            text=formatted_messages,
            images=[images for _ in prompts],
            return_tensors="pt",
            size={"longest_edge": self.max_image_size} if self.max_image_size else None,
            padding=True,
        )
        self.processor.tokenizer.padding_side = original_padding_side

        outputs = self.model.generate(
            **processed.to(self.model.device, self.model.dtype),
            **self.generation_config,
            tokenizer=self.processor.tokenizer,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        response = [
            x.strip()
            for x in self.processor.batch_decode(
                outputs[:, processed.input_ids.shape[1] :], skip_special_tokens=True
            )
        ]

        return response

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
