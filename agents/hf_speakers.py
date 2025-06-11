from typing import Any, Literal, Optional, Union
from PIL import Image
import itertools
from pathlib import Path
import torch
from accelerate import find_executable_batch_size
from transformers import PreTrainedModel, ProcessorMixin
import re
from copy import deepcopy
import os
from .chat_speaker import ChatSpeaker
from .pt_agent import BaseVLMSpeaker
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
    SPEAKER_USER_PROMPT_TARGET,
)
from .utils import ContrastiveDecodingProcessor
from game import RepeatedReferenceGame


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        generation_config=None,
        context_presentation="once",
        feedback_label=False,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_STANDARD,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS,
        target_prompt_template=SPEAKER_USER_PROMPT_TARGET,
        max_image_size=None,
        chat_template_file=None,
        contrastive_decoding=False,
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

        self.image_base_path = os.getenv("IMAGE_BASE_PATH", "")
        self.max_image_size = max_image_size
        self.contrastive_decoding = contrastive_decoding
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

    @find_executable_batch_size(starting_batch_size=64)
    def batch_generate(
        batch_size: int,
        self,
        repeated_reference_games: list[RepeatedReferenceGame],
        num_return_sequences: Optional[int] = None,
        target_lengths=None,
    ):
        # create individual tasks where each task is generating one output for one game
        if not target_lengths is None:
            assert isinstance(
                self, GenerateSpeaker
            ), "target_lengths is only supported for GenerateSpeaker"
            assert (
                num_return_sequences is None
                or len(target_lengths) == num_return_sequences
            )
            n_generations_per_prompt = len(target_lengths)
            tasks = list(
                itertools.chain.from_iterable(
                    [
                        [
                            {
                                "repeated_reference_games": [g],
                                "target_lengths": [l],
                            }
                            for l in target_lengths
                        ]
                        for g in repeated_reference_games
                    ]
                )
            )
        elif num_return_sequences:
            n_generations_per_prompt = num_return_sequences
            tasks = list(
                itertools.chain.from_iterable(
                    [
                        [
                            {"repeated_reference_games": [g]}
                            for _ in range(num_return_sequences)
                        ]
                        for g in repeated_reference_games
                    ]
                )
            )
        else:
            n_generations_per_prompt = 1
            tasks = [
                {"repeated_reference_games": [g]} for g in repeated_reference_games
            ]

        # group tasks into batches
        task_batches = [
            {
                "repeated_reference_games": list(
                    itertools.chain.from_iterable(
                        [x["repeated_reference_games"] for x in batch]
                    )
                ),
                "num_return_sequences": None,
                "target_lengths": (
                    list(
                        itertools.chain.from_iterable(
                            [x["target_lengths"] for x in batch]
                        )
                    )
                    if target_lengths
                    else None
                ),
            }
            for batch in itertools.batched(tasks, batch_size)
        ]

        # generate outputs for each task in the batch
        outputs = list(
            itertools.chain.from_iterable(
                [self.generate(**batch) for batch in task_batches]
            )
        )

        # reorganize the outputs to be one list of outputs for each game
        return list(
            [
                list(itertools.chain.from_iterable(b))
                for b in itertools.batched(outputs, n_generations_per_prompt)
            ]
        )

    def generate(
        self,
        repeated_reference_games: list[RepeatedReferenceGame],
        num_return_sequences: Optional[int] = None,
        target_lengths: Optional[list[int]] = None,
    ):
        batch_messages, batch_image_paths = zip(
            *[self.construct_prompt_messages(game) for game in repeated_reference_games]
        )

        batch_prompts = list()
        batch_images = list()
        if target_lengths:
            assert isinstance(
                self, GenerateSpeaker
            ), "target_lengths is only supported for GenerateSpeaker"
            assert (
                num_return_sequences is None
                or len(target_lengths) == num_return_sequences
            )
            n_generations_per_prompt = len(target_lengths)

            for image_paths, messages in zip(batch_image_paths, batch_messages):
                prompts = list()
                images = list()
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
                                target=label,
                                content=f"a {target_length}-word description",
                            )
                        )
                    prompts.append(prompt_messages)
                    images.append(image_paths)
                batch_prompts.append(prompts)
                batch_images.append(images)
        elif num_return_sequences:
            n_generations_per_prompt = num_return_sequences
            for messages, image_paths in zip(batch_messages, batch_image_paths):
                batch_prompts.append([messages for _ in range(num_return_sequences)])
                batch_images.append([image_paths for _ in range(num_return_sequences)])
        else:
            n_generations_per_prompt = 1
            for messages, image_paths in zip(batch_messages, batch_image_paths):
                batch_prompts.append([messages])
                batch_images.append([image_paths])

        formatted_messages = self.processor.apply_chat_template(
            list(itertools.chain.from_iterable(batch_prompts)),
            add_generation_prompt=True,
            chat_template=self.chat_template,
        )

        batch_image_objects = [
            [Image.open(img).convert("RGB") for img in imgs]
            for imgs in itertools.chain.from_iterable(batch_images)
        ]

        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        processed = self.processor(
            text=formatted_messages,
            images=batch_image_objects,
            return_tensors="pt",
            size={"longest_edge": self.max_image_size} if self.max_image_size else None,
            padding=True,
        )
        self.processor.tokenizer.padding_side = original_padding_side

        if self.contrastive_decoding:
            outputs = self.model.generate(
                **processed.to(self.model.device, self.model.dtype),
                **self.generation_config,
                tokenizer=self.processor.tokenizer,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                logits_processor=[
                    ContrastiveDecodingProcessor(
                        list(
                            itertools.chain.from_iterable(
                                [
                                    [game for _ in range(n_generations_per_prompt)]
                                    for game in repeated_reference_games
                                ]
                            )
                        ),
                        self,
                        alpha=0.05,
                        amateur_temp=1.0,
                    ),
                ],
            )
        else:
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

        return list(itertools.batched(response, n_generations_per_prompt))

    @find_executable_batch_size(starting_batch_size=4)
    def batch_score(batch_size, self, repeated_reference_games):
        return list(
            itertools.chain.from_iterable(
                [
                    self.score(batch)
                    for batch in itertools.batched(repeated_reference_games, batch_size)
                ]
            )
        )

    @torch.no_grad()
    def score(self, repeated_reference_games):
        messages = [
            self.construct_prompt_messages(game, exclude_feedback_on_last=True)[0]
            for game in repeated_reference_games
        ]

        histories = [msg[:-1] for msg in messages]
        formatted_histories = [
            self.processor.apply_chat_template(hist, chat_template=self.chat_template)
            for hist in histories
        ]
        processed_histories = self.processor(
            text=formatted_histories,
            images=[
                list(
                    itertools.chain.from_iterable(
                        [
                            [
                                Image.open(chunk["image_url"]["url"]).convert("RGB")
                                for chunk in m["content"]
                                if chunk["type"] == "image_url"
                            ]
                            for m in hist
                        ]
                    )
                )
                for hist in histories
            ],
            return_tensors="pt",
            padding=True,
        )

        first_pad_indices = []
        for mask in processed_histories.attention_mask:
            pad_positions = (mask == 0).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                first_pad_indices.append(pad_positions[0].item())
            else:
                first_pad_indices.append(processed_histories.input_ids.shape[1])

        formatted_messages = [
            self.processor.apply_chat_template(msg, chat_template=self.chat_template)
            for msg in messages
        ]
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
                            for m in msg
                        ]
                    )
                )
                for msg in messages
            ],
            return_tensors="pt",
            padding=True,
        )

        labels = processed.input_ids.clone()
        labels_pad_token_mask = labels == self.processor.tokenizer.pad_token_id
        labels[labels_pad_token_mask] = -100
        for i, pad_idx in enumerate(first_pad_indices):
            if pad_idx != -1:
                labels[i, :pad_idx] = -100

        outputs = self.model(
            **processed.to(self.model.device, self.model.dtype),
            labels=labels,
            use_cache=False,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).to(shift_logits.device),
            reduction="none",
        )

        return (-1 * loss.view(shift_labels.shape).sum(dim=-1)).tolist()

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class BaseVLMGenerateSpeaker(BaseVLMSpeaker, GenerateSpeaker):
    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        generation_config: dict[str, Any] = None,
        context_presentation: Literal[
            "once", "last_shuffle", "last_no_shuffle"
        ] = "once",
        feedback_label: bool = False,
        max_image_size: Optional[int] = None,
        chat_template_file: Optional[str] = None,
        contrastive_decoding: bool = False,
        demonstration_game: Optional[Union[RepeatedReferenceGame, str]] = None,
    ):
        BaseVLMSpeaker.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            demonstration_game=demonstration_game,
        )

        self.model = model
        self.processor = processor
        if chat_template_file:
            with open(chat_template_file, "r") as f:
                self.chat_template = f.read()
        else:
            self.chat_template = None

        self.image_base_path = os.getenv("IMAGE_BASE_PATH", "")
        self.max_image_size = max_image_size
        self.contrastive_decoding = contrastive_decoding
        self.generation_config = {
            "max_new_tokens": 64,
            "temperature": 1.0,
            "do_sample": True,
            "top_p": 0.9,
            "stop_strings": ["\n"],
        }

        if generation_config is not None:
            self.generation_config.update(generation_config)

        self.text_only_assistant = False

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
