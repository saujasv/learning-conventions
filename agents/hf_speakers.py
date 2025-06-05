from PIL import Image
import itertools
from pathlib import Path
import torch
from accelerate import find_executable_batch_size
import re
from copy import deepcopy
import os
from .chat_speaker import ChatSpeaker
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
    SPEAKER_USER_PROMPT_TARGET,
)
from .utils import ContrastiveDecodingProcessor


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        generation_config=None,
        context_presentation="last_shuffle",
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
        batch_size,
        self,
        repeated_reference_game,
        num_return_sequences=None,
        target_lengths=None,
    ):
        if not num_return_sequences is None:
            if target_lengths is None:
                target_lengths = [None for _ in range(num_return_sequences)]
            else:
                assert len(target_lengths) == num_return_sequences

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
        images = [Image.open(img).convert("RGB") for img in image_paths]

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

        if self.contrastive_decoding:
            outputs = self.model.generate(
                **processed.to(self.model.device, self.model.dtype),
                **self.generation_config,
                tokenizer=self.processor.tokenizer,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                logits_processor=[
                    ContrastiveDecodingProcessor(
                        [repeated_reference_game],
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

        return response

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
