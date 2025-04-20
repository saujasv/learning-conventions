from typing import Tuple, List
import torch
from PIL import Image
from accelerate import find_executable_batch_size
import itertools
from pathlib import Path
from transformers import PixtralProcessor
import warnings
from .chat_speaker import ChatSpeaker
from .hf_listeners import ScoringListener
from .prompts import SPEAKER_SYSTEM_PROMPT_STANDARD, SPEAKER_USER_PROMPT_PHOTOGRAPHS


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
        use_length_token=False,
        max_image_size=None,
    ):
        ChatSpeaker.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            system_prompt_template=system_prompt_template,
            user_prompt=user_prompt,
        )

        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.use_length_token = use_length_token

        self.generation_config = {
            "max_new_tokens": 64,
            "temperature": 0.3,
            "do_sample": True,
            "top_p": 0.9,
            "stop_strings": ["\n"],
        }

        self.inference_strategy = "sampling"

        if generation_config is not None:
            self.inference_strategy = generation_config.pop(
                "inference_strategy", "sampling"
            )
            if self.inference_strategy == "fire":
                self.fire_temperature = generation_config.pop("fire_temperature", 2.0)
                self.fire_standard_temperature = generation_config.pop(
                    "fire_standard_temperature", 0.3
                )
            elif self.inference_strategy == "temperature_decay":
                self.temperature_decay_scale = generation_config.pop(
                    "temperature_decay_scale", 2.0
                )
                self.temperature_decay_target = generation_config.pop(
                    "temperature_decay_target", 0.3
                )

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
            pad_token_id=self.processor.tokenizer.eos_token_id,
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
            self.processor.apply_chat_template(hist) for hist in histories
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
            self.processor.apply_chat_template(msg) for msg in messages
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

    @find_executable_batch_size(starting_batch_size=4)
    def batch_score_isolated(batch_size, self, repeated_reference_games):
        return list(
            itertools.chain.from_iterable(
                [
                    self.score_isolated(batch)
                    for batch in itertools.batched(repeated_reference_games, batch_size)
                ]
            )
        )

    @torch.no_grad()
    def score_isolated(self, repeated_reference_games):
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You should communicate the image to a listener in a message. The listener will try to choose the image correctly based on your message.\n\nYour reply should only contain the message and the message should always be shorter than 20 words. ",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.encode_image(game.trials[-1].target)
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe the image to the listener. Generate only the message containing the description.",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": game.trials[-1].message}],
                },
            ]
            for game in repeated_reference_games
        ]

        histories = [msg[:-1] for msg in messages]
        formatted_histories = [
            self.processor.apply_chat_template(hist) for hist in histories
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
            self.processor.apply_chat_template(msg) for msg in messages
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
