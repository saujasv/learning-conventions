from typing import Tuple, List
import itertools
import random
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from .chat_agent import ChatAgent
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
    SPEAKER_USER_PROMPT_TARGET,
)


class ChatSpeaker(ChatAgent):
    def __init__(
        self,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_STANDARD,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS,
        target_prompt_template=SPEAKER_USER_PROMPT_TARGET,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.system_prompt_template = system_prompt_template
        self.user_prompt = user_prompt
        self.target_prompt_template = target_prompt_template

    def get_label(self, context: Tuple[str], item: str):
        if item is None:
            return "Invalid"
        if item in context:
            return chr(ord("A") + context.index(item))

    def get_intro(self, context: Tuple[str]):
        prompt = self.system_prompt_template.substitute(
            num_images=len(context),
            labels=", ".join([chr(ord("A") + i) for i in range(len(context))]),
        )
        if self.text_only_assistant:
            return [
                {
                    "role": "system",
                    "content": self.user_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
            ]
        else:
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.user_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
            ]

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: int = None,
        exclude_feedback: bool = False,
    ):
        if trial.get_target() is None:
            raise ValueError("Trial must have target for speaker to generate")

        if trial_number is not None:
            trial_prompt = [{"type": "text", "text": f"Round {trial_number}, "}]
        else:
            trial_prompt = [{"type": "text", "text": f"Current round, "}]

        if show_images:
            images_prompt = list(
                itertools.chain.from_iterable(
                    [
                        [
                            {
                                "type": "text",
                                "text": f"\nImage {self.get_label(context, image)}: ",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.encode_image(image),
                                },
                            },
                        ]
                        for image in context
                    ]
                )
            )
        else:
            images_prompt = []

        target_prompt = [
            {
                "type": "text",
                "text": self.target_prompt_template.substitute(
                    target=self.get_label(context, trial.get_target()),
                    content="a description",
                ),
            }
        ]

        if not trial.get_correct() is None:
            if trial.get_message() is None:
                return []

            if hasattr(self, "use_length_token"):
                if self.use_length_token:
                    processor = getattr(self, "processor", None)
                    if not processor:
                        raise ValueError(
                            "Processor must be provided to use length token"
                        )
                    encoded = processor.tokenizer.encode(
                        trial.get_message(), add_special_tokens=False
                    )
                    formatted_message = f"<{len(encoded)}> {trial.get_message()}"
                else:
                    formatted_message = trial.get_message()
            else:
                formatted_message = trial.get_message()

            if self.text_only_assistant:
                message_prompt = [{"role": "assistant", "content": formatted_message}]
            else:
                message_prompt = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": formatted_message,
                            }
                        ],
                    }
                ]

            if not exclude_feedback:
                if trial.get_selection() is None:
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The listener didn't give a valid answer.",
                                }
                            ],
                        }
                    ]
                elif trial.get_correct():
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The listener correctly answered Image {self.get_label(context, trial.get_selection())}.",
                                }
                            ],
                        }
                    ]
                else:
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"The listener mistakenly answered Image {self.get_label(context, trial.get_selection())}."
                                        if self.feedback_label
                                        else "The listener answered incorrectly."
                                    ),
                                }
                            ],
                        }
                    ]
            else:
                feedback_prompt = []
        else:
            message_prompt = []
            feedback_prompt = []

        return [
            {
                "role": "user",
                "content": [
                    *trial_prompt,
                    *images_prompt,
                    *target_prompt,
                ],
            },
            *message_prompt,
            *feedback_prompt,
        ]

    def collapse_turns(self, messages):
        collapsed_messages = list()
        for m in messages:
            if (
                len(collapsed_messages) > 0
                and collapsed_messages[-1]["role"] == m["role"]
            ):
                collapsed_messages[-1]["content"] += m["content"]
            else:
                collapsed_messages.append(m)

        return collapsed_messages

    def generate(self, repeated_reference_game):
        if not hasattr(self, "api_call"):
            raise NotImplementedError(
                "ChatSpeaker can only generate when an api_call method is implemented or the generate method is overriden."
            )

        return self.api_call(self.construct_prompt_messages(repeated_reference_game))
