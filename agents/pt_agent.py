import itertools
import random
from typing import Any, Literal, Optional, Tuple, Union
import json
from game import RepeatedReferenceGame, Trial


class BaseVLMAgent:
    def __init__(
        self,
        context_presentation: Literal[
            "once",
            "no_history",
            "last_shuffle",
            "last_no_shuffle",
            "trial_shuffle",
            "block_shuffle",
        ] = "once",
        feedback_label: bool = False,
        demonstration_game: Optional[Union[RepeatedReferenceGame, str]] = None,
    ):
        # presentation of images and prior trials of the game in the context
        assert context_presentation in [
            "once",  # present context once at the beginning of the game
            "no_history",  # do not present any prior trials
            "last_shuffle",  # present the images once upfront and shuffle the images only for the last trial
            "last_no_shuffle",  # present the images once upfront and do not shuffle the images for the last trial
            "trial_shuffle",  # shuffle the images and present them for each trial
            "block_shuffle",  # shuffle the images and present them at the beginning of each block
        ]
        self.context_presentation = context_presentation

        # whether to include the true label in the feedback message
        self.feedback_label = feedback_label

        if isinstance(demonstration_game, str):
            with open(demonstration_game, "r") as f:
                demonstration_game_json = json.load(f)
            self.demonstration_game = RepeatedReferenceGame.model_validate(
                demonstration_game_json
            )
        else:
            self.demonstration_game = demonstration_game

    def get_images(self, messages: list[dict[str, Any]]):
        """
        Extracts image paths from the messages.
        Args:
            messages: List of message dictionaries.
        Returns:
            A list of image paths.
        """
        images = list()
        for msg in messages:
            for chunk in msg["content"]:
                if chunk["type"] == "image_url":
                    images.append(chunk["image_url"]["url"])

        return images

    def construct_prompt_messages(
        self,
        repeated_reference_game: RepeatedReferenceGame,
        exclude_feedback_on_last: bool = False,
        random_seed: Optional[int] = None,
    ):
        """
        Constructs the prompt messages for the chat agent based on the context presentation type.
        Args:
            repeated_reference_game: The repeated reference game object.
            exclude_feedback_on_last: Whether to exclude feedback on the last trial.
            random_seed: Random seed for shuffling.
        Returns:
            A tuple containing the prompt messages and a list of paths to all images appearing in the messages.
        """
        if self.context_presentation == "no_history":
            demonstration_messages = [
                self.format_trial(
                    self.demonstration_game.trials[-1],
                    self.demonstration_game.context,
                    show_images=True,
                    trial_number=None,
                    exclude_feedback=True,
                    demonstration=True,
                )
            ]
            trial_messages = [
                self.format_trial(
                    repeated_reference_game.trials[-1],
                    repeated_reference_game.context,
                    show_images=True,
                    trial_number=None,
                    exclude_feedback=True,
                )
            ]
            messages = [
                *demonstration_messages,
                *itertools.chain.from_iterable(trial_messages),
            ]
        elif self.context_presentation == "once":
            demonstration_messages = list()
            show_images = True
            for i, trial in enumerate(self.demonstration_game.trials):
                messages = self.format_trial(
                    trial,
                    self.demonstration_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=(
                        exclude_feedback_on_last
                        if i == len(self.demonstration_game.trials) - 1
                        else False
                    ),
                    is_demonstration=True,
                )
                if len(messages) > 0:
                    demonstration_messages.append(messages)
                    show_images = False
                else:
                    continue

            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=(
                        exclude_feedback_on_last
                        if i == len(repeated_reference_game.trials) - 1
                        else False
                    ),
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue
            messages = [
                *itertools.chain.from_iterable(demonstration_messages),
                *itertools.chain.from_iterable(trial_messages),
            ]
        elif self.context_presentation == "last_shuffle":
            if random_seed:
                random.seed(random_seed)

            demonstration_messages = list()
            show_images = True
            for i, trial in enumerate(self.demonstration_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    self.demonstration_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                    is_demonstration=True,
                )
                if len(messages) > 0:
                    demonstration_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_demonstration_trial = self.demonstration_game.trials[-1]
            last_demonstration_trial_context = random.sample(
                self.demonstration_game.context, len(self.demonstration_game.context)
            )
            last_demonstration_trial_messages = self.format_trial(
                last_demonstration_trial,
                last_demonstration_trial_context,
                show_images=True,
                trial_number=len(self.demonstration_game.trials),
                exclude_feedback=exclude_feedback_on_last,
                is_demonstration=True,
            )

            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_trial = repeated_reference_game.trials[-1]
            last_trial_context = random.sample(
                repeated_reference_game.context, len(repeated_reference_game.context)
            )
            last_trial_messages = self.format_trial(
                last_trial,
                last_trial_context,
                show_images=True,
                trial_number=len(repeated_reference_game.trials),
                exclude_feedback=exclude_feedback_on_last,
            )
            messages = [
                *demonstration_messages,
                *last_demonstration_trial_messages,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]
        elif self.context_presentation == "last_no_shuffle":
            demonstration_messages = list()
            show_images = True
            for i, trial in enumerate(self.demonstration_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    self.demonstration_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                    is_demonstration=True,
                )
                if len(messages) > 0:
                    demonstration_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_demonstration_trial = self.demonstration_game.trials[-1]
            last_demonstration_trial_context = self.demonstration_game.context
            last_demonstration_trial_messages = self.format_trial(
                last_demonstration_trial,
                last_demonstration_trial_context,
                show_images=True,
                trial_number=len(self.demonstration_game.trials),
                exclude_feedback=exclude_feedback_on_last,
                is_demonstration=True,
            )

            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_trial = repeated_reference_game.trials[-1]
            last_trial_context = repeated_reference_game.context
            last_trial_messages = self.format_trial(
                last_trial,
                last_trial_context,
                show_images=True,
                trial_number=len(repeated_reference_game.trials),
                exclude_feedback=exclude_feedback_on_last,
            )
            messages = [
                *itertools.chain.from_iterable(demonstration_messages),
                *last_demonstration_trial_messages,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]
        else:
            raise ValueError("Invalid context presentation type.")

        collapsed_messages = self.collapse_turns(messages)
        return collapsed_messages, self.get_images(collapsed_messages)

    def collapse_turns(self, messages: list[dict[str, Any]]):
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

    def encode_image(self, image_path: str):
        return str(image_path)


class BaseVLMSpeaker(BaseVLMAgent):
    def __init__(
        self,
        demonstration_game: Optional[Union[RepeatedReferenceGame, str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(demonstration_game=demonstration_game, *args, **kwargs)

    def get_label(
        self, context: Tuple[str], item: Optional[str], is_demonstration: bool = True
    ):
        if item is None:
            return "Invalid"
        elif is_demonstration:
            return chr(ord("M") + context.index(item))
        else:
            return chr(ord("A") + context.index(item))

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: Optional[int] = None,
        exclude_feedback: bool = False,
        is_demonstration: bool = False,
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
                                "text": f"\nImage {self.get_label(context, image, is_demonstration)}: ",
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
                "text": f"Image {self.get_label(context, trial.get_target(), is_demonstration)}: ",
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
                                    "text": f"Feedback: Invalid answer.",
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
                                    "text": f"Feedback: Correct answer {self.get_label(context, trial.get_selection(), is_demonstration)}.",
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
                                        f"Feedback: Incorrect answer {self.get_label(context, trial.get_selection(), is_demonstration)}."
                                        if self.feedback_label
                                        else "Feedback: Incorrect answer."
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

    def collapse_turns(self, messages: list[dict[str, Any]]):
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
