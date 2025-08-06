import itertools
import random
from typing import Any, Literal, Optional, Tuple, Union
import json
from game import RepeatedReferenceGame, Trial


class BaseAgent:
    def __init__(
        self,
        model_type: Literal["base", "chat"] = "base",
        context_presentation: Literal[
            "once",
            "no_history",
            "last_shuffle",
            "last_no_shuffle",
            "trial_shuffle",
            "block_shuffle",
        ] = "once",
        feedback_label: bool = True,
        demonstration_game: Optional[Union[RepeatedReferenceGame, str]] = None,
    ):
        """
        Unified base agent that combines functionality from BaseVLMAgent and ChatAgent.

        Args:
            model_type: Whether this is a "base" VLM agent or "chat" agent mode
            context_presentation: How to present images and prior trials in context
            feedback_label: Whether to include the true label in feedback messages
            demonstration_game: Optional demonstration game for base mode
        """
        # Validate agent type
        assert model_type in ["base", "chat"], f"Invalid model_type: {model_type}"
        self.model_type = model_type

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

        # demonstration game support (primarily for base mode)
        if demonstration_game is not None:
            if isinstance(demonstration_game, str):
                with open(demonstration_game, "r") as f:
                    demonstration_game_json = json.load(f)
                self.demonstration_game = RepeatedReferenceGame.model_validate(
                    demonstration_game_json
                )
            else:
                self.demonstration_game = demonstration_game
        else:
            self.demonstration_game = None

    def get_intro(self, context: Tuple[str]):
        """
        Get introduction messages. Only used in chat mode.
        In base mode, returns empty list.
        """
        if self.model_type == "chat":
            raise NotImplementedError(
                "The introduction prompt is specific to the agent type (speaker/listener). "
                "Subclasses must implement this method."
            )
        else:
            return []

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
        Constructs the prompt messages based on the context presentation type.
        Combines logic from both BaseVLMAgent and ChatAgent.

        Args:
            repeated_reference_game: The repeated reference game object.
            exclude_feedback_on_last: Whether to exclude feedback on the last trial.
            random_seed: Random seed for shuffling.
        Returns:
            A tuple containing the prompt messages and a list of paths to all images appearing in the messages.
        """
        # Create RNG object if random_seed is provided
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        # Get intro messages (empty for base mode, implemented by subclasses for chat mode)
        if self.model_type == "chat":
            intro = self.get_intro(repeated_reference_game.context)
        else:
            intro = []

        if self.context_presentation == "no_history":
            # Handle demonstration game (base mode only)
            if self.model_type == "base" and self.demonstration_game:
                demonstration_messages = [
                    self.format_trial(
                        self.demonstration_game.trials[-1],
                        self.demonstration_game.context,
                        show_images=True,
                        trial_number=None,
                        exclude_feedback=True,
                        is_demonstration=True,
                    )
                ]
            else:
                demonstration_messages = []

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
                *intro,
                *itertools.chain.from_iterable(demonstration_messages),
                *itertools.chain.from_iterable(trial_messages),
            ]

        elif self.context_presentation == "once":
            # Handle demonstration game (base mode only)
            demonstration_messages = list()
            show_images = True
            if self.model_type == "base" and self.demonstration_game:
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
                *intro,
                *itertools.chain.from_iterable(demonstration_messages),
                *itertools.chain.from_iterable(trial_messages),
            ]

        elif self.context_presentation == "last_shuffle":
            # Handle demonstration game (base mode only)
            demonstration_messages = list()
            show_images = True
            if self.model_type == "base" and self.demonstration_game:
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
                last_demonstration_trial_context = rng.sample(
                    self.demonstration_game.context,
                    len(self.demonstration_game.context),
                )
                last_demonstration_trial_messages = self.format_trial(
                    last_demonstration_trial,
                    last_demonstration_trial_context,
                    show_images=True,
                    trial_number=len(self.demonstration_game.trials),
                    exclude_feedback=exclude_feedback_on_last,
                    is_demonstration=True,
                )
            else:
                last_demonstration_trial_messages = []

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
            last_trial_context = rng.sample(
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
                *intro,
                *itertools.chain.from_iterable(demonstration_messages),
                *last_demonstration_trial_messages,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]

        elif self.context_presentation == "last_no_shuffle":
            # Handle demonstration game (base mode only)
            demonstration_messages = list()
            show_images = True
            if self.model_type == "base" and self.demonstration_game:
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
            else:
                last_demonstration_trial_messages = []

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
                *intro,
                *itertools.chain.from_iterable(demonstration_messages),
                *last_demonstration_trial_messages,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]

        elif self.context_presentation == "trial_shuffle":
            trial_messages = []
            for i, trial in enumerate(repeated_reference_game.trials):
                trial_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )
                trial_messages.append(
                    self.format_trial(
                        trial,
                        trial_context,
                        show_images=True,
                        trial_number=i + 1,
                        exclude_feedback=(
                            exclude_feedback_on_last
                            if i == len(repeated_reference_game.trials) - 1
                            else False
                        ),
                    )
                )
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]

        elif self.context_presentation == "block_shuffle":
            if not repeated_reference_game.validate_block_structure():
                raise ValueError("Game does not have correct block structure.")

            trial_messages = list()
            trial_counter = 0
            block_context = None

            if len(repeated_reference_game.trials) == 0:
                block_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )

            for block_idx, block in enumerate(
                itertools.batched(
                    repeated_reference_game.trials, len(repeated_reference_game.context)
                )
            ):
                block_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )
                block_trials = list()
                block_start_counter = trial_counter
                for trial in block:
                    messages = self.format_trial(
                        trial,
                        block_context,
                        show_images=trial_counter == block_start_counter,
                        trial_number=trial_counter + 1,
                        exclude_feedback=(
                            exclude_feedback_on_last
                            if trial_counter == len(repeated_reference_game.trials) - 1
                            else False
                        ),
                    )
                    if len(messages) > 0:
                        block_trials.append(messages)
                        trial_counter += 1

                trial_messages.extend(block_trials)

            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]

            assert block_context is not None, "Block context should not be None."

        else:
            raise ValueError("Invalid context presentation type.")

        collapsed_messages = self.collapse_turns(messages)
        return collapsed_messages, self.get_images(collapsed_messages)

    def collapse_turns(self, messages: list[dict[str, Any]]):
        """Collapse consecutive messages from the same role."""
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
        """Encode image path for use in messages."""
        return str(image_path)

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: Optional[int] = None,
        exclude_feedback: bool = False,
        is_demonstration: bool = False,
    ):
        """
        Format a trial for inclusion in prompt messages.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "format_trial must be implemented by subclasses (Speaker/Listener)."
        )


class BaseListener(BaseAgent):
    def __init__(self, *args, **kwargs):
        """
        Unified BaseListener that combines functionality from ChatListener and BaseVLMListener.

        Args:
            *args, **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(*args, **kwargs)

    def get_label(
        self, context: Tuple[str], item: Optional[str], is_demonstration: bool = False
    ):
        """Get the label for an item based on the agent type and demonstration status."""
        if item is None:
            return "Invalid"

        # Base/VLM mode uses M-Z for demonstrations, A-Z for regular games
        if is_demonstration:
            return chr(ord("M") + context.index(item))
        else:
            return chr(ord("A") + context.index(item))

    def get_intro(self, context: Tuple[str]):
        """Get introduction messages for chat mode."""
        if self.model_type == "chat":
            return [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Play a game with multiple rounds involving the same set of images. In each round, I will refer to one of the images with a message. You will guess which image I am referring to. If present, the history of previous rounds may help you better understand how I refer to specific images. In each round, answer with the image's label, i.e. one of [{', '.join([chr(i + ord('A')) for i, _ in enumerate(context)])}]. You should still make a guess even when you are not sure. Do not output anything other than the image label you guess.",
                        }
                    ],
                },
            ]
        else:
            # Base models don't need an intro
            return []

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: Optional[int] = None,
        exclude_feedback: bool = False,
        is_demonstration: bool = False,
    ):
        """Format a trial for inclusion in prompt messages."""
        if not trial.get_correct() is None and trial.get_message() is None:
            return list()

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

        # Different message prompts for chat vs base mode
        if self.model_type == "chat":
            message_prompt = [
                {
                    "type": "text",
                    "text": f"\nWhich image is this message referring to: {trial.get_message()}\nOutput the image label only (a single letter).",
                }
            ]
        else:
            # Base/VLM mode
            message_prompt = [
                {
                    "type": "text",
                    "text": f"\nDescription: {trial.get_message()}\nImage:",
                }
            ]

        if not trial.get_correct() is None:
            selection_prompt = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.get_label(context, trial.get_selection(), is_demonstration)}.",
                        }
                    ],
                }
            ]

            if not exclude_feedback:
                if trial.get_correct() is None:
                    # Invalid answer feedback - different for chat vs base mode
                    if self.model_type == "chat":
                        feedback_text = f"Invalid answer. Answer must be one of {','.join([self.get_label(context, item) for i, item in enumerate(context)])}."
                    else:
                        feedback_text = "Feedback: Invalid answer."

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": feedback_text,
                                }
                            ],
                        }
                    ]
                elif trial.get_correct():
                    # Correct answer feedback
                    if self.model_type == "chat":
                        feedback_text = "Correct."
                    else:
                        feedback_text = f"Feedback: Correct answer {self.get_label(context, trial.get_selection(), is_demonstration)}."

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": feedback_text}],
                        }
                    ]
                else:
                    # Wrong answer feedback
                    if self.model_type == "chat":
                        feedback_text = (
                            f"Wrong, I'm referring to image {self.get_label(context, trial.get_target(), is_demonstration)}."
                            if self.feedback_label
                            else "Wrong."
                        )
                    else:
                        feedback_text = (
                            f"Feedback: Incorrect answer. Correct answer is {self.get_label(context, trial.get_target(), is_demonstration)}."
                            if self.feedback_label
                            else "Feedback: Incorrect answer."
                        )

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": feedback_text,
                                }
                            ],
                        }
                    ]
            else:
                feedback_prompt = []
        else:
            selection_prompt = []
            feedback_prompt = []

        return [
            {
                "role": "user",
                "content": [
                    *trial_prompt,
                    *images_prompt,
                    *message_prompt,
                ],
            },
            *selection_prompt,
            *feedback_prompt,
        ]

    def validate_response(self, response, context):
        """Validate the response from the model."""
        if not isinstance(response, str):
            return None

        selection_idx = ord(response[0].upper()) - ord("A")

        if selection_idx < 0 or selection_idx >= len(context):
            return None

        return context[selection_idx]


class BaseSpeaker(BaseAgent):
    def __init__(
        self,
        system_prompt_template=None,
        user_prompt=None,
        target_prompt_template=None,
        *args,
        **kwargs,
    ):
        """
        Unified BaseSpeaker that combines functionality from ChatSpeaker and BaseVLMSpeaker.

        Args:
            system_prompt_template: Template for system prompt (chat mode only)
            user_prompt: User prompt text (chat mode only)
            target_prompt_template: Template for target prompt (chat mode only)
            text_only_assistant: Whether assistant messages should only include text (required for Mistral API)
            *args, **kwargs: Additional arguments passed to BaseAgent
        """
        super().__init__(*args, **kwargs)

        # Import default prompts for chat mode
        if self.model_type == "chat":
            if system_prompt_template is None:
                from .prompts import SPEAKER_SYSTEM_PROMPT_BASIC

                self.system_prompt_template = SPEAKER_SYSTEM_PROMPT_BASIC
            else:
                self.system_prompt_template = system_prompt_template

            if user_prompt is None:
                from .prompts import SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC

                self.user_prompt = SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC
            else:
                self.user_prompt = user_prompt

            if target_prompt_template is None:
                from .prompts import SPEAKER_USER_PROMPT_TARGET_BASIC

                self.target_prompt_template = SPEAKER_USER_PROMPT_TARGET_BASIC
            else:
                self.target_prompt_template = target_prompt_template

    def get_label(
        self, context: Tuple[str], item: Optional[str], is_demonstration: bool = False
    ):
        """Get the label for an item based on the agent type and demonstration status."""
        if item is None:
            return "Invalid"

        # Base/VLM mode uses M-Z for demonstrations, A-Z for regular games
        if is_demonstration:
            return chr(ord("M") + context.index(item))
        else:
            return chr(ord("A") + context.index(item))

    def get_intro(self, context: Tuple[str]):
        """Get introduction messages for chat mode."""
        if self.model_type == "chat":
            prompt = self.system_prompt_template.substitute(
                num_images=len(context),
                labels=", ".join([chr(ord("A") + i) for i in range(len(context))]),
            )
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
        else:
            # Base models don't need an intro
            return []

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: Optional[int] = None,
        exclude_feedback: bool = False,
        is_demonstration: bool = False,
    ):
        """Format a trial for inclusion in prompt messages."""
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

        # Different target prompts for chat vs base mode
        if self.model_type == "chat":
            target_prompt = [
                {
                    "type": "text",
                    "text": self.target_prompt_template.substitute(
                        target=self.get_label(
                            context, trial.get_target(), is_demonstration
                        ),
                        content="a description",
                    ),
                }
            ]
        else:
            # Base/VLM mode
            target_prompt = [
                {
                    "type": "text",
                    "text": f"Image {self.get_label(context, trial.get_target(), is_demonstration)} description: ",
                }
            ]

        if not trial.get_correct() is None:
            if trial.get_message() is None:
                return []

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
                    # Invalid answer feedback - different for chat vs base mode
                    if self.model_type == "chat":
                        feedback_text = "The listener didn't give a valid answer."
                    else:
                        feedback_text = "Feedback: Invalid answer."

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": feedback_text,
                                }
                            ],
                        }
                    ]
                elif trial.get_correct():
                    # Correct answer feedback
                    if self.model_type == "chat":
                        feedback_text = f"The listener correctly answered Image {self.get_label(context, trial.get_selection(), is_demonstration)}."
                    else:
                        feedback_text = f"Feedback: Correct answer {self.get_label(context, trial.get_selection(), is_demonstration)}."

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": feedback_text,
                                }
                            ],
                        }
                    ]
                else:
                    # Wrong answer feedback
                    if self.model_type == "chat":
                        feedback_text = (
                            f"The listener mistakenly answered Image {self.get_label(context, trial.get_selection(), is_demonstration)}."
                            if self.feedback_label
                            else "The listener answered incorrectly."
                        )
                    else:
                        feedback_text = (
                            f"Feedback: Incorrect answer. Correct answer is {self.get_label(context, trial.get_target(), is_demonstration)}."
                            if self.feedback_label
                            else "Feedback: Incorrect answer."
                        )

                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": feedback_text,
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

    def generate(self, repeated_reference_game: RepeatedReferenceGame):
        """Generate a message for the speaker."""
        if not hasattr(self, "api_call"):
            raise NotImplementedError(
                "BaseSpeaker can only generate when an api_call method is implemented or the generate method is overridden."
            )

        messages, _ = self.construct_prompt_messages(repeated_reference_game)
        return self.api_call(messages)
