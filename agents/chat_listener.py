from typing import Tuple, List
import itertools
import random
from game import RepeatedReferenceGame, Trial


class ChatListener:
    def get_label(self, context: Tuple[str], item: str):
        if item is None:
            return "Invalid"
        if item in context:
            return chr(ord("A") + context.index(item))

    def get_intro(self, context: Tuple[str]):
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

    def format_trial(
        self,
        trial: Trial,
        context: Tuple[str],
        show_images: bool = False,
        trial_number: int = None,
        exclude_feedback: bool = False,
    ):
        if not trial.correct is None and trial.message is None:
            return list()

        if trial_number is not None:
            trial_prompt = [{"type": "text", "text": f"Round {trial_number}, "}]
        else:
            trial_prompt = [{"type": "text", "text": f"Current round, "}]

        if show_images:
            images_prompt = itertools.chain.from_iterable(
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
        else:
            images_prompt = []

        message_prompt = [
            {
                "type": "text",
                "text": f"\nWhich image is this message referring to: {trial.message}\nOutput the image label only (a single letter).",
            }
        ]

        if not trial.correct is None:
            if self.text_only_assistant:
                selection_prompt = [
                    {
                        "role": "assistant",
                        "content": f"{self.get_label(context, trial.selection)}.",
                    }
                ]
            else:
                selection_prompt = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{self.get_label(context, trial.selection)}.",
                            }
                        ],
                    }
                ]

            if not exclude_feedback:
                if trial.correct is None:
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Invalid answer. Answer must be one of {','.join([self.get_label(context, item) for item in enumerate(context)])}.",
                                }
                            ],
                        }
                    ]
                elif trial.correct:
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Correct."}],
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
                                        f"Wrong, I'm referring to image {self.get_label(context, trial.target)}."
                                        if self.feedback_label
                                        else "Wrong."
                                    ),
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
        if not isinstance(response, str):
            return None

        selection_idx = ord(response[0].upper()) - ord("A")

        if selection_idx < 0 or selection_idx >= len(context):
            return None

        return context[selection_idx]

    def construct_prompt_messages(
        self, repeated_reference_game, exclude_feedback_on_last=False, random_seed=None
    ):
        intro = self.get_intro(repeated_reference_game.context)
        if self.context_presentation == "no_history":
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
                *itertools.chain.from_iterable(trial_messages),
            ]

            return self.collapse_turns(messages), repeated_reference_game.context
        elif self.context_presentation == "once":
            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials):
                if trial.message is None:
                    continue

                trial_messages.append(
                    self.format_trial(
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
                )
                show_images = False
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            return self.collapse_turns(messages), repeated_reference_game.context
        elif self.context_presentation == "trial_shuffle":
            if random_seed:
                random.seed(random_seed)

            trial_messages = []
            for i, trial in enumerate(repeated_reference_game.trials):
                trial_context = random.sample(
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
            return self.collapse_turns(messages), trial_context
        elif self.context_presentation == "block_shuffle":
            if random_seed:
                random.seed(random_seed)
            if not repeated_reference_game.validate_block_structure():
                raise ValueError("Game does not have correct block structure.")

            trial_messages = list()
            trial_counter = 0
            block_context = None

            if len(repeated_reference_game.trials) == 0:
                block_context = random.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )

            for block_idx, block in enumerate(
                itertools.batched(
                    repeated_reference_game.trials, len(repeated_reference_game.context)
                )
            ):
                block_context = random.sample(
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
            return self.collapse_turns(messages), block_context
        else:
            raise ValueError("Invalid context presentation type.")

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

    def select(self, repeated_reference_game):
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)
        messages, context = self.construct_prompt_messages(repeated_reference_game)
        response = self.api_call(messages)
        return self.validate_response(response, context)
