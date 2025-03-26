from typing import Tuple, List
import itertools
import random
from game import RepeatedReferenceGame, Trial
from .chat_agent import ChatAgent


class ChatListener(ChatAgent):
    def __init__(self, tangrams=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tangrams = tangrams

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

    def select(self, repeated_reference_game):
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)
        messages, context = self.construct_prompt_messages(repeated_reference_game)
        response = self.api_call(messages)
        return self.validate_response(response, context)
