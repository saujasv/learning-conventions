from typing import Tuple, List
import itertools
import random
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from .chat_agent import ChatAgent


class ChatSpeaker(ChatAgent):
    def __init__(self, prompt_type: str = "standard", *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert prompt_type in [
            "standard",
            "explicit",
        ], f"Invalid prompt type {prompt_type}"
        self.prompt_type = prompt_type

    def get_label(self, context: Tuple[str], item: str):
        if item is None:
            return "Invalid"
        if item in context:
            return chr(ord("A") + context.index(item))

    def get_intro(self, context: Tuple[str]):
        if self.prompt_type == "standard":
            prompt = f"Play a repeated reference game with me and a third player (the listener). You will act as the speaker in the game. This game consists of multiple rounds in which the speaker interacts with me and the listener on the same referential context ({len(context)} images). In each round, I give you one of the {len(context)} images as the target. You should communicate the target to the listener in a message. The listener will try to choose the target correctly based on your message. I will tell you which image the listener chooses. The listener will see the {len(context)} images in a different order every round so you cannot communicate the target simply by using its position or label ({', '.join([chr(ord('A') + i) for i in range(len(context))])}).\n\nYour reply should only contain the message and the message should always be shorter than 20 words. Throughout, your message should not exceed one sentence but it does not need to be a full sentence."
        elif self.prompt_type == "explicit":
            prompt = f"Play a repeated reference game with me and a third player (the listener). You will act as the speaker in the game. This game consists of multiple rounds in which the speaker interacts with me and the listener on the same referential context ({len(context)} images). In each round, I give you one of the {len(context)} images as the target. You should communicate the target to the listener in a message. The listener will try to choose the target correctly based on your message. I will tell you which image the listener chooses. The listener will see the {len(context)} images in a different order every round so you cannot communicate the target simply by using its position or label ({', '.join([chr(ord('A') + i) for i in range(len(context))])}).\n\nYour reply should only contain the message and the message should always be shorter than 20 words. Throughout, your message should not exceed one sentence but it does not need to be a full sentence. Start with more detailed messages to ensure the listener's accuracy. As more rounds are completed and the listener understands you better, gradually condense your messages, making them shorter and shorter every round. When creating a shorter message for an image, try to extract salient tokens from the previous messages for this image rather than introducing new words. The short messages should still allow the listener to choose the target correctly. For each image, when you reach a message you think can not be further shortened without hurting the listener's accuracy, you should keep using that message for the rest of the game."

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant who will play a series of reference games with the user. You will generate a message referring to one of the images. The user will guess which image you are referring to. The images are of tangram shapes. Try to avoid referring to specific pieces of the tangram. Try to describe the shape as a whole. Feel free to use the resemblance to any real-world objects, and parts of those real world objects to describe the image.",
                    }
                ],
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
        if trial.target is None:
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
                "text": f"the target image is {self.get_label(context, trial.target)}.",
            }
        ]

        if not trial.correct is None:
            if trial.message is None:
                return []

            if self.text_only_assistant:
                message_prompt = [{"role": "assistant", "content": trial.message}]
            else:
                message_prompt = [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": trial.message,
                            }
                        ],
                    }
                ]

            if not exclude_feedback:
                if trial.selection is None:
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
                elif trial.correct:
                    feedback_prompt = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"The listener correctly answered Image {self.get_label(context, trial.selection)}.",
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
                                        f"The listener mistakenly answered Image {self.get_label(context, trial.selection)}."
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

    # def construct_prompt_messages(
    #     self, repeated_reference_game, exclude_feedback_on_last=False
    # ):
    #     intro = self.get_intro(repeated_reference_game.context)
    #     if self.context_presentation == "no_history":
    #         trial_messages = itertools.chain.from_iterable(
    #             [
    #                 self.format_trial(
    #                     repeated_reference_game.trials[-1],
    #                     repeated_reference_game.context,
    #                     trial_number=None,
    #                     exclude_feedback=True,
    #                     show_images=True,
    #                 )
    #             ]
    #         )
    #     elif self.context_presentation == "once":
    #         trial_messages = list()
    #         show_images = True
    #         for i, trial in enumerate(repeated_reference_game.trials):
    #             if trial.message is None:
    #                 continue

    #             trial_messages.append(
    #                 self.format_trial(
    #                     trial,
    #                     repeated_reference_game.context,
    #                     show_images=show_images,
    #                     trial_number=i + 1,
    #                     exclude_feedback=(
    #                         exclude_feedback_on_last
    #                         if i == len(repeated_reference_game.trials) - 1
    #                         else False
    #                     ),
    #                 )
    #             )
    #             show_images = False
    #     messages = [
    #         *intro,
    #         *itertools.chain.from_iterable(trial_messages),
    #     ]
    #     return self.collapse_turns(messages)

    def generate(self, repeated_reference_game):
        if not hasattr(self, "api_call"):
            raise NotImplementedError(
                "ChatSpeaker can only generate when an api_call method is implemented or the generate method is overriden."
            )

        return self.api_call(self.construct_prompt_messages(repeated_reference_game))
