import time
from openai import APIError, RateLimitError, APITimeoutError, OpenAI
import os
import httpx
import json
import itertools
from io import BytesIO
import base64
from game import Round, Feedback


class GPTAgent:
    def __init__(self, model, response_save_path=None, generation_config=None):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("RRG_OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
            timeout=httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0),
        )

        self.generation_config = {
            "max_output_tokens": 64,
            "temperature": 0,
            "seed": 42,
            "timeout": 60,
        }

        if not generation_config is None:
            self.generation_config.update(generation_config)

        self.retry_after_seconds = 5
        self.retry_limit = 5
        self.response_save_path = response_save_path

    def api_call(self, messages):
        times_retried = 0
        while times_retried <= self.retry_limit:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    seed=self.generation_config["seed"],
                    max_tokens=self.generation_config["max_output_tokens"],
                    temperature=self.generation_config["temperature"],
                    timeout=60,
                )

                if not self.response_save_path is None:
                    with open(self.response_save_path, "a") as f:
                        f.write(
                            json.dumps(
                                {"messages": messages, "response": response.to_dict()}
                            )
                            + "\n"
                        )

                return response.choices[0].message.content
            except APIError as e:
                print(e)
                print(f"retrying in {self.retry_after_seconds} seconds")
                time.sleep(self.retry_after_seconds)
                times_retried += 1
                continue
            except RateLimitError as e:
                print(
                    f"Rate limit exceeded. Waiting and retrying in {self.retry_after_seconds} seconds..."
                )
                time.sleep(self.retry_after_seconds)
                times_retried += 1
                continue
            except APITimeoutError as e:
                print(e)
                time.sleep(self.retry_after_seconds)
                times_retried += 1
                continue
            except Exception as e:
                print(e)
                time.sleep(self.retry_after_seconds)
                times_retried += 1
                continue

        return None

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_label(self, index):
        return chr(ord("A") + index)


class GPTSpeaker(GPTAgent):
    def __init__(self, model, response_save_path=None, generation_config=None):
        super().__init__(model, response_save_path, generation_config)

    # def generate(self, context, target, previous_rounds=None):
    #     intro = self.get_intro()
    #     previous_round_messages = [self.format_round(r) for r in previous_rounds]
    #     current_round_messages = self.format_current_round(context, target)
    #     messages = [intro, *previous_round_messages, *current_round_messages]

    #     response = self.api_call(messages)
    #     selection = self.validate_response(response)


class GPTListener(GPTAgent):
    def __init__(
        self,
        model,
        images_once=True,
        previous_rounds=True,
        generation_config=None,
        response_save_path=None,
    ):
        super().__init__(model, response_save_path, generation_config)
        self.images_once = images_once
        self.previous_rounds = previous_rounds

    def get_intro(self):
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
                        "text": "Play a game with multiple rounds involving the same set of images. In each round, I will refer to one of the images with a message. You will guess which image I am referring to. If present, the history of previous rounds may help you better understand how I refer to specific images. In each round, answer with the image's label, i.e. one of [A, B, C, D]. You should still make a guess even when you are not sure. Do not output anything other than the image label you guess.",
                    }
                ],
            },
        ]

    def format_round(self, round, show_images=False, round_number=None):
        if round_number is not None:
            round_prompt = [{"type": "text", "text": f"Round {round_number}, "}]
        else:
            round_prompt = [{"type": "text", "text": f"Current round, "}]

        if show_images:
            images_prompt = itertools.chain.from_iterable(
                [
                    [
                        {"type": "text", "text": f"\nImage {self.get_label(i)}: "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.encode_image(image)}",
                            },
                        },
                    ]
                    for i, image in enumerate(round.context)
                ]
            )
        else:
            images_prompt = []

        message_prompt = [
            {
                "type": "text",
                "text": f"\nWhich image is this message referring to: {round.message} Output the image label only (a single letter).",
            }
        ]

        if not round.selection is None:
            assert (
                round.feedback is not None
            ), "Feedback must be provided if selection is provided."

            selection_prompt = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.get_label(round.selection)}.",
                        }
                    ],
                }
            ]

            if round.feedback == Feedback.CORRECT:
                feedback_prompt = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Correct."}],
                    }
                ]
            elif round.feedback == Feedback.INVALID:
                feedback_prompt = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Invalid answer. Answer must be one of {','.join([chr(ord('A') + i) for i, _ in enumerate(round.context)])}.",
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
                                "text": f"Wrong, I'm referring to image {self.get_label(round.target)}.",
                            }
                        ],
                    }
                ]
        else:
            selection_prompt = []
            feedback_prompt = []

        return [
            {
                "role": "user",
                "content": [
                    *round_prompt,
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

        selection = ord(response[0].upper()) - ord("A")

        if selection < 0 or selection >= len(context):
            return None

        return selection

    def select(self, context, message, previous_rounds=None):
        intro = self.get_intro()
        previous_round_messages = (
            [
                self.format_round(
                    r, show_images=(False if (i > 0) and self.images_once else True)
                )
                for i, r in enumerate(previous_rounds)
            ]
            if self.previous_rounds
            else []
        )
        current_round_messages = self.format_round(
            Round(context, None, message, None, None),
            show_images=(
                False if len(previous_rounds) > 0 and self.images_once else True
            ),
            round_number=len(previous_rounds) + 1 if self.previous_rounds else None,
        )

        messages = [
            *intro,
            *itertools.chain.from_iterable(previous_round_messages),
            *current_round_messages,
        ]

        response = self.api_call(messages)
        return self.validate_response(response, context)
