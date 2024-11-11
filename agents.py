from typing import Tuple, List
import time
from openai import APIError, RateLimitError, APITimeoutError, OpenAI
import os
import httpx
import json
import itertools
from io import BytesIO
import base64
import torch
from PIL import Image
import random
import sys
from vllm import LLM, SamplingParams
from copy import deepcopy
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from itertools import batched


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

            return messages, repeated_reference_game.context
        elif self.context_presentation == "once":
            trial_messages = [
                self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=i == 0,
                    trial_number=i + 1,
                    exclude_feedback=(
                        exclude_feedback_on_last
                        if i == len(repeated_reference_game.trials) - 1
                        else False
                    ),
                )
                for i, trial in enumerate(repeated_reference_game.trials)
            ]
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            return messages, repeated_reference_game.context
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
            return messages, trial_context
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
                batched(
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
        messages, context = self.construct_prompt_messages(repeated_reference_game)
        response = self.api_call(messages)
        return self.validate_response(response, context)


class ChatSpeaker:
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
                    },
                    *itertools.chain.from_iterable(
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
                    ),
                ],
            },
        ]

    def format_trial(self, context, trial, trial_number=None, exclude_feedback=False):
        if trial_number is not None:
            trial_prompt = [{"type": "text", "text": f"Round {trial_number}, "}]
        else:
            trial_prompt = [{"type": "text", "text": f"Current round, "}]

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

    def construct_prompt_messages(
        self, repeated_reference_game, exclude_feedback_on_last=False
    ):
        intro = self.get_intro(repeated_reference_game.context)
        if self.context_presentation == "no_history":
            trials_messages = [
                self.format_trial(
                    repeated_reference_game.context,
                    repeated_reference_game.trials[-1],
                    trial_number=None,
                    exclude_feedback=True,
                )
            ]

        elif self.context_presentation == "once":
            trials_messages = itertools.chain.from_iterable(
                [
                    self.format_trial(
                        repeated_reference_game.context,
                        trial,
                        trial_number=i + 1,
                        exclude_feedback=(
                            exclude_feedback_on_last
                            if i == len(repeated_reference_game.trials) - 1
                            else False
                        ),
                    )
                    for i, trial in enumerate(repeated_reference_game.trials)
                ]
            )
        messages = [*intro, *trials_messages]
        return self.collapse_turns(messages)

    def generate(self, repeated_reference_game):
        return self.api_call(self.construct_prompt_messages(repeated_reference_game))


class GPTAgent:
    def __init__(self, model, image_base_path, response_save_path=None):
        self.model = model
        self.client = OpenAI(
            timeout=httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0),
        )
        self.image_base_path = image_base_path
        self.retry_after_seconds = 5
        self.retry_limit = 5
        self.response_save_path = response_save_path
        self.text_only_assistant = False

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
        with open(Path(self.image_base_path) / image_path, "rb") as image_file:
            return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode("utf-8")}"


class vLLMAgent:
    def __init__(self, model, tensor_parallel_size=1, image_base_path: str = ""):
        if isinstance(model, str):
            self.llm = LLM(
                model=model,
                tokenizer_mode="mistral",
                limit_mm_per_prompt={"image": 256},
                max_model_len=20000,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif isinstance(model, LLM):
            self.llm = model

        self.text_only_assistant = False
        self.image_base_path = image_base_path

    def api_call(self, messages):
        outputs = self.llm.chat(
            messages=messages, sampling_params=self.generation_config
        )
        return outputs[0].outputs[0].text.strip("\"'")

    def encode_image(self, image_path):
        with open(Path(self.image_base_path) / image_path, "rb") as image_file:
            return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode("utf-8")}"


class GPTSpeaker(GPTAgent, ChatSpeaker):
    def __init__(
        self,
        model,
        image_base_path,
        context_presentation="once",
        feedback_label=False,
        response_save_path=None,
        generation_config=None,
        prompt_type="standard",
    ):
        GPTAgent.__init__(self, model, image_base_path, response_save_path)

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        self.generation_config = {
            "max_output_tokens": 64,
            "temperature": 0.5,
            "seed": 42,
            "timeout": 60,
        }

        if generation_config is not None:
            self.generation_config.update(generation_config)

        self.prompt_type = prompt_type


class GPTListener(GPTAgent, ChatListener):
    def __init__(
        self,
        model,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
        generation_config=None,
        response_save_path=None,
    ):
        GPTAgent.__init__(self, model, image_base_path, response_save_path)
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        self.generation_config = {
            "max_output_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "timeout": 60,
        }

        if generation_config is not None:
            self.generation_config.update(generation_config)


class vLLMSpeaker(vLLMAgent, ChatSpeaker):
    def __init__(
        self,
        model,
        image_base_path,
        context_presentation="once",
        feedback_label=False,
        tensor_parallel_size=1,
        generation_config=None,
        prompt_type="standard",
    ):
        vLLMAgent.__init__(self, model, tensor_parallel_size, image_base_path)

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        if generation_config is None:
            generation_config = {
                "max_tokens": 64,
                "temperature": 0.3,
            }

        self.generation_config = SamplingParams(**generation_config)

        self.prompt_type = prompt_type


class vLLMListener(vLLMAgent, ChatListener):
    def __init__(
        self,
        model,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
        generation_config=None,
        tensor_parallel_size=1,
    ):
        vLLMAgent.__init__(self, model, tensor_parallel_size, image_base_path)

        if generation_config is None:
            generation_config = {
                "max_tokens": 8,
                "temperature": 0,
            }

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        self.generation_config = SamplingParams(**generation_config)


class GenerateListener(ChatListener):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
    ):
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.text_only_assistant = False

    def select(self, repeated_reference_game):
        messages, context = self.construct_prompt_messages(
            repeated_reference_game, random_seed=412
        )
        formatted_messages = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        processed = self.processor(
            text=formatted_messages,
            images=list(
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
            ),
            return_tensors="pt",
        )

        outputs = self.model.generate(
            **processed.to(self.model.device, self.model.dtype),
            max_new_tokens=8,
            do_sample=False,
        )

        response = self.processor.batch_decode(
            outputs[:, processed.input_ids.shape[1] :]
        )[0].strip()

        return self.validate_response(response, context)

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class GenerateSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="once",
        feedback_label=False,
        generation_config=None,
        prompt_type="standard",
    ):
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.prompt_type = prompt_type

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

    def generate(self, repeated_reference_game):
        messages = self.construct_prompt_messages(repeated_reference_game)
        formatted_messages = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        processed = self.processor(
            text=formatted_messages,
            images=list(
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
            ),
            return_tensors="pt",
        )

        outputs = self.model.generate(
            **processed.to(self.model.device, self.model.dtype),
            **self.generation_config,
            tokenizer=self.processor.tokenizer,
        )

        response = self.processor.batch_decode(
            outputs[:, processed.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip(" \n\t\"'")

        return response

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class ScoringListener(ChatListener):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
    ):
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.text_only_assistant = False

    @torch.no_grad()
    def score(self, repeated_reference_game):
        # construct counterfactual games by permuting the options to take each of the possible values
        counterfactual_games = [
            RepeatedReferenceGame(
                context=repeated_reference_game.context,
                trials=[
                    *repeated_reference_game.trials[:-1],
                    Trial(
                        target=repeated_reference_game.trials[-1].target,
                        message=repeated_reference_game.trials[-1].message,
                        selection=c,
                        correct=c == repeated_reference_game.trials[-1].target,
                    ),
                ],
            )
            for c in repeated_reference_game.context
        ]

        counterfactual_prompt_messages, counterfactual_prompt_contexts = zip(
            *[
                self.construct_prompt_messages(
                    cg, exclude_feedback_on_last=True, random_seed=412
                )
                for cg in counterfactual_games
            ]
        )

        assert all(
            [
                counterfactual_prompt_contexts[0] == ctx
                for ctx in counterfactual_prompt_contexts[1:]
            ]
        ), "Contexts for counterfactual games should be the same."

        formatted_counterfactual_prompt_messages = [
            self.processor.apply_chat_template(cfpm)
            for cfpm in counterfactual_prompt_messages
        ]

        processed_all = self.processor(
            text=formatted_counterfactual_prompt_messages,
            images=[
                list(
                    itertools.chain.from_iterable(
                        [
                            [
                                Image.open(chunk["image_url"]["url"]).convert("RGB")
                                for chunk in m["content"]
                                if chunk["type"] == "image_url"
                            ]
                            for m in cfpm
                        ]
                    )
                )
                for cfpm in counterfactual_prompt_messages
            ],
        )

        # get identify the longest prefix that's common to the different perturbed prompts
        # since we changed only the options, the token after this prefix scores the options
        # identify the longest prefix by counting down from the end
        for i in range(processed_all.input_ids.shape[1], -1, -1):
            if (processed_all.input_ids[:, :i] == processed_all.input_ids[0, :i]).all():
                break

        option_token_idx = i
        # get the token that scores the options
        option_tokens = processed_all.input_ids[:, option_token_idx]

        scores = dict()
        processed_inputs = self.processor(
            text=[formatted_counterfactual_prompt_messages[0]],
            images=[
                list(
                    itertools.chain.from_iterable(
                        [
                            [
                                Image.open(chunk["image_url"]["url"]).convert("RGB")
                                for chunk in m["content"]
                                if chunk["type"] == "image_url"
                            ]
                            for m in counterfactual_prompt_messages[0]
                        ]
                    )
                )
            ],
        )

        outputs = self.model(
            **processed_inputs.to(self.model.device, self.model.dtype),
            use_cache=False,
        )

        probs = torch.nn.functional.log_softmax(
            outputs.logits[:, option_token_idx - 1, option_tokens], dim=-1
        )[0].tolist()

        return {
            r: p
            for r, p in zip(
                [g.trials[-1].selection for g in counterfactual_games], probs
            )
        }

    def select(self, repeated_reference_game):
        probs = self.score(repeated_reference_game)
        return max(probs.items(), key=lambda x: x[1])[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class ScoringSpeaker(ChatSpeaker):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="once",
        feedback_label=False,
        prompt_type="standard",
    ):
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.prompt_type = prompt_type

        self.text_only_assistant = False

    @torch.no_grad()
    def score(self, repeated_reference_game):
        messages = self.construct_prompt_messages(
            repeated_reference_game, exclude_feedback_on_last=True
        )

        context = messages[:-1]
        formatted_context = self.processor.apply_chat_template(context)
        processed_context = self.processor(
            text=formatted_context,
            images=list(
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
            ),
            return_tensors="pt",
        )

        formatted_messages = self.processor.apply_chat_template(messages)
        processed = self.processor(
            text=formatted_messages,
            images=list(
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
            ),
            return_tensors="pt",
        )

        labels = processed.input_ids.clone()
        labels[:, : processed_context.input_ids.shape[1]] = -100

        outputs = self.model(
            **processed.to(self.model.device, self.model.dtype),
            labels=labels,
            use_cache=False,
        )

        log_p = -outputs.loss.item() * (
            processed.input_ids.shape[1] - processed_context.input_ids.shape[1]
        )
        return log_p

    def generate(self, repeated_reference_game):
        raise NotImplementedError(
            "ScoringSpeaker can only be used for scoring sequences as a speaker agent"
        )

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class JointInferenceListener(ChatListener):
    def __init__(
        self,
        listener_model,
        listener_processor,
        speaker_model,
        speaker_processor,
        listener_lambda=0.5,
        image_base_path: str = "",
        listener_context_presentation="block_shuffle",
        speaker_context_presentation="once",
        listener_feedback_label=False,
        speaker_feedback_label=False,
        speaker_prompt_type="standard",
    ):
        self.listener = ScoringListener(
            listener_model,
            listener_processor,
            image_base_path,
            listener_context_presentation,
            listener_feedback_label,
        )
        self.speaker = ScoringSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_prompt_type,
        )

        self.listener_lambda = listener_lambda

    def score(self, repeated_reference_game):
        listener_outputs = self.listener.score(repeated_reference_game)
        speaker_outputs = {
            referent: self.speaker.score(
                RepeatedReferenceGame(
                    context=repeated_reference_game.context,
                    trials=[
                        *repeated_reference_game.trials[:-1],
                        Trial(
                            target=referent,
                            message=repeated_reference_game.trials[-1].message,
                            selection=referent,
                            correct=True,
                        ),
                    ],
                )
            )
            for referent in repeated_reference_game.context
        }

        speaker_logprobs = torch.tensor(
            [speaker_outputs[r] for r in repeated_reference_game.context]
        )
        listener_logprobs = torch.tensor(
            [listener_outputs[r] for r in repeated_reference_game.context]
        )

        joint_logprobs_unnormalized = (
            listener_logprobs * self.listener_lambda
            + (1 - self.listener_lambda) * speaker_logprobs
        )

        joint_logprobs = (
            joint_logprobs_unnormalized - joint_logprobs_unnormalized.logsumexp(dim=-1)
        )

        return {
            r: p.item() for r, p in zip(repeated_reference_game.context, joint_logprobs)
        }

    def select(self, repeated_reference_game):
        logprobs = self.score(repeated_reference_game)
        return max(logprobs.items(), key=lambda x: x[1])[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)


class JointInferenceSpeaker(ChatSpeaker):
    def __init__(
        self,
        speaker_model,
        speaker_processor,
        listener_model,
        listener_processor,
        image_base_path: str = "",
        speaker_lambda=0.5,
        num_speaker_samples=5,
        listener_context_presentation="block_shuffle",
        speaker_context_presentation="once",
        listener_feedback_label=False,
        speaker_feedback_label=False,
        speaker_prompt_type="standard",
        speaker_generation_config=None,
    ):
        self.listener = ScoringListener(
            listener_model,
            listener_processor,
            image_base_path,
            listener_context_presentation,
            listener_feedback_label,
        )
        self.generate_speaker = GenerateSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_generation_config,
            speaker_prompt_type,
        )
        self.scoring_speaker = ScoringSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path,
            speaker_context_presentation,
            speaker_feedback_label,
            speaker_prompt_type,
        )

        self.num_speaker_samples = num_speaker_samples
        self.speaker_lambda = speaker_lambda

    def generate(self, repeated_reference_game):
        speaker_samples = list(
            set(
                self.generate_speaker.generate(repeated_reference_game)
                for _ in range(self.num_speaker_samples)
            )
        )
        speaker_logprobs = torch.tensor(
            [
                self.scoring_speaker.score(
                    RepeatedReferenceGame(
                        context=repeated_reference_game.context,
                        trials=[
                            *repeated_reference_game.trials[:-1],
                            Trial(
                                target=repeated_reference_game.trials[-1].target,
                                message=s,
                                selection=repeated_reference_game.trials[-1].target,
                                correct=True,
                            ),
                        ],
                    )
                )
                for s in speaker_samples
            ]
        )

        listener_logprobs = torch.tensor(
            [
                self.listener.score(
                    RepeatedReferenceGame(
                        context=repeated_reference_game.context,
                        trials=[
                            *repeated_reference_game.trials[:-1],
                            Trial(
                                target=repeated_reference_game.trials[-1].target,
                                message=s,
                                selection=repeated_reference_game.trials[-1].target,
                                correct=True,
                            ),
                        ],
                    )
                )[repeated_reference_game.trials[-1].target]
                for s in speaker_samples
            ]
        )

        joint_logprobs_unnormalized = (
            listener_logprobs * self.speaker_lambda
            + (1 - self.speaker_lambda) * speaker_logprobs
        )

        joint_logprobs = (
            joint_logprobs_unnormalized - joint_logprobs_unnormalized.logsumexp(dim=-1)
        )

        return speaker_samples[joint_logprobs.argmax().item()]
