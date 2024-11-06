from typing import Tuple, List
import time
from openai import APIError, RateLimitError, APITimeoutError, OpenAI
import os
import httpx
import json
import itertools
from io import BytesIO
import base64
from transformers import Idefics2ForConditionalGeneration, AutoProcessor
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
import random
import sys
from vllm import LLM, SamplingParams
from copy import deepcopy
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from itertools import batched

sys.path.append(os.path.abspath("./cogen"))
from cogen.models.joint_inference import IdeficsJointInferenceModel


class ChatAPIListener:
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
                                "url": f"data:image/png;base64,{self.encode_image(image)}",
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
        intro = self.get_intro(repeated_reference_game.context)
        if self.context_presentation == "no_history":
            trial_messages = [
                self.format_trial(
                    repeated_reference_game.trials[-1],
                    repeated_reference_game.context,
                    show_images=True,
                    trial_number=None,
                )
            ]
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            response = self.api_call(messages)
            return self.validate_response(response, repeated_reference_game.context)
        elif self.context_presentation == "once":
            trial_messages = [
                self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=i == 0,
                    trial_number=i + 1,
                )
                for i, trial in enumerate(repeated_reference_game.trials)
            ]
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            response = self.api_call(messages)
            return self.validate_response(response, repeated_reference_game.context)
        elif self.context_presentation == "trial_shuffle":
            trial_context = random.sample(
                repeated_reference_game.context,
                len(repeated_reference_game.context),
            )
            trial_messages = [
                self.format_trial(
                    trial,
                    trial_context,
                    show_images=True,
                    trial_number=i + 1,
                )
                for i, trial in enumerate(repeated_reference_game.trials)
            ]
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            response = self.api_call(messages)
            return self.validate_response(response, trial_context)
        elif self.context_presentation == "block_shuffle":
            if not repeated_reference_game.validate_block_structure():
                raise ValueError("Game does not have correct block structure.")

            trial_messages = list()
            trial_counter = 0
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
                    )
                    if len(messages) > 0:
                        block_trials.append(messages)
                        trial_counter += 1

                trial_messages.extend(block_trials)

            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
            response = self.api_call(messages)
            return self.validate_response(response, block_context)
        else:
            raise ValueError("Invalid context presentation type.")


class ChatAPISpeaker:
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
                                        "url": f"data:image/png;base64,{self.encode_image(image)}",
                                    },
                                },
                            ]
                            for image in context
                        ]
                    ),
                ],
            },
        ]

    def format_trial(self, context, trial, trial_number=None):
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

    def generate(self, repeated_reference_game):
        intro = self.get_intro(repeated_reference_game.context)
        if self.context_presentation == "no_history":
            trials_messages = [
                self.format_trial(
                    repeated_reference_game.context,
                    repeated_reference_game.trials[-1],
                    trial_number=None,
                )
            ]

        elif self.context_presentation == "once":
            trials_messages = itertools.chain.from_iterable(
                [
                    self.format_trial(
                        repeated_reference_game.context, trial, trial_number=i + 1
                    )
                    for i, trial in enumerate(repeated_reference_game.trials)
                ]
            )
        messages = [*intro, *trials_messages]

        return self.api_call(messages)


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
            return base64.b64encode(image_file.read()).decode("utf-8")


class PixtralAgent:
    def __init__(self, model, tensor_parallel_size=1):
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

        self.text_only_assistant = True

    def api_call(self, messages):
        outputs = self.llm.chat(
            messages=messages, sampling_params=self.generation_config
        )
        return outputs[0].outputs[0].text.strip("\"'")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class GPTSpeaker(GPTAgent, ChatAPISpeaker):
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


class GPTListener(GPTAgent, ChatAPIListener):
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


class PixtralSpeaker(PixtralAgent, ChatAPISpeaker):
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
        PixtralAgent.__init__(self, model, tensor_parallel_size)

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        if generation_config is None:
            generation_config = {
                "max_tokens": 64,
                "temperature": 0.3,
            }

        self.generation_config = SamplingParams(**generation_config)

        self.prompt_type = prompt_type


class PixtralListener(PixtralAgent, ChatAPIListener):
    def __init__(
        self,
        model,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
        generation_config=None,
        tensor_parallel_size=1,
    ):
        PixtralAgent.__init__(self, model, tensor_parallel_size)

        if generation_config is None:
            generation_config = {
                "max_tokens": 8,
                "temperature": 0,
            }

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        self.generation_config = SamplingParams(**generation_config)


class CoGenListener:
    def __init__(
        self,
        checkpoint: str = None,
        anno_len_threshold: int = 40,
        comprehension_prompt: str = "verbose_instruction",
        context_size: int = 10,
        evaluation_type: str = "joint",
        from_scratch: bool = True,
        generation_prompt: str = "information_after",
        listener_filter: str = "no_neg_gen",
        listener_lambda: float = 0.5,
        load_from_checkpoint: bool = False,
        lora_dropout: float = 0.05,
        lora_r: int = 16,
        lora_subset: str = "vision_resampler",
        model_family_name: str = "full",
        no_lora: bool = False,
        no_shuffling: bool = False,
        noise_filter: str = "",
        num_samples: int = 10,
        only_seed: bool = False,
        sampling_type: str = "nucleus",
        seed: int = 133395,
        shared_parameters: bool = True,
        speaker_filter: str = "no_neg_comp",
        speaker_lambda: float = 0.5,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 1.0,
        training_type: str = "multitask",
        weight_decay: float = 0.1,
        index_to_token_path: str = None,
    ):
        self.config = {
            "anno_len_threshold": anno_len_threshold,
            "comprehension_prompt_type": comprehension_prompt,
            "context_size": context_size,
            "evaluation_type": evaluation_type,
            "from_scratch": from_scratch,
            "generation_prompt_type": generation_prompt,
            "listener_filter": listener_filter,
            "listener_lambda": listener_lambda,
            "load_from_checkpoint": load_from_checkpoint,
            "lora_dropout": lora_dropout,
            "lora_r": lora_r,
            "lora_subset": lora_subset,
            "model_family_name": model_family_name,
            "no_lora": no_lora,
            "no_shuffling": no_shuffling,
            "noise_filter": noise_filter,
            "num_samples": num_samples,
            "only_seed": only_seed,
            "sampling_type": sampling_type,
            "seed": seed,
            "shared_parameters": shared_parameters,
            "speaker_filter": speaker_filter,
            "speaker_lambda": speaker_lambda,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "training_type": training_type,
            "weight_decay": weight_decay,
            "index_to_token_path": index_to_token_path,
        }
        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False,
            size={"longest_edge": 448, "shortest_edge": 224},
        )
        model = self.initialize_idefics()
        model.load_adapter(checkpoint, model.active_adapter)
        self.model = IdeficsJointInferenceModel(
            listener_lambda, speaker_lambda, model=model
        )

        # hardcoded values for Idefics2
        self.index_to_token = [
            28734,
            28740,
            28750,
            28770,
            28781,
            28782,
            28784,
            28787,
            28783,
            28774,
        ]

    def initialize_idefics(self):
        # Initialize the model
        checkpoint = "HuggingFaceM4/idefics2-8b"
        model = Idefics2ForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ).cuda()
        if self.config["no_lora"]:
            return model

        # Add LoRA adapters
        if self.config["lora_subset"] == "all":
            target_modules = r".*(text_model|vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        elif self.config["lora_subset"] == "theirs":  # from IDEFICS-2 tutorial notebook
            target_modules = r".*(text_model|vision_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        elif self.config["lora_subset"] == "vision_resampler":
            target_modules = r"(.*(vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)|(.*(k_proj|q_proj|v_proj).*$)"
        elif self.config["lora_subset"] == "standard":
            target_modules = r".*(k_proj|q_proj|v_proj).*$"
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=8,
            lora_dropout=self.config["lora_dropout"],
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )

        return get_peft_model(model, lora_config)

    def construct_speaker_full_prompt(self, target_anno, target_idx):
        messages = self.construct_speaker_base_prompt(target_idx)

        # Assistant response
        target_anno = target_anno.lower().strip()
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_anno}]}
        )

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        ).strip()

    def construct_speaker_base_prompt(self, target_idx, process=False):
        messages = []

        if self.config["generation_prompt_type"] == "information_after":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f". Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "information_after_strict":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your must produce a caption that describes your target image and only your target image. ",
                        },
                        {
                            "type": "text",
                            "text": "When someone reads your description, they should be able to pick your target image from the set of 10 images. ",
                        },
                        {
                            "type": "text",
                            "text": f"Your target for this turn will be Image {target_idx}.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f". Your target image is Image {target_idx}. Produce your distinctive caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "reversed_image_description":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n"}
                    )

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif (
            self.config["generation_prompt_type"]
            == "reversed_image_description_letters"
        ):
            image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == target_idx:
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Target image start: "}
                    )

                messages[0]["content"].append({"type": "image"})
                messages[0]["content"].append(
                    {"type": "text", "text": f"Image {image_labels[i]}. "}
                )

                if i == target_idx:
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Target image end\n"}
                    )
                else:
                    messages[0]["content"].append({"type": "text", "text": f"\n"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {image_labels[target_idx]}. Produce your caption now.",
                }
            )
        elif (
            self.config["generation_prompt_type"]
            == "reversed_image_description_repetition"
        ):
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                        {
                            "type": "text",
                            "text": f"Your target will be Image {target_idx}.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n"}
                    )

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "information_before":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description. ",
                        },
                        {
                            "type": "text",
                            "text": f"Your target image is Image {target_idx}. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {"type": "text", "text": f". Produce your caption now."}
            )
        else:
            print("Invalid prompt type")
            assert "False"

        if process:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            ).strip()
            return prompt
        else:
            return messages

    def select(self, context, message, previous_rounds=None):
        (
            images,
            l_input_tokens,
            l_attn_mask,
            l_image_attn_mask,
            s_input_tokens,
            s_attn_mask,
            s_image_attn_mask,
            s_target_mask,
            s_target_tokens,
        ) = self.prepare_inputs(context, message)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.no_grad():
                listener_log_probs, speaker_log_probs, joint_log_probs = (
                    self.model.forward(
                        "joint_comprehension",
                        [
                            images,
                            l_input_tokens,
                            l_attn_mask,
                            l_image_attn_mask,
                            self.index_to_token,
                            s_input_tokens,
                            s_attn_mask,
                            s_image_attn_mask,
                            s_target_mask,
                            s_target_tokens,
                        ],
                    )
                )

        return listener_log_probs.argmax().item()

    def construct_listener_full_prompt(
        self,
        target_anno,
        target_idx=0,
    ):
        target_anno = target_anno.lower().strip()
        messages = []

        if self.config["comprehension_prompt_type"] == "information_after":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f". Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {"type": "text", "text": f" Which image does this caption describe?"}
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "information_before":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                        {"type": "text", "text": f"Your caption is: {target_anno}"},
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption

            messages[0]["content"].append(
                {"type": "text", "text": f" Which image does this caption describe?"}
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "verbose_instruction":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f". Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "reversed_image_description":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will given a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n "}
                    )

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f"Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif (
            self.config["comprehension_prompt_type"]
            == "reversed_image_description_letters"
        ):
            image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will given a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                messages[0]["content"].append(
                    {"type": "text", "text": f" Image {image_labels[i]}.\n"}
                )

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f"Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image A, B, C, D, E, F, G, H, I or J?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {image_labels[target_idx]}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "no_instruction":
            # User side: Intro
            messages.append({"role": "user", "content": []})

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n "}
                    )

            # User side: Caption
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Which image does the caption '{target_anno}' describe?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        else:
            print("Invalid prompt type")
            assert "False"

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        ).strip()

    def prepare_inputs(self, context, message):
        # Generate the raw image sequence and sim idx
        raw_images = [
            Image.open(img.replace("tangram_png/tangram_png", "tangram_png"))
            for img in context
        ]
        # if not self.config["no_shuffling"]:
        #     random.shuffle(raw_images)

        # Create the prompt and inputs for the listener
        prompt = self.construct_listener_full_prompt(
            message,
        )
        outputs = self.processor(
            text=[prompt],
            images=[raw_images],
            padding=True,
            return_tensors="pt",
        )
        l_input_tokens = outputs["input_ids"][:, :-2]
        l_attn_mask = outputs["attention_mask"][:, :-2]
        l_attn_mask[(l_input_tokens == 0).bool()] = 0
        images = outputs["pixel_values"]
        l_image_attn_mask = outputs["pixel_attention_mask"]

        base_prompt_outputs = self.processor(
            text=[
                self.construct_speaker_base_prompt(i, process=True) for i in range(10)
            ],
            images=[raw_images] * 10,
            padding=False,
        )
        prompt_outputs = self.processor(
            text=[self.construct_speaker_full_prompt(message, i) for i in range(10)],
            images=[raw_images] * 10,
            padding=True,
            return_tensors="pt",
        )
        padding_tokens = (prompt_outputs.input_ids == 0).sum(dim=1).tolist()
        n_prefix_tokens = [
            len(base_tokens) + n_padding_tokens
            for base_tokens, n_padding_tokens in zip(
                base_prompt_outputs.input_ids, padding_tokens
            )
        ]

        s_input_tokens = prompt_outputs["input_ids"][:, :-1]
        s_attn_mask = prompt_outputs["attention_mask"][:, :-1]
        s_attn_mask[(s_input_tokens == 0).bool()] = 0
        s_image_attn_mask = prompt_outputs["pixel_attention_mask"]
        s_target_tokens = prompt_outputs["input_ids"][:, 1:]
        s_target_mask = torch.ones_like(s_attn_mask)
        for i, n_prefix in enumerate(n_prefix_tokens):
            s_target_mask[i, :n_prefix] = 0

        return (
            images.to("cuda"),
            l_input_tokens.to("cuda"),
            l_attn_mask.to("cuda"),
            l_image_attn_mask.to("cuda"),
            s_input_tokens.unsqueeze(0).to("cuda"),
            s_attn_mask.unsqueeze(0).to("cuda"),
            s_image_attn_mask.unsqueeze(0).to("cuda"),
            s_target_mask.unsqueeze(0).to("cuda"),
            s_target_tokens.unsqueeze(0).to("cuda"),
        )


if __name__ == "__main__":
    listener = CoGenListener(
        "cogen/data_and_checkpoints/experiments/joint_training/r3_full/run/checkpoints/acc"
    )
