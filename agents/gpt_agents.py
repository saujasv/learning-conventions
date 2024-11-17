from typing import Tuple, List
import time
from openai import APIError, RateLimitError, APITimeoutError, OpenAI
import os
import httpx
import json
import itertools
from io import BytesIO
import base64
from PIL import Image
import random
from pathlib import Path
from .chat_listener import ChatListener
from .chat_speaker import ChatSpeaker


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
