from typing import Tuple, List
from io import BytesIO
import base64
import os
from PIL import Image

from .base_agent import BaseSpeaker, BaseListener
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
)
from pathlib import Path
from .game import RepeatedReferenceGame, Trial
import openai
import random


class OpenAIAPIAgent:
    def __init__(self, model, base_url, image_base_path: str = ""):
        self.model = model
        self.client = openai.OpenAI(
            base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "")
        )

        self.text_only_assistant = False
        self.image_base_path = os.getenv("IMAGE_BASE_PATH", "")

    def api_call(self, messages, num_return_sequences=1):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.generation_config,
            n=num_return_sequences,
        )

    def encode_image(self, image_path):
        with open(Path(self.image_base_path) / image_path, "rb") as image_file:
            return f"data:image/{self.image_format};base64,{base64.b64encode(image_file.read()).decode("utf-8")}"


class OpenAIAPISpeaker(OpenAIAPIAgent, BaseSpeaker):
    def __init__(
        self,
        model,
        base_url,
        context_presentation="once",
        feedback_label=False,
        generation_config=None,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_STANDARD,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS,
        image_format="jpg",
    ):
        OpenAIAPIAgent.__init__(
            self,
            model=model,
            base_url=base_url,
        )

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.system_prompt_template = system_prompt_template
        self.user_prompt = user_prompt
        self.image_format = image_format

        if generation_config is None:
            self.generation_config = {
                "max_tokens": 64,
                "temperature": 0.3,
            }
        else:
            self.generation_config = generation_config

    def generate(
        self, repeated_reference_games, num_return_sequences=1, target_lengths=None
    ):
        responses = list()
        for g in repeated_reference_games:
            messages, _ = self.construct_prompt_messages(g)
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.generation_config,
                n=num_return_sequences,
            )
            responses.append(
                [x.message.content.strip().strip('"') for x in outputs.choices]
            )
        return responses

    def batch_generate(
        self, repeated_reference_games, num_return_sequences=1, target_lengths=None
    ):
        # This method is here just for compatibility
        return self.generate(
            repeated_reference_games, num_return_sequences, target_lengths
        )


class OpenAIAPIListener(OpenAIAPIAgent, BaseListener):
    def __init__(
        self,
        model,
        base_url=None,
        context_presentation="once",
        feedback_label=False,
        generation_config=None,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_STANDARD,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS,
        image_format="jpg",
    ):
        OpenAIAPIAgent.__init__(
            self,
            model=model,
            base_url=base_url,
        )

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label
        self.system_prompt_template = system_prompt_template
        self.user_prompt = user_prompt
        self.image_format = image_format

        if generation_config is None:
            self.generation_config = {
                "max_tokens": 8,
                "temperature": 0,
            }
        else:
            self.generation_config = generation_config

    def validate_response(self, response, context):
        if not isinstance(response, str):
            return None

        selection_idx = ord(response[0].upper()) - ord("A")

        if selection_idx < 0 or selection_idx >= len(context):
            return None

        return {x: 0 if x == context[selection_idx] else -float("inf") for x in context}

    def score(self, repeated_reference_games, return_logits=False):
        responses = list()
        for g in repeated_reference_games:
            if g.trials[-1].message is None:
                responses.append(random.choice(g.context))
            else:
                image2path = {self.encode_image(x): x for x in g.context}
                messages, images = self.construct_prompt_messages(g)
                response = self.api_call(messages)
                responses.append(
                    self.validate_response(
                        response.choices[0].message.content,
                        [image2path[x] for x in images[-len(g.context) :]],
                    )
                )
        return responses

    def batch_score(self, repeated_reference_games, return_logits=False):
        # This method is here just for compatibility
        return self.score(repeated_reference_games, return_logits)
