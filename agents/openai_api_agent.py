from typing import Tuple, List
from io import BytesIO
import base64
from PIL import Image

from .chat_speaker import ChatSpeaker
from .prompts import (
    SPEAKER_SYSTEM_PROMPT_STANDARD,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS,
)
from pathlib import Path
from game import RepeatedReferenceGame, Trial
import openai


class OpenAIAPIAgent:
    def __init__(self, model, base_url, api_key, image_base_path: str = ""):
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

        self.text_only_assistant = False
        self.image_base_path = image_base_path

    def api_call(self, messages):
        outputs = self.llm.chat(
            messages=messages, sampling_params=self.generation_config
        )
        return outputs[0].outputs[0].text.strip("\"'")

    def encode_image(self, image_path):
        with open(Path(self.image_base_path) / image_path, "rb") as image_file:
            return f"data:image/{self.image_format};base64,{base64.b64encode(image_file.read()).decode("utf-8")}"


class OpenAIAPISpeaker(OpenAIAPIAgent, ChatSpeaker):
    def __init__(
        self,
        model,
        base_url,
        api_key,
        image_base_path,
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
            api_key=api_key,
            image_base_path=image_base_path,
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

    def generate(self, repeated_reference_game, num_return_sequences=1):
        messages, _ = self.construct_prompt_messages(repeated_reference_game)
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.generation_config,
            n=num_return_sequences,
        )
        return [x.message.content.strip().strip('"') for x in outputs.choices]
