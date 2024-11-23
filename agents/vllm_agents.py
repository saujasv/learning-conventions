from typing import Tuple, List
from io import BytesIO
import base64
from PIL import Image
from vllm import LLM, SamplingParams
from .chat_listener import ChatListener
from .chat_speaker import ChatSpeaker


class vLLMAgent:
    def __init__(self, model, tensor_parallel_size=1, image_base_path: str = ""):
        if isinstance(model, str):
            self.llm = LLM(
                model=model,
                tokenizer_mode="mistral",  # NOTE: This is hardcoded to pixtral for now
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
