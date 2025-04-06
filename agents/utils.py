import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessor


class FIRELogitsWarper(LogitsProcessor):
    def __init__(
        self,
        num_return_sequences,
        fire_temperature: float = 2.0,
        standard_temperature: float = 0.3,
    ):
        self.fire = torch.ones((num_return_sequences,), dtype=torch.bool)
        self.fire_temperature = fire_temperature
        self.standard_temperature = standard_temperature

    def __call__(self, input_ids, logits):
        non_inf_mask = ~torch.isinf(logits)
        num_non_inf = non_inf_mask.sum(dim=-1)
        output = torch.where(
            self.fire.to(logits.device).unsqueeze(1),
            logits / self.fire_temperature,
            logits / self.standard_temperature,
        )

        self.fire = self.fire & ~(num_non_inf.to(self.fire.device) > 0)

        return output


class TemperatureDecayLogitsWarper(LogitsProcessor):
    def __init__(self, scale: float = 2.0, target_temperature: float = 0.3):
        self.scale = scale
        self.target_temperature = target_temperature
        self.idx = 0

    def __call__(self, input_ids, logits):
        temp = self.scale * np.exp(-self.idx) + self.target_temperature
        self.idx += 1
        return logits / temp


CHAT_TEMPLATE = '{% if messages[0]["role"] == "system" %}\n    {% set system_message = messages[0]["content"][0]["text"] %}\n    {% set loop_messages = messages[1:] %}\n{% else %}\n    {% set loop_messages = messages %}\n{% endif %}\n\n{{ bos_token }}\n{% for message in loop_messages %}\n    {% if (message[\'role\'] == \'user\') != (loop.index0 % 2 == 0) %}\n        {{ raise_exception(\'After the optional system message, conversation roles must alternate user/assistant/user/assistant/...\') }}\n    {% endif %}\n    {% if message["role"] == "user" %}\n        {% if loop.first and system_message is defined %}\n            {{ "[INST]" + system_message + "\\n\\n" }}\n        {% else %}\n            {{ "[INST]" }}\n        {% endif %}\n        {% if message["content"] is not string %}\n            {% for chunk in message["content"] %}\n                {% if chunk["type"] == "text" %}\n                    {{ chunk["text"] }}\n                {% elif chunk["type"] == "image_url" %}\n                    {{ "[IMG]" }}\n                {% else %}\n                    {{ raise_exception("Unrecognized content type!") }}\n                {% endif %}\n            {% endfor %}\n        {% else %}\n            {{ message["content"] }}\n        {% endif %}\n        {{ "[/INST]" }}\n    {% elif message["role"] == "assistant" %}\n        {% if message["content"] is not string %}\n            {% for chunk in message["content"] %}\n                {% if chunk["type"] == "text" %}\n                    {{ chunk["text"] }}\n                {% else %}\n                    {{ raise_exception("Unrecognized content type!") }}\n                {% endif %}\n            {% endfor %}\n        {% else %}\n            {{ message["content"] }}\n        {% endif %}\n    {% else %}\n        {{ raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {% endif %}\n{% endfor %}'

QWEN_CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
