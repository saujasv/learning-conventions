import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessor
from game import RepeatedReferenceGame, Trial
from PIL import Image


class ContrastiveDecodingProcessor(LogitsProcessor):
    def __init__(self, games, speaker, alpha=0.0, amateur_temp=0.5):
        self.alpha = alpha
        self.amateur_temp = amateur_temp
        self.speaker = speaker
        no_context_messages, no_context_images = zip(
            *[
                speaker.construct_prompt_messages(
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[Trial(target=game.trials[-1].target)],
                        random_seed=412,
                    )
                )
                for game in games
            ]
        )

        full_messages, full_images = zip(
            *[
                speaker.construct_prompt_messages(
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=game.trials,
                        random_seed=412,
                    )
                )
                for game in games
            ]
        )

        formatted_messages = speaker.processor.apply_chat_template(
            no_context_messages,
            add_generation_prompt=True,
            chat_template=speaker.chat_template,
        )
        self.no_context_inputs = speaker.processor(
            text=formatted_messages,
            images=[
                [Image.open(img).convert("RGB") for img in x] for x in no_context_images
            ],
            return_tensors="pt",
            padding=True,
        )

        formatted_messages = speaker.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            chat_template=speaker.chat_template,
        )
        self.full_inputs = speaker.processor(
            text=formatted_messages,
            images=[[Image.open(img).convert("RGB") for img in x] for x in full_images],
            return_tensors="pt",
            padding=True,
        )

    def __call__(self, input_ids, scores):
        num_return_sequences = input_ids.shape[0] // self.full_inputs.input_ids.shape[0]

        new_tokens = input_ids[:, self.full_inputs.input_ids.shape[1] :]

        input_ids = torch.cat(
            [
                self.no_context_inputs.input_ids.unsqueeze(1)
                .expand(-1, num_return_sequences, -1)
                .flatten(0, 1)
                .to(input_ids.device),
                new_tokens,
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                self.no_context_inputs.attention_mask.unsqueeze(1)
                .expand(-1, num_return_sequences, -1)
                .flatten(0, 1)
                .to(input_ids.device),
                torch.ones_like(new_tokens, device=input_ids.device),
            ],
            dim=1,
        )
        token_type_ids = torch.cat(
            [
                self.no_context_inputs.token_type_ids.unsqueeze(1)
                .expand(-1, num_return_sequences, -1)
                .flatten(0, 1)
                .to(input_ids.device),
                torch.zeros_like(new_tokens, device=input_ids.device),
            ],
            dim=1,
        )

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = self.speaker.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            pixel_values=self.no_context_inputs.pixel_values.unsqueeze(1)
            .expand(-1, num_return_sequences, -1, -1, -1)
            .flatten(0, 1)
            .to(self.speaker.model.device),
        )

        p_exp = torch.nn.functional.softmax(scores, dim=-1)
        V_head = torch.ge(p_exp, self.alpha * p_exp.max(axis=1).values.unsqueeze(1))
        cd_score = torch.nn.functional.log_softmax(
            scores, dim=-1
        ) - torch.nn.functional.log_softmax(
            outputs.logits[:, -1, :] / self.amateur_temp, dim=-1
        )

        cd_score.masked_fill_(~V_head, float("-inf"))

        return cd_score


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
