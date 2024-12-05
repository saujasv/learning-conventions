from transformers import PixtralProcessor, Idefics3Processor
from trl import DataCollatorForCompletionOnlyLM
from game import RepeatedReferenceGame, Trial
import itertools
from PIL import Image


def get_chat_template_features(processor):
    if isinstance(processor, PixtralProcessor):
        return {
            "response_template": "[/INST]",
            "instruction_template": "[INST]",
        }
    elif isinstance(processor, Idefics3Processor):
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")


class RepeatedReferenceGameCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, agent):
        super().__init__(
            **get_chat_template_features(agent.processor),
            tokenizer=agent.processor.tokenizer,
            mlm=False,
        )
        self.agent = agent

    def __call__(self, batch):
        games = [RepeatedReferenceGame.model_validate(g) for g in batch]
        game_messages = [self.agent.construct_prompt_messages(game) for game in games]
        message_texts = [
            self.agent.processor.apply_chat_template(m) for m, _ in game_messages
        ]
        message_images = [
            list(
                itertools.chain.from_iterable(
                    [
                        [
                            Image.open(f'{chunk["image_url"]["url"]}.png').convert(
                                "RGB"
                            )
                            for chunk in m["content"]
                            if chunk["type"] == "image_url"
                        ]
                        for m in messages
                    ]
                )
            )
            for messages, _ in game_messages
        ]

        processed = [
            self.agent.processor(text=text, images=images, return_tensors="pt")
            for text, images in zip(message_texts, message_images)
        ]

        collated_batch = super().torch_call(
            [{k: v[0] for k, v in p.items()} for p in processed]
        )

        return collated_batch
