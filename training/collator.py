from typing import Any, Union
from transformers import (
    PixtralProcessor,
    Idefics3Processor,
    DataCollatorForLanguageModeling,
)
from transformers.models.pixtral.processing_pixtral import BatchMixFeature
from transformers.feature_extraction_utils import BatchFeature
from game import RepeatedReferenceGame, Trial
import itertools
from PIL import Image
import numpy as np


def get_chat_template_features(processor):
    if isinstance(processor, PixtralProcessor):
        return "[/INST]", "[INST]"
    elif isinstance(processor, Idefics3Processor):
        return "Assistant:", "User:"
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")


class RepeatedReferenceGameCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        agent,
        mask_only_last=True,
        ignore_index=-100,
        padding_free=False,
        padding_side="left",
    ):
        super().__init__(
            tokenizer=agent.processor.tokenizer,
            mlm=False,
        )
        self.response_template, self.instruction_template = get_chat_template_features(
            agent.processor
        )

        if isinstance(self.instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        if isinstance(self.response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        self.ignore_index = ignore_index
        self.padding_free = padding_free
        self.agent = agent
        self.mask_only_last = mask_only_last
        self.padding_side = padding_side

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        # import ipdb

        # ipdb.set_trace()
        # original_padding_side = self.tokenizer.padding_side
        # self.tokenizer.padding_side = self.padding_side
        # batch = super().torch_call(
        #     [{k: v for k, v in x.items() if k != "pixel_values"} for x in examples]
        # )
        # self.tokenizer.padding_side = original_padding_side
        # batch = super().torch_call(examples)

        labels = examples.input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch = BatchFeature(data={"labels": labels, **examples})

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            mask_out_spans = list()
            for i in range(batch.input_ids.shape[0]):
                example_mask_out_spans = list()
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(
                    batch["labels"][i] == self.response_token_ids[0]
                )[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            assistant_idx : assistant_idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_idxs.append(
                            assistant_idx + len(self.response_token_ids)
                        )

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                        UserWarning,
                    )
                    # batch["labels"][i, :] = self.ignore_index
                    example_mask_out_spans.append((0, None))

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if (
                        human_token_ids
                        == batch["labels"][i][
                            human_idx : human_idx + len(human_token_ids)
                        ].tolist()
                    ):
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_seq_length`.",
                        UserWarning,
                    )
                    # batch["labels"][i, :] = self.ignore_index
                    example_mask_out_spans.append((0, None))

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(
                    zip(human_token_ids_idxs, response_token_ids_idxs)
                ):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        # batch["labels"][i, start:end] = self.ignore_index
                        example_mask_out_spans.append((start, end))
                    else:
                        # batch["labels"][i, :end] = self.ignore_index
                        example_mask_out_spans.append((0, end))

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    # batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index
                    example_mask_out_spans.append((human_token_ids_idxs[-1], None))

                mask_out_spans.append(example_mask_out_spans)

        if self.mask_only_last:
            merged_mask_out_spans = []
            for i, spans in enumerate(mask_out_spans):
                *rest, last = spans
                if len(rest) > 0:
                    merged_mask_out_spans.append([(rest[0][0], rest[-1][1]), last])
                else:
                    merged_mask_out_spans.append([last])

            mask_out_spans = merged_mask_out_spans

        for i, spans in enumerate(mask_out_spans):
            for start, end in spans:
                batch["labels"][i, start:end] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = (
                attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            )
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch
        # return BatchMixFeature(
        #     data={**batch, "pixel_values": [x["pixel_values"] for x in examples]}
        # )

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
                            Image.open(f'{chunk["image_url"]["url"]}').convert("RGB")
                            for chunk in m["content"]
                            if chunk["type"] == "image_url"
                        ]
                        for m in messages
                    ]
                )
            )
            for messages, _ in game_messages
        ]

        # processed = [
        #     self.agent.processor(text=text, images=images, return_tensors="pt")
        #     for text, images in zip(message_texts, message_images)
        # ]

        # collated_batch = self.torch_call(
        #     [{k: v for k, v in p.items()} for p in processed]
        # )

        processed = self.agent.processor(
            text=message_texts, images=message_images, return_tensors="pt", padding=True
        )

        collated_batch = self.torch_call(processed)

        return collated_batch


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from datasets import load_dataset
    from agents import GenerateSpeaker
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import torch

    processor = AutoProcessor.from_pretrained("saujasv/Idefics3-8B-Llama3")
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(
        "saujasv/Idefics3-8B-Llama3",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    agent = GenerateSpeaker(
        None,
        processor,
        image_base_path="square-black-imgs/",
        context_presentation="last_shuffle",
    )

    dataset = load_dataset(
        "json",
        data_files="/data/tir/projects/tir3/users/svadugur/tangrams/base_model_training_games/n=100_incremental_num_referents=5_num_trials=25_min_parts=0_max_parts=0.jsonl",
    )

    collator = RepeatedReferenceGameCollator(agent, mask_only_last=False)
    dataloader = DataLoader(
        dataset["train"], batch_size=1, collate_fn=collator, shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(**batch.to(model.device, model.dtype))

    import ipdb

    ipdb.set_trace()
