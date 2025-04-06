from typing import Tuple, List
import torch
from PIL import Image
import random
import sys
from copy import deepcopy
import itertools
from pathlib import Path
from game import RepeatedReferenceGame, Trial
from .chat_listener import ChatListener


class GenerateListener(ChatListener):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
        max_image_size=None,
    ):
        ChatListener.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
        )
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
        self.text_only_assistant = False
        self.max_image_size = max_image_size

    def select(self, repeated_reference_game):
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)

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
            size={"longest_edge": self.max_image_size} if self.max_image_size else None,
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


class ScoringListener(ChatListener):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="block_shuffle",
        feedback_label=False,
    ):
        ChatListener.__init__(
            self,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
        )
        self.model = model
        self.processor = processor
        self.image_base_path = image_base_path
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
                        # correct=c == repeated_reference_game.trials[-1].target,
                        correct=True,
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
        counterfactual_prompt_images = [
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
        ]

        processed_all = self.processor(
            text=formatted_counterfactual_prompt_messages,
            images=counterfactual_prompt_images,
            return_tensors="pt",
            size={"longest_edge": self.max_image_size} if self.max_image_size else None,
        )

        # get identify the longest prefix that's common to the different perturbed prompts
        # since we changed only the options, the token after this prefix scores the options
        # identify the longest prefix by counting down from the end
        for i in range(processed_all.input_ids.shape[1] - 1, -1, -1):
            if (processed_all.input_ids[:, :i] == processed_all.input_ids[0, :i]).all():
                break

        option_token_idx = i
        # get the token that scores the options
        option_tokens = processed_all.input_ids[:, option_token_idx]

        # doing this again because the formatted messages are passed by reference
        # and the image tokens get expanded in the first call to self.processor
        formatted_counterfactual_prompt_messages = [
            self.processor.apply_chat_template(cfpm)
            for cfpm in counterfactual_prompt_messages
        ]
        counterfactual_prompt_images = [
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
        ]

        processed_inputs = self.processor(
            text=[formatted_counterfactual_prompt_messages[0]],
            images=[counterfactual_prompt_images[0]],
            return_tensors="pt",
            size={"longest_edge": self.max_image_size} if self.max_image_size else None,
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
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)
        probs = self.score(repeated_reference_game)
        return max(probs.items(), key=lambda x: x[1])[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
