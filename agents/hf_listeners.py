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


class ScoringListener(ChatListener):
    def __init__(
        self,
        model,
        processor,
        image_base_path: str = "",
        context_presentation="once",
        feedback_label=False,
        max_image_size=None,
        chat_template_file=None,
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
        if chat_template_file:
            with open(chat_template_file, "r") as f:
                self.chat_template = f.read()
        else:
            self.chat_template = None

    @find_executable_batch_size(starting_batch_size=16)
    def batch_score(batch_size, self, repeated_reference_games, return_logits=False):
        return list(
            itertools.chain.from_iterable(
                [
                    self.score(batch, return_logits=return_logits)
                    for batch in itertools.batched(repeated_reference_games, batch_size)
                ]
            )
        )

    @torch.no_grad()
    def score(self, repeated_reference_games, return_logits=False):
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "right"
        model_input_texts = list()
        model_input_images = list()
        batch_option_tokens = list()
        batch_options = list()
        for repeated_reference_game in repeated_reference_games:
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
                            correct=True,
                        ),
                    ],
                )
                for c in repeated_reference_game.context
            ]

            counterfactual_prompt_messages, counterfactual_prompt_images = zip(
                *[
                    self.construct_prompt_messages(
                        cg, exclude_feedback_on_last=True, random_seed=412
                    )
                    for cg in counterfactual_games
                ]
            )

            assert all(
                [
                    counterfactual_prompt_images[0] == ctx
                    for ctx in counterfactual_prompt_images[1:]
                ]
            ), "Contexts for counterfactual games should be the same."

            counterfactual_prompt_images = [
                [Image.open(img) for img in imgs]
                for imgs in counterfactual_prompt_images
            ]

            formatted_counterfactual_prompt_messages = [
                self.processor.apply_chat_template(
                    cfpm, chat_template=self.chat_template
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
                self.processor.apply_chat_template(
                    cfpm, chat_template=self.chat_template
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

        logits = torch.stack(
            [
                outputs.logits[i, option_token_idx - 1, option_tokens]
                for i, (option_token_idx, option_tokens) in enumerate(
                    batch_option_tokens
                )
            ]
        )

        self.processor.tokenizer.padding_side = original_padding_side
        if return_logits:
            return [
                {r: p for r, p in zip(opts, lgt)}
                for opts, lgt in zip(batch_options, logits.tolist())
            ]
        else:
            probs = torch.nn.functional.log_softmax(
                logits,
                dim=-1,
            ).tolist()

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
