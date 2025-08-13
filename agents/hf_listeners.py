import torch
import os
from PIL import Image
import random
from accelerate import find_executable_batch_size
import itertools
from pathlib import Path
import numpy as np
from .game import RepeatedReferenceGame, Trial
from .base_agent import BaseListener
from transformers import PreTrainedModel, ProcessorMixin
from typing import Any, Literal, Optional, Union


class ScoringListener(BaseListener):
    def __init__(
        self,
        model_type: Literal["base", "chat"],
        model: PreTrainedModel,
        processor: ProcessorMixin,
        context_presentation: Literal[
            "once", "last_shuffle", "last_no_shuffle"
        ] = "once",
        feedback_label: bool = True,
        max_image_size: Optional[int] = None,
        chat_template_file: Optional[str] = None,
        demonstration_game: Optional[Union[RepeatedReferenceGame, str]] = None,
        ensemble: int = 1,
    ):
        BaseListener.__init__(
            self,
            model_type=model_type,
            context_presentation=context_presentation,
            feedback_label=feedback_label,
            demonstration_game=demonstration_game,
        )
        self.model = model
        self.processor = processor
        self.image_base_path = os.getenv("IMAGE_BASE_PATH", "")
        self.text_only_assistant = False
        self.max_image_size = max_image_size
        self.ensemble = ensemble
        self.chat_template = None
        if chat_template_file:
            with open(chat_template_file, "r") as f:
                self.chat_template = f.read()

    @find_executable_batch_size(starting_batch_size=4)
    def batch_score(batch_size, self, repeated_reference_games, return_logits=False):
        return list(
            itertools.chain.from_iterable(
                [
                    self.score(batch, return_logits=return_logits)
                    for batch in itertools.batched(repeated_reference_games, batch_size)
                ]
            )
        )

    @find_executable_batch_size(starting_batch_size=8)
    def batch_inference(
        batch_size, self, model_input_texts, model_input_images, batch_option_tokens
    ):
        """
        Batch inference method that processes texts and images and returns model outputs.

        Args:
            model_input_texts: List of text inputs
            model_input_images: List of list of image paths

        Returns:
            List of model outputs
        """
        all_outputs = []

        # Create batches and zip them together for clean iteration
        text_batches = itertools.batched(model_input_texts, batch_size)
        image_batches = itertools.batched(model_input_images, batch_size)
        option_token_batches = itertools.batched(batch_option_tokens, batch_size)

        for batch_texts, batch_images, option_tokens in zip(
            text_batches, image_batches, option_token_batches
        ):
            # Process images - convert paths to PIL Images
            processed_batch_images = [
                [Image.open(img).convert("RGB") for img in img_list]
                for img_list in batch_images
            ]

            # Process the batch
            processed_inputs = self.processor(
                text=list(batch_texts),
                images=processed_batch_images,
                return_tensors="pt",
                size=(
                    {"longest_edge": self.max_image_size}
                    if self.max_image_size
                    else None
                ),
                padding=True,
            )

            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    **processed_inputs.to(self.model.device, self.model.dtype),
                    use_cache=False,
                )

            logits = torch.stack(
                [
                    outputs.logits[i, option_token_idx - 1, opts]
                    for i, (option_token_idx, opts) in enumerate(option_tokens)
                ]
            )

            all_outputs.append(logits)

        return torch.cat(all_outputs)

    @torch.no_grad()
    def score(self, repeated_reference_games, return_logits=False, seed=None):
        model_input_texts = list()
        model_input_images = list()
        batch_option_tokens = list()
        batch_options = list()
        game_ensemble_mapping = (
            list()
        )  # Track which batch element belongs to which (game_idx, ensemble_idx)

        base_seed = seed if seed is not None else random.randint(0, 1000000)

        for game_idx, repeated_reference_game in enumerate(repeated_reference_games):
            # Ensemble loop for this game - collect inputs
            for ensemble_idx in range(self.ensemble):
                ensemble_seed = base_seed + ensemble_idx

                # construct counterfactual games by permuting the options to take each of the possible values
                counterfactual_games = [
                    RepeatedReferenceGame(
                        context=repeated_reference_game.context,
                        trials=[
                            *repeated_reference_game.trials[:-1],
                            Trial(
                                target=repeated_reference_game.trials[-1].target,
                                message=repeated_reference_game.trials[-1].message,
                                interpretation={
                                    x: (0 if x == c else -float("inf"))
                                    for x in repeated_reference_game.context
                                },
                            ),
                        ],
                    )
                    for c in repeated_reference_game.context
                ]

                counterfactual_prompt_messages, counterfactual_prompt_images = zip(
                    *[
                        self.construct_prompt_messages(
                            cg, exclude_feedback_on_last=True, random_seed=ensemble_seed
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

                formatted_counterfactual_prompt_messages = [
                    self.processor.apply_chat_template(
                        cfpm, chat_template=self.chat_template
                    )
                    for cfpm in counterfactual_prompt_messages
                ]

                processed_all = self.processor(
                    text=formatted_counterfactual_prompt_messages,
                    images=[
                        [Image.open(img).convert("RGB") for img in ctx]
                        for ctx in counterfactual_prompt_images
                    ],
                    return_tensors="pt",
                    size=(
                        {"longest_edge": self.max_image_size}
                        if self.max_image_size
                        else None
                    ),
                )

                # get identify the longest prefix that's common to the different perturbed prompts
                # since we changed only the options, the token after this prefix scores the options
                # identify the longest prefix by counting down from the end
                for i in range(processed_all.input_ids.shape[1] - 1, -1, -1):
                    if (
                        processed_all.input_ids[:, :i] == processed_all.input_ids[0, :i]
                    ).all():
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

                # Collect inputs for batch processing
                model_input_texts.append(formatted_counterfactual_prompt_messages[0])
                model_input_images.append(counterfactual_prompt_images[0])
                batch_option_tokens.append((option_token_idx, option_tokens))
                batch_options.append(
                    [g.trials[-1].get_selection() for g in counterfactual_games]
                )
                game_ensemble_mapping.append((game_idx, ensemble_idx))

        logits = self.batch_inference(
            model_input_texts, model_input_images, batch_option_tokens
        )

        # Regroup results by game and ensemble
        batch_results = []
        if return_logits:
            batch_results = [
                {r: p for r, p in zip(opts, lgt)}
                for opts, lgt in zip(batch_options, logits.tolist())
            ]
        else:
            probs = torch.nn.functional.log_softmax(
                logits,
                dim=-1,
            ).tolist()
            batch_results = [
                {r: p for r, p in zip(opts, prob)}
                for opts, prob in zip(batch_options, probs)
            ]

        # Organize results by game and ensemble
        num_games = len(repeated_reference_games)
        game_ensemble_results = [[] for _ in range(num_games)]

        for batch_idx, (game_idx, ensemble_idx) in enumerate(game_ensemble_mapping):
            game_ensemble_results[game_idx].append(batch_results[batch_idx])

        # Average ensemble results for each game
        final_results = []
        for game_idx in range(num_games):
            ensemble_results_for_game = game_ensemble_results[game_idx]

            if self.ensemble == 1:
                averaged_game_result = ensemble_results_for_game[0]
            else:
                game_options = list(ensemble_results_for_game[0].keys())
                averaged_game_result = {}
                for option in game_options:
                    option_values = [
                        np.exp(result[option]) for result in ensemble_results_for_game
                    ]
                    averaged_game_result[option] = np.log(
                        sum(option_values) / len(option_values)
                    )

            final_results.append(averaged_game_result)

        return final_results

    def select(self, repeated_reference_game):
        if repeated_reference_game.trials[-1].message is None:
            return random.choice(repeated_reference_game.context)
        probs = self.score(repeated_reference_game)
        return max(probs.items(), key=lambda x: x[1])[0]

    def encode_image(self, image_path):
        return str(Path(self.image_base_path) / image_path)
