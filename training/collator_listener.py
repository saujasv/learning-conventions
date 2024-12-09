import itertools
from PIL import Image
from transformers import DataCollatorForCompletionOnlyLM

class CustomCollator:
    def __init__(self, model, processor):
        """
        Initialize the custom collator.

        Args:
            model: The model used for scoring.
            processor: The processor for text and image processing.
        """
        self.model = model
        self.processor = processor

    def score(self, repeated_reference_game):
        """
        Compute scores for a repeated reference game by constructing counterfactual games.

        Args:
            repeated_reference_game: An instance of a repeated reference game.

        Returns:
            A collator with the appropriate scoring data.
        """
        # Construct counterfactual games
        counterfactual_games = [
            RepeatedReferenceGame(
                context=repeated_reference_game.context,
                trials=[
                    *repeated_reference_game.trials[:-1],
                    Trial(
                        target=repeated_reference_game.trials[-1].target,
                        message=repeated_reference_game.trials[-1].message,
                        selection=c,
                        correct=c == repeated_reference_game.trials[-1].target,
                    ),
                ],
            )
            for c in repeated_reference_game.context
        ]

        # Create prompt messages and contexts
        counterfactual_prompt_messages, counterfactual_prompt_contexts = zip(
            *[
                self.construct_prompt_messages(
                    cg, exclude_feedback_on_last=True, random_seed=412
                )
                for cg in counterfactual_games
            ]
        )

        # Ensure contexts are consistent
        assert all(
            [
                counterfactual_prompt_contexts[0] == ctx
                for ctx in counterfactual_prompt_contexts[1:]
            ]
        ), "Contexts for counterfactual games should be the same."

        # Format prompt messages
        formatted_counterfactual_prompt_messages = [
            self.processor.apply_chat_template(cfpm)
            for cfpm in counterfactual_prompt_messages
        ]

        # Process all inputs
        processed_all = self.processor(
            text=formatted_counterfactual_prompt_messages,
            images=[
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
            ],
        )

        # Identify the token scoring the options
        for i in range(processed_all.input_ids.shape[1] - 1, -1, -1):
            if (processed_all.input_ids[:, :i] == processed_all.input_ids[0, :i]).all():
                break

        option_token_idx = i
        option_tokens = processed_all.input_ids[:, option_token_idx]

        # Process a single input for reference
        processed_inputs = self.processor(
            text=[formatted_counterfactual_prompt_messages[0]],
            images=[
                list(
                    itertools.chain.from_iterable(
                        [
                            [
                                Image.open(chunk["image_url"]["url"]).convert("RGB")
                                for chunk in m["content"]
                                if chunk["type"] == "image_url"
                            ]
                            for m in counterfactual_prompt_messages[0]
                        ]
                    )
                )
            ],
        )

        # Generate model outputs
        outputs = self.model(
            **processed_inputs.to(self.model.device, self.model.dtype),
            use_cache=False,
        )

        # Return a Hugging Face collator
        return DataCollatorForCompletionOnlyLM(
            response_template=outputs.logits,
            instruction_template=processed_inputs,
            tokenizer=self.processor.tokenizer,
        )

    def __call__(self, batch):
        """
        Collate function to process a batch of repeated reference games.

        Args:
            batch: A list of repeated reference games.

        Returns:
            Collated outputs.
        """
        collated_outputs = [self.score(repeated_reference_game) for repeated_reference_game in batch]
        return collated_outputs



# collator = CustomCollator(model=model, processor=processor)
# data_loader = DataLoader(dataset, batch_size=8, collate_fn=collator)