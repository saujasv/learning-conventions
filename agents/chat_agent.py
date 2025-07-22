import itertools
from random import Random
from pathlib import Path


class ChatAgent:
    def __init__(
        self,
        context_presentation="once",
        feedback_label=False,
        text_only_assistant=False,
    ):
        # presentation of images and prior trials of the game in the context
        assert context_presentation in [
            "once",  # present context once at the beginning of the game
            "no_history",  # do not present any prior trials
            "last_shuffle",  # present the images once upfront and shuffle the images only for the last trial
            "last_no_shuffle",  # present the images once upfront and do not shuffle the images for the last trial
            "trial_shuffle",  # shuffle the images and present them for each trial
            "block_shuffle",  # shuffle the images and present them at the beginning of each block
        ]
        self.context_presentation = context_presentation

        # whether to include the true label in the feedback message
        self.feedback_label = feedback_label

        # assistant messages only include text (required for Mistral API)
        self.text_only_assistant = text_only_assistant

    def get_intro(self, context):
        raise NotImplementedError(
            "The introduction prompt is specific to the agent type (speaker/listener)."
        )

    def get_images(self, messages):
        """
        Extracts image paths from the messages.
        Args:
            messages: List of message dictionaries.
        Returns:
            A list of image paths.
        """
        images = list()
        for msg in messages:
            for chunk in msg["content"]:
                if chunk["type"] == "image_url":
                    images.append(chunk["image_url"]["url"])

        return images

    def construct_prompt_messages(
        self,
        repeated_reference_game,
        exclude_feedback_on_last=False,
        random_seed=None,
    ):
        """
        Constructs the prompt messages for the chat agent based on the context presentation type.
        Args:
            repeated_reference_game: The repeated reference game object.
            exclude_feedback_on_last: Whether to exclude feedback on the last trial.
            random_seed: Random seed for shuffling.
        Returns:
            A tuple containing the prompt messages and a list of paths to all images appearing in the messages.
        """
        # Create RNG object if random_seed is provided
        rng = Random(random_seed) if random_seed is not None else Random()

        intro = self.get_intro(repeated_reference_game.context)
        if self.context_presentation == "no_history":
            trial_messages = [
                self.format_trial(
                    repeated_reference_game.trials[-1],
                    repeated_reference_game.context,
                    show_images=True,
                    trial_number=None,
                    exclude_feedback=True,
                )
            ]
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
        elif self.context_presentation == "once":
            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=(
                        exclude_feedback_on_last
                        if i == len(repeated_reference_game.trials) - 1
                        else False
                    ),
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
        elif self.context_presentation == "last_shuffle":
            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_trial = repeated_reference_game.trials[-1]
            last_trial_context = rng.sample(
                repeated_reference_game.context, len(repeated_reference_game.context)
            )
            last_trial_messages = self.format_trial(
                last_trial,
                last_trial_context,
                show_images=True,
                trial_number=len(repeated_reference_game.trials),
                exclude_feedback=exclude_feedback_on_last,
            )
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]
        elif self.context_presentation == "last_no_shuffle":
            trial_messages = list()
            show_images = True
            for i, trial in enumerate(repeated_reference_game.trials[:-1]):
                messages = self.format_trial(
                    trial,
                    repeated_reference_game.context,
                    show_images=show_images,
                    trial_number=i + 1,
                    exclude_feedback=False,
                )
                if len(messages) > 0:
                    trial_messages.append(messages)
                    show_images = False
                else:
                    continue

            last_trial = repeated_reference_game.trials[-1]
            last_trial_context = repeated_reference_game.context
            last_trial_messages = self.format_trial(
                last_trial,
                last_trial_context,
                show_images=True,
                trial_number=len(repeated_reference_game.trials),
                exclude_feedback=exclude_feedback_on_last,
            )
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
                *last_trial_messages,
            ]
        elif self.context_presentation == "trial_shuffle":
            trial_messages = []
            for i, trial in enumerate(repeated_reference_game.trials):
                trial_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )
                trial_messages.append(
                    self.format_trial(
                        trial,
                        trial_context,
                        show_images=True,
                        trial_number=i + 1,
                        exclude_feedback=(
                            exclude_feedback_on_last
                            if i == len(repeated_reference_game.trials) - 1
                            else False
                        ),
                    )
                )
            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]
        elif self.context_presentation == "block_shuffle":
            if not repeated_reference_game.validate_block_structure():
                raise ValueError("Game does not have correct block structure.")

            trial_messages = list()
            trial_counter = 0
            block_context = None

            if len(repeated_reference_game.trials) == 0:
                block_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )

            for block_idx, block in enumerate(
                itertools.batched(
                    repeated_reference_game.trials, len(repeated_reference_game.context)
                )
            ):
                block_context = rng.sample(
                    repeated_reference_game.context,
                    len(repeated_reference_game.context),
                )
                block_trials = list()
                block_start_counter = trial_counter
                for trial in block:
                    messages = self.format_trial(
                        trial,
                        block_context,
                        show_images=trial_counter == block_start_counter,
                        trial_number=trial_counter + 1,
                        exclude_feedback=(
                            exclude_feedback_on_last
                            if trial_counter == len(repeated_reference_game.trials) - 1
                            else False
                        ),
                    )
                    if len(messages) > 0:
                        block_trials.append(messages)
                        trial_counter += 1

                trial_messages.extend(block_trials)

            messages = [
                *intro,
                *itertools.chain.from_iterable(trial_messages),
            ]

            assert block_context is not None, "Block context should not be None."
        else:
            raise ValueError("Invalid context presentation type.")

        collapsed_messages = self.collapse_turns(messages)
        return collapsed_messages, self.get_images(collapsed_messages)

    def collapse_turns(self, messages):
        collapsed_messages = list()
        for m in messages:
            if (
                len(collapsed_messages) > 0
                and collapsed_messages[-1]["role"] == m["role"]
            ):
                collapsed_messages[-1]["content"] += m["content"]
            else:
                collapsed_messages.append(m)

        return collapsed_messages

    def encode_image(self, image_path):
        return str(image_path)
