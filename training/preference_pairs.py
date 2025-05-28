from typing import List, Optional, Tuple
import uuid
from copy import deepcopy
import numpy as np
from game import RepeatedReferenceGame, Trial
from agents.chat_speaker import ChatSpeaker
from agents.chat_listener import ChatListener
from agents.hf_speakers import GenerateSpeaker
from agents.hf_listeners import ScoringListener
from agents.prompts import (
    SPEAKER_SYSTEM_PROMPT_BASIC,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC,
    SPEAKER_USER_PROMPT_TARGET_BASIC,
)
from training.simulation_utils import (
    sequence_targets,
    sequence_targets_blocks,
    informativity_and_cost_preference,
    informativity_margin_and_cost_preference,
    correctness_and_cost_preference,
    informativity_preference,
    length_change_preference,
    wnr_change_preference,
)
from tqdm import tqdm

# Map preference criterion names to functions
PREFERENCE_FUNCTIONS = {
    "informativity_and_cost_preference": informativity_and_cost_preference,
    "informativity_preference": informativity_preference,
    "length_change_preference": length_change_preference,
    "wnr_change_preference": wnr_change_preference,
    "informativity_margin_and_cost_preference": informativity_margin_and_cost_preference,
    "correctness_and_cost_preference": correctness_and_cost_preference,
}

TARGET_SEQUENCE_FUNCTIONS = {
    "sequence_targets": sequence_targets,
    "sequence_targets_blocks": sequence_targets_blocks,
}


def make_preference_pairs(
    game: RepeatedReferenceGame,
    sampled_trials: List[Trial],
    preference_criterion: callable,
) -> List[Tuple[Trial, Trial]]:
    """
    Make preference pairs from sampled trials.

    Args:
        sampled_trials (List[Trial]): List of sampled trials.
        preference_criterion (callable): Function to determine preference between trials.

    Returns:
        List[Tuple[Trial, Trial]]: List of preference pairs where the first trial is preferred over the second.
    """
    preference_pairs = list()
    for i, trial1 in enumerate(sampled_trials):
        for j, trial2 in enumerate(sampled_trials):
            if i == j:
                continue
            if preference_criterion(game, trial1, trial2):
                preference_pairs.append((trial1, trial2))
    return preference_pairs


def sample_trial(
    game: RepeatedReferenceGame,
    target: str,
    speaker: ChatSpeaker,
    listener: ChatListener,
    num_samples: Optional[int] = None,
    target_lengths: Optional[List[int]] = None,
) -> List[Trial]:
    """
    Generate samples for one trial of the game and create preference pairs.

    Args:
        game (RepeatedReferenceGame): The game instance.
        target (str): The target item for the trial.
        speaker (ChatSpeaker): The speaker agent.
        listener (ChatListener): The listener agent.
        num_samples (int): Number of trials to sample.
        preference_criterion (callable): Function to determine preference between trials.
    Returns:
        List[Trial]: List of sampled trials.
        List[Tuple[Trial, Trial]]: List of preference pairs where the first trial is preferred over the second.
    """
    sampled_messages = speaker.batch_generate(
        RepeatedReferenceGame(
            context=game.context,
            trials=[*game.trials, Trial(target=target)],
        ),
        num_return_sequences=num_samples,
        target_lengths=target_lengths,
    )

    listener_interpretations = listener.batch_score(
        [
            RepeatedReferenceGame(
                context=game.context,
                trials=[*game.trials, Trial(target=target, message=m)],
            )
            for m in sampled_messages
        ],
    )

    sampled_trials = [
        Trial(
            target=target,
            message=msg,
            interpretation=interp,
        )
        for msg, interp in zip(sampled_messages, listener_interpretations)
    ]

    return sampled_trials


def sample_game(
    context: List[str],
    num_trials: int,
    speaker: GenerateSpeaker,
    listener: ScoringListener,
    target_sequence_function: Optional[callable] = None,
    preference_criterion: Optional[callable] = None,
    num_samples: Optional[int] = None,
    target_lengths: Optional[List[int]] = None,
):
    """
    Sample a game with multiple trials and generate preference pairs.
    Args:
        context (List[str]): List of items in the context.
        num_trials (int): Number of trials to generate.
        speaker (GenerateSpeaker): The speaker agent.
        listener (ScoringListener): The listener agent.
        num_samples (int): Number of trials to sample.
        preference_criterion (callable): Function to determine preference between trials.
    Returns:
        List[dict]: List of dictionaries containing game data, sampled trials, and preference pairs.
    """
    game = RepeatedReferenceGame(context=context, trials=[])

    if target_sequence_function is None:
        targets = sequence_targets(context, num_trials)
    else:
        targets = target_sequence_function(context, num_trials)

    data = list()
    game_id = str(uuid.uuid4())
    for tgt in tqdm(targets, desc=f"Sampling game {game_id}"):
        sampled_trials = sample_trial(
            game,
            tgt,
            speaker,
            listener,
            num_samples=num_samples,
            target_lengths=target_lengths,
        )
        if preference_criterion is not None:
            preference_pairs = make_preference_pairs(
                sampled_trials, preference_criterion
            )
        else:
            preference_pairs = None

        data.append(
            {
                "game_id": game_id,
                "game": deepcopy(game),
                "trial_id": str(uuid.uuid4()),
                "sampled_trials": sampled_trials,
                "preference_pairs": preference_pairs,
            }
        )

        idx = np.random.choice(len(sampled_trials))
        game.trials.append(sampled_trials[idx])

    return data


def run_sampling(config_path: str):
    import json
    import jsonlines
    from pydantic_core import to_jsonable_python
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from agents.hf_speakers import GenerateSpeaker
    from agents.hf_listeners import ScoringListener

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(config.get("contexts_file"), "r") as f:
        contexts = json.load(f)

    speaker_model = AutoModelForImageTextToText.from_pretrained(
        **config.get("speaker_model")
    )
    speaker_processor = AutoProcessor.from_pretrained(**config.get("speaker_processor"))
    speaker = GenerateSpeaker(
        speaker_model,
        speaker_processor,
        system_prompt_template=SPEAKER_SYSTEM_PROMPT_BASIC,
        user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC,
        target_prompt_template=SPEAKER_USER_PROMPT_TARGET_BASIC,
        **config.get("speaker_config"),
    )

    listener_model = AutoModelForImageTextToText.from_pretrained(
        **config.get("listener_model")
    )
    listener_processor = AutoProcessor.from_pretrained(
        **config.get("listener_processor")
    )
    listener = ScoringListener(
        listener_model,
        listener_processor,
        **config.get("listener_config"),
    )

    preference_criterion = config.get("preference_criterion")
    if preference_criterion:
        preference_function = PREFERENCE_FUNCTIONS[preference_criterion]
    else:
        preference_function = None
    target_sequence_type = config.get("target_sequence_function")
    if target_sequence_type:
        target_sequence_function = TARGET_SEQUENCE_FUNCTIONS[target_sequence_type]
    else:
        target_sequence_function = None

    for context in contexts:
        data = sample_game(
            context,
            config.get("num_trials"),
            speaker,
            listener,
            target_sequence_function=target_sequence_function,
            preference_criterion=preference_function,
            num_samples=config.get("num_samples"),
            target_lengths=config.get("target_lengths"),
        )

        with jsonlines.open(config.get("save_file"), mode="a") as writer:
            writer.write_all(to_jsonable_python(data))


def run_preference_pairs(
    samples_file: str,
    preference_criterion: str,
    save_file: str,
    **preference_criterion_kwargs,
):
    import jsonlines
    from pydantic_core import to_jsonable_python

    print(preference_criterion_kwargs)

    # Get the function from the name
    preference_function = PREFERENCE_FUNCTIONS.get(preference_criterion)
    if preference_function is None:
        raise ValueError(f"Unknown preference criterion: {preference_criterion}")

    with jsonlines.open(samples_file, "r") as reader:
        data = list(reader)

    for x in tqdm(data, desc="Making preference pairs"):
        x["preference_pairs"] = make_preference_pairs(
            RepeatedReferenceGame.model_validate(x["game"]),
            list(map(Trial.model_validate, x["sampled_trials"])),
            lambda game, trial1, trial2: preference_function(
                game, trial1, trial2, **preference_criterion_kwargs
            ),
        )

    with jsonlines.open(save_file, "w") as writer:
        writer.write_all(to_jsonable_python(data))
