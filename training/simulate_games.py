from typing import List, Optional, Tuple
import uuid
from copy import deepcopy
import numpy as np
import itertools
from agents.game import RepeatedReferenceGame, Trial
from agents.base_agent import BaseSpeaker, BaseListener
from agents.hf_speakers import GenerateSpeaker
from agents.static_agents import ReplaySpeaker, OracleListener
from agents.hf_listeners import ScoringListener
from agents.openai_api_agent import OpenAIAPIListener
from agents.prompts import (
    SPEAKER_SYSTEM_PROMPT_BASIC,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC,
    SPEAKER_USER_PROMPT_TARGET_BASIC,
)
from training.simulation_utils import (
    sequence_targets,
    sequence_targets_blocks,
    TargetSequenceReplay,
    sequence_targets_blocks_controlled,
    hard_correctness_preference,
    hard_correctness_or_cost_preference,
    cost_preference,
)
from tqdm import tqdm
import random
from pathlib import Path

# Map preference criterion names to functions
PREFERENCE_FUNCTIONS = {
    "hard_correctness_preference": hard_correctness_preference,
    "hard_correctness_or_cost_preference": hard_correctness_or_cost_preference,
    "cost_preference": cost_preference,
}

TARGET_SEQUENCE_FUNCTIONS = {
    "sequence_targets": sequence_targets,
    "sequence_targets_blocks": sequence_targets_blocks,
    "sequence_targets_blocks_controlled": sequence_targets_blocks_controlled,
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


def make_preference_pairs_copeland(
    game: RepeatedReferenceGame,
    sampled_trials: List[Trial],
    preference_criterion: callable,
) -> List[Tuple[Trial, Trial]]:
    """
    Make preference pairs from sampled trials using Copeland winner.

    Args:
        sampled_trials (List[Trial]): List of sampled trials.
        preference_criterion (callable): Function to determine preference between trials.

    Returns:
        List[Tuple[Trial, Trial]]: List of preference pairs where the first trial is preferred over the second.
    """
    preference_pairs = list()
    for i, trial1 in enumerate(sampled_trials):
        preference_pairs_with_trial1 = list()
        for j, trial2 in enumerate(sampled_trials):
            if i == j:
                continue
            if preference_criterion(game, trial1, trial2):
                preference_pairs_with_trial1.append((trial1, trial2))
        preference_pairs.append(preference_pairs_with_trial1)

    n_copeland = max([len(p) for p in preference_pairs])

    return list(
        itertools.chain.from_iterable(
            [p for p in preference_pairs if len(p) == n_copeland]
        )
    )


def select_next_trial_random(
    game, sampled_trials, preference_criterion="informativity_and_cost_preference"
):
    """
    Randomly select one of the sampled trials.

    Args:
        game (RepeatedReferenceGame): The game instance.
        sampled_trials (List[Trial]): List of sampled trials.

    Returns:
        Trial: A randomly selected trial from sampled_trials.
    """
    return sampled_trials[
        0
    ]  # since the messages are assumed to be sampled independently, choosing the first always should still be a random choice


def select_next_trial_best(game, sampled_trials, preference_criterion=None):
    """
    Select the best trial from the sampled trials.
    """
    if preference_criterion is None:
        preference_criterion = PREFERENCE_FUNCTIONS[
            "hard_correctness_or_cost_preference"
        ]

    preference_pairs = make_preference_pairs_copeland(
        game, sampled_trials, preference_criterion
    )

    if len(preference_pairs) == 0:
        return sampled_trials[0]

    return preference_pairs[0][0]


def select_next_trial_mix(game, sampled_trials, preference_criterion=None):
    """
    Select the best trial from the sampled trials.
    """
    if random.random() < 0.5:
        return select_next_trial_best(game, sampled_trials, preference_criterion)
    else:
        return select_next_trial_random(game, sampled_trials, preference_criterion)


SELECT_NEXT_TRIAL_FUNCTIONS = {
    "select_next_trial_random": select_next_trial_random,
    "select_next_trial_best": select_next_trial_best,
    "select_next_trial_mix": select_next_trial_mix,
}


def sample_trial(
    game: RepeatedReferenceGame,
    target: str,
    speaker: BaseSpeaker,
    listener: BaseListener,
    num_samples: Optional[int] = None,
    target_lengths: Optional[List[int]] = None,
    context_id: Optional[str] = None,
) -> List[Trial]:
    """
    Generate samples for one trial of the game and create preference pairs.

    Args:
        game (RepeatedReferenceGame): The game instance.
        target (str): The target item for the trial.
        speaker (BaseSpeaker): The speaker agent.
        listener (BaseListener): The listener agent.
        num_samples (int): Number of trials to sample.
        preference_criterion (callable): Function to determine preference between trials.
    Returns:
        List[Trial]: List of sampled trials.
        List[Tuple[Trial, Trial]]: List of preference pairs where the first trial is preferred over the second.
    """
    if isinstance(speaker, ReplaySpeaker):
        sampled_messages = speaker.batch_generate(
            [
                RepeatedReferenceGame(
                    context=game.context,
                    trials=[*game.trials, Trial(target=target)],
                )
            ],
            num_return_sequences=num_samples,
            context_ids=[context_id],
        )[0]
    else:
        sampled_messages = speaker.batch_generate(
            [
                RepeatedReferenceGame(
                    context=game.context,
                    trials=[*game.trials, Trial(target=target)],
                )
            ],
            num_return_sequences=num_samples,
            target_lengths=target_lengths,
        )[0]

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
    select_next_trial: Optional[callable] = None,
    num_samples: Optional[int] = None,
    target_lengths: Optional[List[int]] = None,
    context_id: Optional[str] = None,
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
        select_next_trial (callable): Function to select the next trial.
    Returns:
        List[dict]: List of dictionaries containing game data, sampled trials, and preference pairs.
    """
    game = RepeatedReferenceGame(context=context, trials=[])

    if target_sequence_function is None:
        targets = sequence_targets(context, num_trials)
    elif isinstance(target_sequence_function, TargetSequenceReplay):
        targets = target_sequence_function(context, num_trials, context_id)
    else:
        targets = target_sequence_function(context, num_trials)

    if select_next_trial is None:
        select_next_trial = select_next_trial_random

    data = list()
    for tgt in tqdm(targets, desc=f"Sampling game {context_id}"):
        sampled_trials = sample_trial(
            game,
            tgt,
            speaker,
            listener,
            num_samples=num_samples,
            target_lengths=target_lengths,
            context_id=context_id,
        )
        if preference_criterion is not None:
            preference_pairs = make_preference_pairs_copeland(
                game, sampled_trials, preference_criterion
            )
        else:
            preference_pairs = None

        data.append(
            {
                "game_id": context_id,
                "game": deepcopy(game),
                "trial_id": str(uuid.uuid4()),
                "sampled_trials": sampled_trials,
                "preference_pairs": preference_pairs,
            }
        )

        next_trial = select_next_trial(
            game, sampled_trials, preference_criterion=preference_criterion
        )
        game.trials.append(next_trial)

    return data


def run_sampling(config_path: str):
    import json
    import jsonlines
    from pydantic_core import to_jsonable_python
    from transformers import AutoModelForImageTextToText, AutoProcessor

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(config.get("contexts_file"), "r") as f:
        contexts = json.load(f)

    if config.get("speaker_model_type") == "replay":
        speaker = ReplaySpeaker(config.get("speaker_config")["replay_data_path"])
    elif config.get("speaker_model_type") == "cogen":
        speaker = CoGenAgent(
            **config.get("speaker_config"),
        )
    else:
        speaker_model = AutoModelForImageTextToText.from_pretrained(
            **config.get("speaker_model")
        )
        speaker_processor = AutoProcessor.from_pretrained(
            **config.get("speaker_processor")
        )
        speaker_model_type = config.get("speaker_model_type")
        speaker = GenerateSpeaker(
            speaker_model_type,
            speaker_model,
            speaker_processor,
            **config.get("speaker_config"),
        )

    listener_model_type = config.get("listener_model_type")
    if listener_model_type == "oracle":
        listener = OracleListener()
    elif listener_model_type == "openai":
        listener = OpenAIAPIListener(
            **config.get("listener_config"),
        )
    elif listener_model_type == "cogen":
        from agents.cogen_agents import CoGenAgent

        listener = CoGenAgent(
            **config.get("listener_config"),
        )
    else:
        listener_model = AutoModelForImageTextToText.from_pretrained(
            **config.get("listener_model")
        )
        listener_processor = AutoProcessor.from_pretrained(
            **config.get("listener_processor")
        )
        listener = ScoringListener(
            listener_model_type,
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
    if target_sequence_type in TARGET_SEQUENCE_FUNCTIONS:
        target_sequence_function = TARGET_SEQUENCE_FUNCTIONS[target_sequence_type]
    elif Path(target_sequence_type).exists():
        target_sequence_function = TargetSequenceReplay(target_sequence_type)
    else:
        target_sequence_function = None

    select_next_trial = config.get("select_next_trial")
    if select_next_trial:
        select_next_trial_fn = SELECT_NEXT_TRIAL_FUNCTIONS[select_next_trial]
    else:
        select_next_trial_fn = select_next_trial_random

    Path(config.get("save_file")).parent.mkdir(parents=True, exist_ok=True)

    if Path(config.get("save_file")).exists():
        with jsonlines.open(config.get("save_file")) as reader:
            completed_contexts = set(x["game_id"] for x in reader)
    else:
        completed_contexts = set()

    for context_id, context in contexts.items():
        if context_id in completed_contexts:
            print(
                f"Skipping context {context_id} because it has already been completed"
            )
            continue

        data = sample_game(
            context,
            config.get("num_trials"),
            speaker,
            listener,
            target_sequence_function=target_sequence_function,
            preference_criterion=preference_function,
            select_next_trial=select_next_trial_fn,
            num_samples=config.get("num_samples"),
            target_lengths=config.get("target_lengths"),
            context_id=context_id,
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

    Path(save_file).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(save_file, "w") as writer:
        writer.write_all(to_jsonable_python(data))


def run_preference_pairs_copeland(
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
        x["preference_pairs"] = make_preference_pairs_copeland(
            RepeatedReferenceGame.model_validate(x["game"]),
            list(map(Trial.model_validate, x["sampled_trials"])),
            lambda game, trial1, trial2: preference_function(
                game, trial1, trial2, **preference_criterion_kwargs
            ),
        )

    Path(save_file).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(save_file, "w") as writer:
        writer.write_all(to_jsonable_python(data))
