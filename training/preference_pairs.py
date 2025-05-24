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
)
from tqdm import tqdm


def make_preference_pairs(
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
            if preference_criterion(trial1, trial2):
                preference_pairs.append((trial1, trial2))
    return preference_pairs


def sample_trial(
    game: RepeatedReferenceGame,
    target: str,
    speaker: ChatSpeaker,
    listener: ChatListener,
    preference_criterion: callable,
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

    targets = sequence_targets_blocks(context, num_trials)

    data = list()
    game_id = str(uuid.uuid4())
    for tgt in tqdm(targets, desc=f"Sampling game {game_id}"):
        sampled_trials = sample_trial(
            game,
            tgt,
            speaker,
            listener,
            preference_criterion,
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


def run(config_path: str):
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

    for context in contexts:
        data = sample_game(
            context,
            config.get("num_trials"),
            speaker,
            listener,
            informativity_and_cost_preference,
            num_samples=config.get("num_samples"),
            target_lengths=config.get("target_lengths"),
        )

        with jsonlines.open(config.get("save_file"), mode="a") as writer:
            writer.write_all(to_jsonable_python(data))
