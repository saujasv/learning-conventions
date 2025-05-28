from typing import List, Dict
import numpy as np
import random
from game import RepeatedReferenceGame, Trial
import spacy
from Levenshtein import distance as edit_distance

nlp = spacy.load("en_core_web_sm")

TAGSET = ["NOUN", "ADJ", "VERB", "ADV", "PROPN", "NUM", "PRON", "ADP"]


def wnr(s1, s2):
    if not s1 or not s2:
        return None
    if len(s1) == 0 or len(s2) == 0:
        return None
    s1_content = [t.lemma_.lower() for t in s1 if t.pos_ in TAGSET]
    s2_content = [t.lemma_.lower() for t in s2 if t.pos_ in TAGSET]
    return edit_distance(s1_content, s2_content, weights=(1, 0, 1)) / len(s1)


def get_messages_by_target(game: RepeatedReferenceGame) -> Dict[str, List[str]]:
    """
    Get the messages by target for a game.
    """
    messages_by_target = {t: list() for t in game.context}
    for trial in game.trials:
        messages_by_target[trial.get_target()].append(trial.get_message())
    return messages_by_target


def sequence_targets(context: List[str], num_trials: int) -> List[str]:
    """
    Generate a sequence of targets based on the context.

    Args:
        context (List[str]): List of items in the context.
        num_trials (int): Number of trials to generate.
    Returns:
        List[str]: List of target items.
    """
    targets = list()
    used_targets = list()
    while len(targets) < num_trials:
        if len(used_targets) == len(context):
            idx = np.random.choice(len(context))
            targets.append(context[idx])
        elif len(used_targets) == 0:
            idx = np.random.choice(len(context))
            targets.append(context[idx])
            used_targets.append(context[idx])
        else:
            if np.random.rand() < 0.5:
                unused_targets = list(set(context) - set(used_targets))
                idx = np.random.choice(len(unused_targets))
                targets.append(unused_targets[idx])
                used_targets.append(unused_targets[idx])
            else:
                idx = np.random.choice(len(used_targets))
                targets.append(used_targets[idx])
    return targets


def sequence_targets_blocks(context: List[str], num_trials: int) -> List[str]:
    """
    Generate a sequence of targets based on the context.

    Args:
        context (List[str]): List of items in the context.
        num_trials (int): Number of trials to generate.
    Returns:
        List[str]: List of target items.
    """
    random.shuffle(context)
    repeated_targets = context[: len(context) // 2]
    control_targets = context[len(context) // 2 :]
    targets = list()
    for c in control_targets:
        targets.extend(random.sample([*repeated_targets, c], len(repeated_targets) + 1))
    targets.extend(random.sample(repeated_targets, len(repeated_targets)))

    return targets


def informativity_preference(game: RepeatedReferenceGame, trial1: Trial, trial2: Trial):
    """
    Determine if trial1 is preferred over trial2 based on informativity.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    if (
        trial1.get_interpretation()[trial1.get_target()]
        > trial2.get_interpretation()[trial2.get_target()]
    ):
        return True

    return False


def informativity_and_cost_preference(
    game: RepeatedReferenceGame, trial1: Trial, trial2: Trial
):
    """
    Determine if trial1 is preferred over trial2 based on informativity and cost.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    num_tokens1 = len(nlp(trial1.get_message()))
    num_tokens2 = len(nlp(trial2.get_message()))
    if (
        trial1.get_correct()
        and (
            trial1.get_interpretation()[trial1.get_target()]
            > trial2.get_interpretation()[trial2.get_target()]
        )
        and num_tokens1 < num_tokens2
    ):
        return True

    return False


def informativity_margin_and_cost_preference(
    game: RepeatedReferenceGame, trial1: Trial, trial2: Trial, min_p_target: float = 0.0
):
    """
    Determine if trial1 is preferred over trial2 based on informativity and cost.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    num_tokens1 = len(nlp(trial1.get_message()))
    num_tokens2 = len(nlp(trial2.get_message()))
    if (
        trial1.get_interpretation()[trial1.get_target()] > np.log(min_p_target)
        and (
            trial1.get_interpretation()[trial1.get_target()]
            > trial2.get_interpretation()[trial2.get_target()]
        )
        and num_tokens1 < num_tokens2
    ):
        return True

    return False


def correctness_and_cost_preference(
    game: RepeatedReferenceGame, trial1: Trial, trial2: Trial
):
    """
    Determine if trial1 is preferred over trial2 based on informativity and cost.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    num_tokens1 = len(nlp(trial1.get_message()))
    num_tokens2 = len(nlp(trial2.get_message()))
    if trial1.get_correct() and not trial2.get_correct() and num_tokens1 < num_tokens2:
        return True

    return False


def length_change_preference(game: RepeatedReferenceGame, trial1: Trial, trial2: Trial):
    """
    Determine if trial1 is preferred over trial2 based on length change.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    messages_by_target = get_messages_by_target(game)
    if len(messages_by_target[trial1.get_target()]) == 0:
        # we are in the first block
        return informativity_and_cost_preference(game, trial1, trial2)
    else:
        if len(nlp(trial1.get_message())) < len(
            nlp(messages_by_target[trial1.get_target()][-1])
        ) and len(nlp(trial2.get_message())) > len(
            nlp(messages_by_target[trial1.get_target()][-1])
        ):
            return True
        return False


def wnr_change_preference(game: RepeatedReferenceGame, trial1: Trial, trial2: Trial):
    """
    Determine if trial1 is preferred over trial2 based on wnr change.

    Args:
        game (RepeatedReferenceGame): Game with all previous trials.
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    messages_by_target = get_messages_by_target(game)
    if len(messages_by_target[trial1.get_target()]) == 0:
        # we are in the first block
        return informativity_and_cost_preference(game, trial1, trial2)
    else:
        processed_message1 = nlp(trial1.get_message())
        processed_message2 = nlp(trial2.get_message())
        processed_previous_message = nlp(messages_by_target[trial1.get_target()][-1])
        if wnr(processed_message1, processed_previous_message) < wnr(
            processed_message2, processed_previous_message
        ):
            return True
        return False
