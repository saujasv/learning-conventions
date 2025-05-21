from typing import List
import numpy as np
import random
from game import RepeatedReferenceGame, Trial
import stanza

nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma",
    tokenize_no_ssplit=True,
    use_gpu=False,
)


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
    repeated_targets = context[:len(context) // 2]
    control_targets = context[len(context) // 2 :]
    targets = list()
    for c in control_targets:
        targets.extend(random.sample([*repeated_targets, c], len(repeated_targets) + 1))
    targets.extend(random.sample(repeated_targets, len(repeated_targets)))

    return targets


def informativity_preference(trial1: Trial, trial2: Trial):
    """
    Determine if trial1 is preferred over trial2 based on informativity.

    Args:
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


def informativity_and_cost_preference(trial1: Trial, trial2: Trial):
    """
    Determine if trial1 is preferred over trial2 based on informativity and cost.

    Args:
        trial1 (Trial): First trial.
        trial2 (Trial): Second trial.
    Returns:
        bool: True if trial1 is preferred, False otherwise.
    """
    num_tokens1 = sum([len(s.words) for s in nlp(trial1.get_message()).sentences])
    num_tokens2 = sum([len(s.words) for s in nlp(trial2.get_message()).sentences])
    if (
        trial1.get_interpretation()[trial1.get_target()]
        > trial2.get_interpretation()[trial2.get_target()]
        and num_tokens1 < num_tokens2
    ):
        return True

    return False
