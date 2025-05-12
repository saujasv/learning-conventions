from typing import List, Optional
import numpy as np
from pydantic import BaseModel
import uuid
from game import RepeatedReferenceGame, Trial
from agents.chat_speaker import ChatSpeaker
from agents.chat_listener import ChatListener
from tqdm import tqdm


class RepeatedReferenceGameSample(BaseModel):
    sample_id: str
    game: RepeatedReferenceGame
    listener_interpretation: Optional[dict] = None
    speaker_logprob: Optional[float] = None
    description_logprob: Optional[float] = None


def get_next_target(
    game: RepeatedReferenceGame, control_referents: Optional[List[str]] = None
) -> str:
    if control_referents is None:
        control_referents = list()

    if control_referents is None:
        num_complete_blocks = len(game.trials) // len(game.context)
    else:
        block_size = len(game.context) // 2 + 1
        num_complete_blocks = len(game.trials) // block_size
    current_block_targets = [
        x.target for x in game.trials[num_complete_blocks * len(game.context) :]
    ]
    chosen_control_targets = [
        x.target for x in game.trials if x.target in control_referents
    ]
    remaining_targets = [
        x
        for x in game.context
        if x not in current_block_targets and x not in chosen_control_targets
    ]
    if len(remaining_targets) == 0:
        return None
    else:
        return np.random.choice(remaining_targets)


def perturb_sample(message: str, num_perturbations: int) -> List[str]:
    words = message.split()
    if 1 >= 2 ** len(words) - 1:
        return list()
    masks = [
        list(map(int, bin(x)[2:].zfill(len(words))))
        for x in np.random.randint(
            1, 2 ** len(words) - 1, size=min(num_perturbations, 2 ** len(words) - 2)
        )
    ]
    perturbed_samples = list()
    for m in masks:
        perturbed_samples.append(
            " ".join([w for w, mask in zip(words, m) if mask == 1])
        )

    return perturbed_samples


def step(
    frontier: List[RepeatedReferenceGameSample],
    speaker: ChatSpeaker,
    listener: ChatListener,
    num_samples: int = 1,
    num_perturbations: int = 1,
    control_referents: Optional[List[str]] = None,
):
    """
    Perform a single step of the rollout process.

    Args:
        frontier (List[RepeatedReferenceGameSample]): The current frontier of games.
        speaker (ChatSpeaker): The speaker agent.
        listener (ChatListener): The listener agent.
        num_samples (int): The number of samples to generate.
        num_perturbations (int): The number of perturbations to apply to each sample.

    Returns:
        List[RepeatedReferenceGameSample]: The updated frontier of games.
    """
    samples = list()
    for game_state in frontier:
        target = get_next_target(game_state.game, control_referents)
        if target is None:
            continue
        sample_id = str(uuid.uuid4())
        for _ in range(num_samples):
            message = speaker.generate(
                RepeatedReferenceGame(
                    context=game_state.game.context,
                    trials=[*game_state.game.trials, Trial(target=target)],
                )
            )

            perturbations = perturb_sample(message, num_perturbations)

            expansion_trials = [
                Trial(target=target, message=message),
                *[Trial(target=target, message=x) for x in perturbations],
            ]

            interpretations = listener.batch_score(
                [
                    RepeatedReferenceGame(
                        context=game_state.game.context,
                        trials=[*game_state.game.trials, t],
                    )
                    for t in expansion_trials
                ]
            )

            selections = [max(x, key=x.get) for x in interpretations]

            expansion_trials = [
                Trial(
                    target=t.target,
                    message=t.message,
                    selection=s,
                    correct=s == t.target,
                )
                for t, s in zip(expansion_trials, selections)
            ]

            speaker_logprobs = speaker.batch_score(
                [
                    RepeatedReferenceGame(
                        context=game_state.game.context,
                        trials=[*game_state.game.trials, t],
                    )
                    for t in expansion_trials
                ]
            )

            description_logprobs = speaker.batch_score_isolated(
                [
                    RepeatedReferenceGame(
                        context=game_state.game.context,
                        trials=[*game_state.game.trials, t],
                    )
                    for t in expansion_trials
                ]
            )

            samples.extend(
                [
                    RepeatedReferenceGameSample(
                        sample_id=sample_id,
                        game=RepeatedReferenceGame(
                            context=game_state.game.context,
                            trials=[*game_state.game.trials, t],
                        ),
                        listener_interpretation=listener_interpretation,
                        speaker_logprob=speaker_logprob,
                        description_logprob=description_logprob,
                    )
                    for t, listener_interpretation, speaker_logprob, description_logprob in zip(
                        expansion_trials,
                        interpretations,
                        speaker_logprobs,
                        description_logprobs,
                    )
                ]
            )

    return samples


def state_utility(sample: RepeatedReferenceGameSample) -> float:
    return 1.0


def rollout(
    context: List[str],
    speaker: ChatSpeaker,
    listener: ChatListener,
    max_trials: int,
    num_samples: int,
    num_perturbations: int,
    num_expansions: int,
    use_controlled_referents: bool = False,
):
    beam = [
        RepeatedReferenceGameSample(
            sample_id=str(uuid.uuid4()), game=RepeatedReferenceGame(context=context)
        )
    ]

    control_referents = (
        np.random.choice(context, len(context) // 2, replace=False)
        if use_controlled_referents
        else None
    )
    samples = list()
    with tqdm(total=max_trials) as pbar:
        pbar.set_description("Rollout")
        while len(beam) > 0 and len(beam[0].game.trials) < max_trials:
            candidates = step(
                beam,
                speaker,
                listener,
                num_samples=num_samples,
                num_perturbations=num_perturbations,
                control_referents=control_referents,
            )
            samples.extend(candidates)

            candidate_scores = np.array([state_utility(x) for x in candidates])
            candidate_probs = np.exp(candidate_scores - np.max(candidate_scores))
            candidate_probs /= np.sum(candidate_probs)
            selected_idx = np.argsort(candidate_probs + np.random.gumbel())[
                :num_expansions
            ]
            beam = [candidates[i] for i in selected_idx]
            pbar.update(1)

    return samples


def main(
    model_name_or_path: str,
    contexts_file: str,
    save_path: str,
    images_base_path: str,
    num_blocks: int,
    num_samples: int,
    num_perturbations: int,
    num_expansions: int,
    control_referents: bool = False,
):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from agents.hf_speakers import GenerateSpeaker
    from agents.hf_listeners import ScoringListener
    import torch
    import json
    import jsonlines

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation={"text_config": "flash_attention_2"},
    )

    speaker = GenerateSpeaker(
        model,
        processor,
        image_base_path=images_base_path,
        generation_config={
            "do_sample": True,
            "max_new_tokens": 32,
            "min_p": 0.1,
            "temperature": 1.5,
        },
        context_presentation="once",
        max_image_size=256,
    )

    listener = ScoringListener(
        model,
        processor,
        image_base_path=images_base_path,
        context_presentation="once",
        max_image_size=256,
    )

    with open(contexts_file, "r") as f:
        contexts = json.load(f)

    all_samples = list()
    for context in contexts:
        if control_referents:
            num_trials = (len(context) // 2 + 1) * num_blocks
        else:
            num_trials = len(context) * num_blocks

        try:
            samples = rollout(
                context,
                speaker,
                listener,
                num_trials,
                num_samples,
                num_perturbations,
                num_expansions,
                control_referents,
            )
        except Exception as e:
            print(f"Error during rollout: {e}")
            continue

        all_samples.extend(samples)
        with jsonlines.open(save_path, mode="w") as writer:
            for sample in all_samples:
                writer.write(sample.model_dump(mode="json"))
