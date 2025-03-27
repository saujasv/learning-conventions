from vllm import LLM
import numpy as np
from agents.vllm_agents import vLLMSpeaker, vLLMListener
import json
import jsonlines
from game import RepeatedReferenceGame, Trial
from tqdm import tqdm

ABLATION_PROBABILITIES = [
    0.2,
    0.2,
    0.4,
    0.4,
    0.6,
    0.6,
    0.8,
    0.8,
    0.2,
    0.2,
    0.4,
    0.4,
    0.6,
    0.6,
    0.8,
    0.8,
]


def ablate_description(description, p):
    words = description.split(" ")
    mask = [0 for _ in words]

    while np.sum(mask) == 0:
        mask = np.random.choice([0, 1], size=len(words), p=[p, 1 - p])

    return " ".join([word for i, word in enumerate(words) if mask[i]])


def simulate_sampling(
    speaker_model_name_or_path="mistral-community/pixtral-12b",
    speaker_tensor_parallel_size=2,
    listener_model_name_or_path=None,
    listener_tensor_parallel_size=1,
    image_base_path="/data/tir/projects/tir1/corpora/MSCOCO/images",
    speaker_sampling_method="top_p",
    games_path="data/coco_games.json",
    tangrams=False,
    num_samples=8,
    save_path="test.jsonl",
):
    if speaker_sampling_method == "top_p":
        speaker_generation_config = {
            "max_tokens": 64,
            "top_p": 0.95,
            "temperature": 0.8,
            "stop": ["\n"],
        }
    elif speaker_sampling_method == "min_p":
        speaker_generation_config = {
            "max_tokens": 64,
            "min_p": 0.1,
            "temperature": 1.5,
            "stop": ["\n"],
        }

    llm = LLM(
        speaker_model_name_or_path,
        tensor_parallel_size=speaker_tensor_parallel_size,
        gpu_memory_utilization=0.95,
        limit_mm_per_prompt={"image": 256},
        max_model_len=16384,
    )
    speaker = vLLMSpeaker(
        llm,
        image_base_path=image_base_path,
        context_presentation="last_no_shuffle",
        feedback_label=True,
        generation_config=speaker_generation_config,
        tangrams=tangrams,
    )

    if listener_model_name_or_path:
        listener_llm = LLM(
            listener_model_name_or_path,
            tensor_parallel_size=listener_tensor_parallel_size,
            gpu_memory_utilization=0.95,
            limit_mm_per_prompt={"image": 256},
            max_model_len=16384,
        )
        listener = vLLMListener(
            listener_llm,
            image_base_path=image_base_path,
            context_presentation="last_shuffle",
            feedback_label=True,
            generation_config=None,
            tangrams=tangrams,
        )
    else:
        listener = vLLMListener(
            llm,
            image_base_path=image_base_path,
            context_presentation="last_shuffle",
            feedback_label=True,
            generation_config=None,
            tangrams=tangrams,
        )

    with open(games_path) as f:
        games = {
            k: RepeatedReferenceGame.model_validate(v) for k, v in json.load(f).items()
        }

    record = list()

    for gameid, game in games.items():
        sampled_prior_trials = list()
        tiled_prior_trials = list()
        first_description_map = dict()
        for trial_idx, trial in enumerate(tqdm(game.trials)):
            sampled_descriptions = speaker.generate(
                RepeatedReferenceGame(
                    context=game.context,
                    trials=[*sampled_prior_trials, Trial(target=trial.target)],
                ),
                num_return_sequences=num_samples,
            )

            sampled_descriptions_interpretations = listener.score(
                [
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[
                            *sampled_prior_trials,
                            Trial(target=trial.target, message=d),
                        ],
                    )
                    for d in sampled_descriptions
                ]
            )

            sampled_descriptions_interpretations_no_context = listener.score(
                [
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[Trial(target=trial.target, message=d)],
                    )
                    for d in sampled_descriptions
                ]
            )

            selected_description = sampled_descriptions[0]
            for desc, interp in zip(
                sampled_descriptions, sampled_descriptions_interpretations
            ):
                if max(interp, key=interp.get) == trial.target:
                    selected_description = desc
                    break

            if not trial.target in first_description_map:
                first_description_map[trial.target] = selected_description

            tiled_description_interpretations = listener.score(
                [
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[
                            *tiled_prior_trials,
                            Trial(target=trial.target, message=selected_description),
                        ],
                    )
                ]
            )

            ablated_descriptions = [
                ablate_description(selected_description, p)
                for p in ABLATION_PROBABILITIES[:num_samples]
            ]

            ablated_descriptions_interpretations = listener.score(
                [
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[
                            *sampled_prior_trials,
                            Trial(target=trial.target, message=d),
                        ],
                    )
                    for d in ablated_descriptions
                ]
            )

            ablated_descriptions_interpretations_no_context = listener.score(
                [
                    RepeatedReferenceGame(
                        context=game.context,
                        trials=[Trial(target=trial.target, message=d)],
                    )
                    for d in ablated_descriptions
                ]
            )

            sampled_prior_trials.append(
                Trial(
                    target=trial.target,
                    message=selected_description,
                    selection=trial.target,
                    correct=True,
                )
            )

            tiled_prior_trials.append(
                Trial(
                    target=trial.target,
                    message=first_description_map[trial.target],
                    selection=trial.target,
                    correct=True,
                )
            )

            record.append(
                {
                    "speaker_model": speaker_model_name_or_path,
                    "listener_model": listener_model_name_or_path,
                    "speaker_sampling_method": speaker_sampling_method,
                    "gameid": gameid,
                    "trial_idx": trial_idx,
                    "target": trial.target,
                    "sampled_descriptions": sampled_descriptions,
                    "sampled_descriptions_interpretations": sampled_descriptions_interpretations,
                    "sampled_descriptions_interpretations_no_context": sampled_descriptions_interpretations_no_context,
                    "selected_description": selected_description,
                    "tiled_description_interpretations": tiled_description_interpretations,
                    "ablated_descriptions": ablated_descriptions,
                    "ablated_descriptions_interpretations": ablated_descriptions_interpretations,
                    "ablated_descriptions_interpretations_no_context": ablated_descriptions_interpretations_no_context,
                }
            )

            with jsonlines.open(save_path, "w") as writer:
                writer.write_all(record)
