from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import numpy as np
from agents.hf_speakers import GenerateSpeaker
from agents.hf_listeners import ScoringListener
import json
import jsonlines
from game import RepeatedReferenceGame, Trial
from tqdm import tqdm
from pathlib import Path

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
    speaker_type="model",
    speaker_model_name_or_path="mistral-community/pixtral-12b",
    listener_type="model",
    listener_model_name_or_path=None,
    image_base_path="/data/tir/projects/tir1/corpora/MSCOCO/images",
    speaker_sampling_method="top_p",
    games_path="data/coco_games.json",
    tangrams=False,
    num_samples=8,
    save_path="test.jsonl",
    listener_context_presentation="last_shuffle",
    speaker_context_presentation="last_no_shuffle",
):
    if speaker_sampling_method == "top_p":
        speaker_generation_config = {
            "max_new_tokens": 64,
            "top_p": 0.95,
            "temperature": 0.8,
            "stop_strings": ["\n"],
        }
    elif speaker_sampling_method == "min_p":
        speaker_generation_config = {
            "max_new_tokens": 64,
            "min_p": 0.1,
            "temperature": 1.5,
            "stop_strings": ["\n"],
        }

    if speaker_type == "model":
        model = AutoModelForVision2Seq.from_pretrained(
            speaker_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation={"text_config": "flash_attention_2"},
        )
        processor = AutoProcessor.from_pretrained(speaker_model_name_or_path)
        speaker = GenerateSpeaker(
            model,
            processor,
            image_base_path=image_base_path,
            generation_config=speaker_generation_config,
            context_presentation=speaker_context_presentation,
            feedback_label=True,
            max_image_size=256,
            tangrams=tangrams,
        )
    elif speaker_type == "replay":
        speaker = dict()
        with jsonlines.open(speaker_model_name_or_path) as reader:
            for x in reader:
                speaker[(x["gameid"], x["trial_idx"])] = {
                    "sampled_descriptions": x["sampled_descriptions"],
                    "ablated_descriptions": x["ablated_descriptions"],
                    "target": x["target"],
                }

    if listener_type == "model":
        if listener_model_name_or_path:
            listener_model = AutoModelForVision2Seq.from_pretrained(
                listener_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:1" if speaker_type == "model" else "cuda:0",
                attn_implementation={"text_config": "flash_attention_2"},
            )
            listener_processor = AutoProcessor.from_pretrained(
                listener_model_name_or_path
            )

            listener = ScoringListener(
                listener_model,
                listener_processor,
                image_base_path=image_base_path,
                context_presentation=listener_context_presentation,
                feedback_label=True,
                max_image_size=256,
            )
        else:
            listener = ScoringListener(
                model,
                processor,
                image_base_path=image_base_path,
                context_presentation=listener_context_presentation,
                feedback_label=True,
                max_image_size=256,
            )

    with open(games_path) as f:
        games = {
            k: RepeatedReferenceGame.model_validate(v) for k, v in json.load(f).items()
        }

    if Path(save_path).exists():
        with jsonlines.open(save_path) as reader:
            record = list(reader)
    else:
        record = list()

    completed_trials = [(r["gameid"], r["trial_idx"]) for r in record]
    print(f"Completed trials: {completed_trials}")

    for gameid, game in games.items():
        sampled_prior_trials = list()
        tiled_prior_trials = list()
        first_description_map = dict()
        for trial_idx, trial in enumerate(tqdm(game.trials)):
            if (gameid, trial_idx) in completed_trials:
                continue

            if speaker_type == "replay":
                assert (
                    speaker[(gameid, trial_idx)]["target"] == trial.target
                ), "Targets do not match"
                sampled_descriptions = speaker[(gameid, trial_idx)][
                    "sampled_descriptions"
                ]
            else:
                sampled_descriptions = [
                    speaker.generate(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[*sampled_prior_trials, Trial(target=trial.target)],
                        ),
                    )
                    for _ in range(num_samples)
                ]

            if listener_type == "model":
                sampled_descriptions_interpretations = [
                    listener.score(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[
                                *sampled_prior_trials,
                                Trial(target=trial.target, message=d),
                            ],
                        )
                    )
                    for d in sampled_descriptions
                ]
                # if len(sampled_descriptions) == 1:
                #     sampled_descriptions_interpretations = [
                #         sampled_descriptions_interpretations
                #     ]

                sampled_descriptions_interpretations_no_context = [
                    listener.score(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[Trial(target=trial.target, message=d)],
                        )
                    )
                    for d in sampled_descriptions
                ]

                # if len(sampled_descriptions) == 1:
                #     sampled_descriptions_interpretations_no_context = [
                #         sampled_descriptions_interpretations_no_context
                #     ]
                selected_description = sampled_descriptions[0]
                for desc, interp in zip(
                    sampled_descriptions, sampled_descriptions_interpretations
                ):
                    if max(interp, key=interp.get) == trial.target:
                        selected_description = desc
                        break
            else:
                sampled_descriptions_interpretations = [
                    {x: 0 if x == trial.target else None for x in game.context}
                    for d in sampled_descriptions
                ]

                sampled_descriptions_interpretations_no_context = [
                    {x: 0 if x == trial.target else None for x in game.context}
                    for d in sampled_descriptions
                ]

            if not trial.target in first_description_map:
                first_description_map[trial.target] = selected_description

            if listener_type == "model":
                tiled_description_interpretations = [
                    listener.score(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[
                                *tiled_prior_trials,
                                Trial(
                                    target=trial.target, message=selected_description
                                ),
                            ],
                        )
                    )
                ]
            else:
                tiled_description_interpretations = [
                    {x: 0 if x == trial.target else None for x in game.context}
                ]

            if speaker_type == "replay":
                ablated_descriptions = speaker[(gameid, trial_idx)][
                    "ablated_descriptions"
                ]
            else:
                ablated_descriptions = [
                    ablate_description(selected_description, p)
                    for p in ABLATION_PROBABILITIES[:num_samples]
                ]

            if listener_type == "model":
                ablated_descriptions_interpretations = [
                    listener.score(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[
                                *sampled_prior_trials,
                                Trial(target=trial.target, message=d),
                            ],
                        )
                    )
                    for d in ablated_descriptions
                ]
                if len(ablated_descriptions) == 1:
                    ablated_descriptions_interpretations = [
                        ablated_descriptions_interpretations
                    ]

                ablated_descriptions_interpretations_no_context = [
                    listener.score(
                        RepeatedReferenceGame(
                            context=game.context,
                            trials=[Trial(target=trial.target, message=d)],
                        )
                    )
                    for d in ablated_descriptions
                ]
                if len(ablated_descriptions) == 1:
                    ablated_descriptions_interpretations_no_context = [
                        ablated_descriptions_interpretations_no_context
                    ]
            else:
                ablated_descriptions_interpretations = [
                    {x: 0 if x == trial.target else None for x in game.context}
                    for d in ablated_descriptions
                ]

                ablated_descriptions_interpretations_no_context = [
                    {x: 0 if x == trial.target else None for x in game.context}
                    for d in ablated_descriptions
                ]

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
