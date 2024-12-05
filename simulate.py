import json
import random
from pathlib import Path
from PIL import Image
from game import RepeatedReferenceGame, Trial
from pydantic_core import from_json
from agents import (
    GPTListener,
    GPTSpeaker,
    vLLMListener,
    vLLMSpeaker,
    ScoringListener,
    GenerateListener,
    GenerateSpeaker,
    JointInferenceListener,
    JointInferenceSpeaker,
    CoGenListener,
)
from tqdm import tqdm
import yaml
from collections import defaultdict
from vllm import LLM
from copy import deepcopy
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def simulate(
    repeated_reference_game,
    speaker=None,
    listener=None,
    teacher_force=False,
    gameid=None,
):
    simulated_trials = list()
    for i, trial in enumerate(
        tqdm(
            repeated_reference_game.trials,
            desc=f"Simulating {gameid}" if gameid else "Simulating",
        )
    ):
        previous_trials = (
            deepcopy(repeated_reference_game.trials[:i])
            if teacher_force
            else deepcopy(simulated_trials)
        )
        trials = [
            *previous_trials,
            Trial(target=trial.target),
        ]

        if not speaker is None:
            trials[-1].message = speaker.generate(
                RepeatedReferenceGame(
                    context=repeated_reference_game.context, trials=trials
                )
            )
        else:
            trials[-1].message = trial.message

        if not isinstance(listener, str):
            trials[-1].selection = listener.select(
                RepeatedReferenceGame(
                    context=repeated_reference_game.context, trials=trials
                )
            )
        elif listener == "replay":
            trials[-1].selection = repeated_reference_game.trials[i].selection
        elif listener == "oracle":
            trials[-1].selection = trials[-1].target

        if trials[-1].selection is None:
            trials[-1].correct = False
        elif trials[-1].selection == trial.target:
            trials[-1].correct = True
        else:
            trials[-1].correct = False

        simulated_trials.append(trials[-1])

    return RepeatedReferenceGame(
        context=repeated_reference_game.context, trials=simulated_trials
    )


def load_games(games_path):
    with open(games_path, "r") as f:
        data = json.load(f)

    games = dict()
    for gameid, rrg_data in data.items():
        rrg = RepeatedReferenceGame.model_validate(rrg_data)
        games[gameid] = rrg

    return games


def main(config_path, config_idx=None):
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    for i, config in enumerate(configs["experiment_configs"]):
        if not config_idx is None and i != config_idx:
            continue

        print(config)

        games = load_games(config["games_path"])

        if config.get("model_name_or_path", None):
            processor = AutoProcessor.from_pretrained(config["model_name_or_path"])
            model = AutoModelForVision2Seq.from_pretrained(
                config["model_name_or_path"], torch_dtype=torch.float16
            ).to("cuda")
        else:
            model = None
            processor = None

        if config["listener_type"] == "scoring":
            listener = ScoringListener(
                model,
                processor,
                **config["listener_config"],
                image_base_path=config["images_path"],
            )
        elif config["listener_type"] == "generate":
            listener = GenerateListener(
                model,
                processor,
                **config["listener_config"],
                image_base_path=config["images_path"],
            )
        elif config["listener_type"] == "joint_inference":
            listener = JointInferenceListener(
                model,
                processor,
                model,
                processor,
                **config["listener_config"],
                image_base_path=config["images_path"],
            )
        elif config["listener_type"] == "gpt":
            listener = GPTListener(
                **config["listener_config"], image_base_path=config["images_path"]
            )
        elif config["listener_type"] == "cogen":
            listener = CoGenListener(
                **config["listener_config"], image_base_path=config["images_path"]
            )
        elif config["listener_type"] == "replay":
            listener = "replay"
        elif config["listener_type"] == "oracle":
            listener = "oracle"

        if config["speaker_type"] == "replay":
            speaker = None
        elif config["speaker_type"] == "gpt":
            speaker = GPTSpeaker(
                **config["speaker_config"], image_base_path=config["images_path"]
            )
        elif config["speaker_type"] == "generate":
            speaker = GenerateSpeaker(
                model,
                processor,
                **config["speaker_config"],
                image_base_path=config["images_path"],
            )
        elif config["speaker_type"] == "joint_inference":
            speaker = JointInferenceSpeaker(
                model,
                processor,
                model,
                processor,
                **config["speaker_config"],
                image_base_path=config["images_path"],
            )

        simulated_games = dict()
        iterator = configs.get("selected_games", games.keys())
        for gameid in tqdm(iterator, desc="Simulating games"):
            game = games[gameid]
            simulated_game = simulate(
                game,
                listener=listener,
                speaker=speaker,
                teacher_force=config.get("teacher_force", False),
                gameid=gameid,
            )

            simulated_games[gameid] = simulated_game.model_dump(mode="json")
            with open(config["simulation_save_path"], "w") as f:
                json.dump(simulated_games, f)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
