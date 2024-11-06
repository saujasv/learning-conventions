import json
import random
from pathlib import Path
from PIL import Image
from game import RepeatedReferenceGame, Trial
from pydantic_core import from_json
from agents import (
    GPTListener,
    GPTSpeaker,
    CoGenListener,
    PixtralSpeaker,
    PixtralListener,
)
from tqdm import tqdm
import yaml
from collections import defaultdict
from vllm import LLM
from copy import deepcopy


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

        if not listener is None:
            trials[-1].selection = listener.select(
                RepeatedReferenceGame(
                    context=repeated_reference_game.context, trials=trials
                )
            )
        else:
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
        rrg = RepeatedReferenceGame.model_validate_json(json.dumps(rrg_data))
        games[gameid] = rrg

    return games


def main(config_path, config_idx=None):
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    for i, config in enumerate(configs["experiment_configs"]):
        if not config_idx is None and i != config_idx:
            continue

        games = load_games(config["games_path"])

        if config["listener_type"] == "pixtral" and config["speaker_type"] == "pixtral":
            llm = LLM(
                model=config["listener_config"]["model"],
                tokenizer_mode="mistral",
                limit_mm_per_prompt={"image": 64},
                max_model_len=15625,
                tensor_parallel_size=config["listener_config"]["tensor_parallel_size"],
            )
            listener = PixtralListener(model=llm)
            speaker = PixtralSpeaker(model=llm)
        elif config["listener_type"] == "gpt":
            listener = GPTListener(
                **config["listener_config"], image_base_path=config["images_path"]
            )
        elif config["listener_type"] == "pixtral":
            listener = PixtralListener(
                **config["listener_config"], image_base_path=config["images_path"]
            )
        elif config["listener_type"] == "cogen":
            listener = CoGenListener(
                **config["listener_config"], image_base_path=config["images_path"]
            )
        elif config["listener_type"] == "oracle":
            listener = None

        if config["speaker_type"] == "replay":
            speaker = None
        elif config["speaker_type"] == "gpt":
            speaker = GPTSpeaker(
                **config["speaker_config"], image_base_path=config["images_path"]
            )
        elif (
            config["speaker_type"] == "pixtral" and config["listener_type"] != "pixtral"
        ):
            speaker = PixtralSpeaker(**config["speaker_config"])

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
