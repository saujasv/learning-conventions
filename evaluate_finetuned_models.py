import os
import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from game import RepeatedReferenceGame, Trial
from transformers import AutoModelForVision2Seq, AutoProcessor
from agents import GenerateSpeaker, GenerateListener, ScoringListener
import numpy as np

# models_path = Path(os.environ["MODELS_PATH"])
# images_path = Path(os.environ["IMAGES_PATH"])
# games_path = Path(os.environ["GAMES_PATH"])
# results_save_dir = Path(os.environ["RESULTS_SAVE_DIR"])


def evaluate(
    speaker_model_name_or_path,
    listener_model_name_or_path,
    games_path,
    images_path,
    save_path,
    n_speaker_samples=16,
    speaker_feedback_label=False,
    listener_feedback_label=True,
    speaker_adapter_path=None,
    listener_adapter_path=None,
    use_length_token=False,
    tangrams=False,
):
    processor = AutoProcessor.from_pretrained(speaker_model_name_or_path)
    speaker_model = AutoModelForVision2Seq.from_pretrained(
        speaker_model_name_or_path, torch_dtype="auto", device_map="cuda:0"
    )
    if speaker_adapter_path:
        speaker_model.load_adapter(speaker_adapter_path)
    speaker = GenerateSpeaker(
        speaker_model,
        processor,
        image_base_path=str(images_path),
        generation_config={"do_sample": True, "top_p": 0.95, "temperature": 0.8},
        feedback_label=speaker_feedback_label,
        use_length_token=use_length_token,
        context_presentation="last_no_shuffle",
        tangrams=tangrams,
    )

    listener_model = AutoModelForVision2Seq.from_pretrained(
        listener_model_name_or_path, torch_dtype="auto", device_map="cuda:1"
    )
    if listener_adapter_path:
        listener_model.load_adapter(listener_adapter_path)
    listener = ScoringListener(
        listener_model,
        processor,
        image_base_path=str(images_path),
        context_presentation="last_shuffle",
        feedback_label=listener_feedback_label,
    )

    with open(games_path) as f:
        games = {
            gameid: RepeatedReferenceGame.model_validate(g)
            for gameid, g in json.load(f).items()
        }

    logs = defaultdict(list)
    for gameid, game in games.items():
        simulated_trials = list()
        for i, trial in enumerate(
            tqdm(
                game.trials,
                desc=f"Simulating {gameid}" if gameid else "Simulating",
            )
        ):
            previous_trials = deepcopy(simulated_trials)
            best_trial = None
            for sample_idx in range(2, n_speaker_samples + 2):
                trials = [
                    *previous_trials,
                    Trial(target=trial.target),
                ]

                if use_length_token:
                    trials[-1].message = speaker.generate(
                        RepeatedReferenceGame(context=game.context, trials=trials),
                        target_length=sample_idx,
                    )
                else:
                    trials[-1].message = speaker.generate(
                        RepeatedReferenceGame(context=game.context, trials=trials)
                    )

                if trials[-1].message is None:
                    listener_probs = {}
                    trials[-1].selection = random.choice(game.context)
                else:
                    listener_probs = listener.score(
                        RepeatedReferenceGame(context=game.context, trials=trials)
                    )
                    trials[-1].selection = max(
                        listener_probs.items(), key=lambda x: x[1]
                    )[0]

                if trials[-1].selection is None:
                    trials[-1].correct = False
                elif trials[-1].selection == trial.target:
                    trials[-1].correct = True
                else:
                    trials[-1].correct = False

                logs[f"{gameid}_trialidx={i}_sampleidx={sample_idx}"].append(
                    {
                        "game": RepeatedReferenceGame(
                            context=game.context, trials=trials
                        ).model_dump(mode="json"),
                        "listener_probs": listener_probs,
                    }
                )

                if best_trial is None:
                    best_trial = trials[-1]
                elif trials[-1].correct and not best_trial.correct:
                    best_trial = trials[-1]
                elif (
                    trials[-1].message
                    and best_trial.message
                    and len(trials[-1].message) >= len(best_trial.message)
                ):
                    best_trial = trials[-1]

            simulated_trials.append(best_trial)

        with open(save_path, "w") as f:
            json.dump(logs, f)


def evaluate_dropout(
    speaker_model_name_or_path,
    listener_model_name_or_path,
    games_path,
    images_path,
    save_path,
    n_speaker_samples=16,
    speaker_feedback_label=False,
    listener_feedback_label=True,
    speaker_adapter_path=None,
    listener_adapter_path=None,
    use_length_token=False,
    tangrams=False,
):
    processor = AutoProcessor.from_pretrained(speaker_model_name_or_path)
    speaker_model = AutoModelForVision2Seq.from_pretrained(
        speaker_model_name_or_path, torch_dtype="auto", device_map="cuda:0"
    )
    if speaker_adapter_path:
        speaker_model.load_adapter(speaker_adapter_path)
    speaker = GenerateSpeaker(
        speaker_model,
        processor,
        image_base_path=str(images_path),
        generation_config={"do_sample": True, "top_p": 0.95, "temperature": 0.8},
        feedback_label=speaker_feedback_label,
        use_length_token=use_length_token,
        context_presentation="last_no_shuffle",
        tangrams=tangrams,
    )

    listener_model = AutoModelForVision2Seq.from_pretrained(
        listener_model_name_or_path, torch_dtype="auto", device_map="cuda:1"
    )
    if listener_adapter_path:
        listener_model.load_adapter(listener_adapter_path)
    listener = ScoringListener(
        listener_model,
        processor,
        image_base_path=str(images_path),
        context_presentation="last_shuffle",
        feedback_label=listener_feedback_label,
    )

    with open(games_path) as f:
        games = {
            gameid: RepeatedReferenceGame.model_validate(g)
            for gameid, g in json.load(f).items()
        }

    logs = defaultdict(list)
    for gameid, game in games.items():
        simulated_trials = list()
        for i, trial in enumerate(
            tqdm(
                game.trials,
                desc=f"Simulating {gameid}" if gameid else "Simulating",
            )
        ):
            previous_trials = deepcopy(simulated_trials)
            trials = [
                *previous_trials,
                Trial(target=trial.target),
            ]

            original_message = speaker.generate(
                RepeatedReferenceGame(context=game.context, trials=trials)
            )

            trials[-1].message = original_message

            if trials[-1].message is None:
                listener_probs = {}
                trials[-1].selection = random.choice(game.context)
            else:
                listener_probs = listener.score(
                    RepeatedReferenceGame(context=game.context, trials=trials)
                )
                trials[-1].selection = max(listener_probs.items(), key=lambda x: x[1])[
                    0
                ]

            if trials[-1].selection is None:
                trials[-1].correct = False
            elif trials[-1].selection == trial.target:
                trials[-1].correct = True
            else:
                trials[-1].correct = False

            original_selection = trials[-1].selection

            logs[f"{gameid}_trialidx={i}_sampleidx=0"].append(
                {
                    "game": RepeatedReferenceGame(
                        context=game.context, trials=trials
                    ).model_dump(mode="json"),
                    "listener_probs": listener_probs,
                }
            )

            original_message_words = original_message.split()

            for sample_idx in range(2, n_speaker_samples + 2):
                trials = [
                    *previous_trials,
                    Trial(target=trial.target),
                ]

                prop = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8}[sample_idx % 4]

                mask = np.random.choice(
                    [0, 1], size=len(original_message_words), p=[prop, 1 - prop]
                )

                trials[-1].message = " ".join(
                    [word for i, word in enumerate(original_message_words) if mask[i]]
                )

                if trials[-1].message is None:
                    listener_probs = {}
                    trials[-1].selection = random.choice(game.context)
                else:
                    listener_probs = listener.score(
                        RepeatedReferenceGame(context=game.context, trials=trials)
                    )
                    trials[-1].selection = max(
                        listener_probs.items(), key=lambda x: x[1]
                    )[0]

                if trials[-1].selection is None:
                    trials[-1].correct = False
                elif trials[-1].selection == trial.target:
                    trials[-1].correct = True
                else:
                    trials[-1].correct = False

                logs[f"{gameid}_trialidx={i}_sampleidx={sample_idx}"].append(
                    {
                        "game": RepeatedReferenceGame(
                            context=game.context, trials=trials
                        ).model_dump(mode="json"),
                        "listener_probs": listener_probs,
                    }
                )

            simulated_trials.append(
                Trial(
                    target=trial.target,
                    message=original_message,
                    selection=original_selection,
                    correct=trial.target == original_selection,
                )
            )

        with open(save_path, "w") as f:
            json.dump(logs, f)


def evaluate_oracle(
    speaker_model_name_or_path,
    games_path,
    images_path,
    save_path,
    n_speaker_samples=16,
    speaker_feedback_label=False,
    use_length_token=False,
):
    processor = AutoProcessor.from_pretrained(speaker_model_name_or_path)
    speaker_model = AutoModelForVision2Seq.from_pretrained(
        speaker_model_name_or_path, torch_dtype="auto", device_map="cuda:0"
    )
    speaker = GenerateSpeaker(
        speaker_model,
        processor,
        image_base_path=str(images_path),
        generation_config={"do_sample": True, "top_p": 0.95, "temperature": 0.7},
        feedback_label=speaker_feedback_label,
        use_length_token=use_length_token,
    )

    with open(games_path) as f:
        games = {
            gameid: RepeatedReferenceGame.model_validate(g)
            for gameid, g in json.load(f).items()
        }

    logs = defaultdict(list)
    for gameid, game in games.items():
        simulated_trials = list()
        for i, trial in enumerate(
            tqdm(
                game.trials,
                desc=f"Simulating {gameid}" if gameid else "Simulating",
            )
        ):
            previous_trials = deepcopy(simulated_trials)
            best_trial = None
            for sample_idx in range(2, n_speaker_samples + 2):
                trials = [
                    *previous_trials,
                    Trial(target=trial.target),
                ]

                if use_length_token:
                    trials[-1].message = speaker.generate(
                        RepeatedReferenceGame(context=game.context, trials=trials),
                        target_length=sample_idx,
                    )
                else:
                    trials[-1].message = speaker.generate(
                        RepeatedReferenceGame(context=game.context, trials=trials)
                    )

                trials[-1].selection = trials[-1].target

                if trials[-1].selection is None:
                    trials[-1].correct = False
                elif trials[-1].selection == trial.target:
                    trials[-1].correct = True
                else:
                    trials[-1].correct = False

                logs[f"{gameid}_trialidx={i}_sampleidx={sample_idx}"].append(
                    {
                        "game": RepeatedReferenceGame(
                            context=game.context, trials=trials
                        ).model_dump(mode="json")
                    }
                )

                best_trial = trials[-1]

            simulated_trials.append(best_trial)

        with open(save_path, "w") as f:
            json.dump(logs, f)


def evaluate_replay(
    listener_model_name_or_path,
    games_path,
    images_path,
    save_path,
    listener_feedback_label=False,
):
    processor = AutoProcessor.from_pretrained(listener_model_name_or_path)

    listener_model = AutoModelForVision2Seq.from_pretrained(
        listener_model_name_or_path, torch_dtype="auto", device_map="cuda:1"
    )
    listener = ScoringListener(
        listener_model,
        processor,
        image_base_path=str(images_path),
        context_presentation="last_shuffle",
        feedback_label=listener_feedback_label,
    )

    with open(games_path) as f:
        games = {
            gameid: RepeatedReferenceGame.model_validate(g)
            for gameid, g in json.load(f).items()
        }

    logs = defaultdict(list)
    for gameid, game in games.items():
        simulated_trials = list()
        for i, trial in enumerate(
            tqdm(
                game.trials,
                desc=f"Simulating {gameid}" if gameid else "Simulating",
            )
        ):
            previous_trials = deepcopy(game.trials[:i])
            best_trial = None
            trials = [
                *previous_trials,
                Trial(target=trial.target),
            ]

            trials[-1].message = game.trials[i].message

            if trials[-1].message is None:
                listener_probs = {}
                trials[-1].selection = random.choice(game.context)
            else:
                listener_probs = listener.score(
                    RepeatedReferenceGame(context=game.context, trials=trials)
                )
                trials[-1].selection = max(listener_probs.items(), key=lambda x: x[1])[
                    0
                ]

            if trials[-1].selection is None:
                trials[-1].correct = False
            elif trials[-1].selection == trial.target:
                trials[-1].correct = True
            else:
                trials[-1].correct = False

            logs[f"{gameid}_trialidx={i}_sampleidx=2"].append(
                {
                    "game": RepeatedReferenceGame(
                        context=game.context, trials=trials
                    ).model_dump(mode="json"),
                    "listener_probs": listener_probs,
                }
            )

            simulated_trials.append(trials[-1])

        with open(save_path, "w") as f:
            json.dump(logs, f)


def evaluate_reuse():
    processor = AutoProcessor.from_pretrained("saujasv/pixtral-12b")
    speaker_model = AutoModelForVision2Seq.from_pretrained(
        "saujasv/pixtral-12b", torch_dtype="auto", device_map="cuda:0"
    )
    speaker_model.load_adapter(
        models_path / "pixtral_full_adapter_r=16-speaker_parts" / "checkpoint-350"
    )
    speaker = GenerateSpeaker(
        speaker_model,
        processor,
        image_base_path=str(images_path),
        generation_config={"do_sample": True, "top_p": 0.95, "temperature": 0.7},
        use_length_token=True,
    )

    listener_model = AutoModelForVision2Seq.from_pretrained(
        "saujasv/pixtral-12b", torch_dtype="auto", device_map="cuda:1"
    )
    listener_model.load_adapter(
        models_path
        / "pixtral_full_adapter-listener_parts-length_token"
        / "checkpoint-600"
    )
    listener = GenerateListener(
        listener_model,
        processor,
        image_base_path=str(images_path),
        context_presentation="last_shuffle",
    )

    with open(games_path) as f:
        games = {
            gameid: RepeatedReferenceGame.model_validate(g)
            for gameid, g in json.load(f).items()
        }

    logs = defaultdict(list)
    for gameid, game in games.items():
        for sample_idx in range(2, 17):
            simulated_trials = list()
            target_to_utterance_map = dict()
            for i, trial in enumerate(
                tqdm(
                    game.trials,
                    desc=f"Simulating {gameid}" if gameid else "Simulating",
                )
            ):
                previous_trials = deepcopy(simulated_trials)

                trials = [
                    *previous_trials,
                    Trial(target=trial.target),
                ]

                if trial.target in target_to_utterance_map:
                    trials[-1].message = target_to_utterance_map[trial.target]
                else:
                    target_to_utterance_map[trial.target] = speaker.generate(
                        RepeatedReferenceGame(context=game.context, trials=trials),
                        target_length=sample_idx,
                    )

                trials[-1].selection = listener.select(
                    RepeatedReferenceGame(context=game.context, trials=trials)
                )

                if trials[-1].selection is None:
                    trials[-1].correct = False
                elif trials[-1].selection == trial.target:
                    trials[-1].correct = True
                else:
                    trials[-1].correct = False

                logs[f"{gameid}_trialidx={i}_sampleidx={sample_idx}"].append(
                    RepeatedReferenceGame(
                        context=game.context, trials=trials
                    ).model_dump(mode="json")
                )

                simulated_trials.append(trials[-1])

        with open(results_save_dir / "reuse.json", "w") as f:
            json.dump(logs, f)
