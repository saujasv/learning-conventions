import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from game import RepeatedReferenceGame, Trial
from copy import deepcopy
import numpy as np
from agents.hf_speakers import GenerateSpeaker
from agents.hf_listeners import ScoringListener
from agents.prompts import (
    SPEAKER_SYSTEM_PROMPT_BASIC,
    SPEAKER_USER_PROMPT_TARGET_BASIC,
    SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC,
)
from tqdm import tqdm
import stanza
from Levenshtein import distance as edit_distance
import itertools
import jsonlines
import json

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma", use_gpu=False)

TAGSET = ["NOUN", "ADJ", "VERB", "ADV", "PROPN", "NUM", "PRON", "ADP"]


def wnr(s1, s2):
    s1_content = [t["lemma"].lower() for t in s1 if t["upos"] in TAGSET]
    s2_content = [t["lemma"].lower() for t in s2 if t["upos"] in TAGSET]
    return edit_distance(s1_content, s2_content, weights=(1, 0, 1)) / len(s1)


def process_message(message):
    return list(
        itertools.chain.from_iterable(
            [[w.to_dict() for w in s.words] for s in nlp(message).sentences]
        )
    )


def select_next_message(
    samples, last_message, listener_interpretations, target, policy
):
    selections = [
        max(interpretation, key=interpretation.get)
        for interpretation in listener_interpretations
    ]
    correct_idx = [
        idx for idx, selection in enumerate(selections) if selection == target
    ]
    if policy == "random":
        # Randomly select a sample
        idx = np.random.choice(len(samples))
    elif policy == "random_correct":
        # Randomly select a sample that is correct
        # if there are no correct samples, select a random sample
        if len(correct_idx) == 0:
            idx = np.random.choice(len(samples))
        else:
            idx = np.random.choice(correct_idx)
    elif policy == "shortest_correct":
        # Select the shortest correct sample
        # if there are no correct samples, select a random sample
        if len(correct_idx) == 0:
            idx = np.random.choice(len(samples))
        else:
            lengths = [
                (i, len(process_message(sample)))
                for i, sample in enumerate(samples)
                if i in correct_idx
            ]
            idx = min(lengths, key=lambda x: x[1])[0]
    elif policy == "min_wnr":
        # Select the sample with the minimum WNR to the last message
        # if there are no correct samples, select a random sample
        if last_message is None:
            idx = np.random.choice(correct_idx) if len(correct_idx) > 0 else 0
        else:
            if len(correct_idx) == 0:
                wnr_values = [
                    (i, wnr(process_message(sample), process_message(last_message)))
                    for i, sample in enumerate(samples)
                ]

            else:
                wnr_values = [
                    (i, wnr(process_message(sample), process_message(last_message)))
                    for i, sample in enumerate(samples)
                    if i in correct_idx
                ]

            idx = min(wnr_values, key=lambda x: x[1])[0]

    next_message = samples[idx]
    interpretation = listener_interpretations[idx]

    return next_message, interpretation


def simulate_game(
    task: dict,
    speaker: GenerateSpeaker,
    listener: ScoringListener,
    target_lengths: list,
    next_message_policy: str,
):
    game = RepeatedReferenceGame.model_validate(task["game"])
    simulated_game = RepeatedReferenceGame(context=game.context, trials=[])
    messages_by_target = {t: list() for t in game.context}
    game_data = {
        "samples": list(),
        "interpretations": list(),
        "next_message": list(),
        "next_message_interpretation": list(),
        "target_lengths": list(),
        "processed_samples": list(),
        "processed_previous_messages": list(),
    }

    for trial_idx, trial in enumerate(tqdm(game.trials)):
        if speaker:
            samples = speaker.batch_generate(
                RepeatedReferenceGame(
                    context=simulated_game.context,
                    trials=[*simulated_game.trials, Trial(target=trial.target)],
                ),
                target_lengths=target_lengths,
            )
        else:
            samples = task["samples"][trial_idx]

        processed_samples = [process_message(sample) for sample in samples]

        listener_interpretations = listener.batch_score(
            [
                RepeatedReferenceGame(
                    context=simulated_game.context,
                    trials=[
                        *simulated_game.trials,
                        Trial(target=trial.target, message=sample),
                    ],
                )
                for sample in samples
            ],
        )

        if speaker:
            next_message, interpretation = select_next_message(
                samples,
                (
                    messages_by_target[trial.target][-1]
                    if len(messages_by_target[trial.target]) > 0
                    else None
                ),
                listener_interpretations,
                trial.target,
                next_message_policy,
            )
        else:
            next_message, interpretation = (
                task["next_message"][trial_idx],
                listener_interpretations[
                    task["samples"][trial_idx].index(task["next_message"][trial_idx])
                ],
            )

        selection = max(interpretation, key=interpretation.get)

        simulated_game.trials.append(
            Trial(
                target=trial.target,
                message=next_message,
                selection=selection,
                correct=selection == trial.target,
            )
        )

        game_data["samples"].append(samples)
        game_data["interpretations"].append(listener_interpretations)
        game_data["next_message"].append(next_message)
        game_data["next_message_interpretation"].append(interpretation)
        game_data["target_lengths"].append(target_lengths)
        game_data["processed_samples"].append(processed_samples)
        game_data["processed_previous_messages"].append(
            [process_message(m) for m in messages_by_target[trial.target]]
        )

        messages_by_target[trial.target].append(next_message)

    return {"game": simulated_game.model_dump(mode="json"), **game_data}


def main(config):
    speaker_config = config.get("speaker", {})
    speaker_model_name_or_path = speaker_config.get("model_name_or_path", None)

    listener_config = config.get("listener", {})
    listener_model_name_or_path = listener_config.get("model_name_or_path", None)

    if speaker_model_name_or_path == listener_model_name_or_path:
        processor = AutoProcessor.from_pretrained(speaker_model_name_or_path)
        model = AutoModelForImageTextToText.from_pretrained(
            speaker_model_name_or_path,
            torch_dtype=speaker_config.get("torch_dtype", "bfloat16"),
            device_map=speaker_config.get("device_map", "auto"),
            attn_implementation=speaker_config.get(
                "attn_implementation", "flash_attention_2"
            ),
        )

        speaker_processor = processor
        speaker_model = model
        listener_processor = processor
        listener_model = model
    else:
        if speaker_model_name_or_path != "replay":
            speaker_processor = AutoProcessor.from_pretrained(
                speaker_model_name_or_path
            )
            speaker_model = AutoModelForImageTextToText.from_pretrained(
                speaker_model_name_or_path,
                torch_dtype=speaker_config.get("torch_dtype", "bfloat16"),
                device_map=speaker_config.get("device_map", "auto"),
                attn_implementation=speaker_config.get(
                    "attn_implementation", "flash_attention_2"
                ),
            )

        listener_processor = AutoProcessor.from_pretrained(listener_model_name_or_path)
        listener_model = AutoModelForImageTextToText.from_pretrained(
            listener_model_name_or_path,
            torch_dtype=listener_config.get("torch_dtype", "bfloat16"),
            device_map=listener_config.get("device_map", "auto"),
            attn_implementation=listener_config.get(
                "attn_implementation", "flash_attention_2"
            ),
        )

    if speaker_model_name_or_path != "replay":
        speaker = GenerateSpeaker(
            speaker_model,
            speaker_processor,
            image_base_path=speaker_config.get("image_base_path", "."),
            context_presentation=speaker_config.get("context_presentation", "once"),
            feedback_label=True,
            system_prompt_template=SPEAKER_SYSTEM_PROMPT_BASIC,
            user_prompt=SPEAKER_USER_PROMPT_PHOTOGRAPHS_BASIC,
            target_prompt_template=SPEAKER_USER_PROMPT_TARGET_BASIC,
            max_image_size=speaker_config.get("max_image_size", None),
            generation_config=speaker_config.get(
                "generation_config",
                {
                    "do_sample": True,
                    "max_new_tokens": 64,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            ),
            chat_template_file=speaker_config.get("chat_template_file", None),
        )
    else:
        speaker = None

    listener = ScoringListener(
        listener_model,
        listener_processor,
        image_base_path=listener_config.get("image_base_path", "."),
        context_presentation=listener_config.get("context_presentation", "once"),
        feedback_label=True,
        max_image_size=listener_config.get("max_image_size", None),
        chat_template_file=listener_config.get("chat_template_file", None),
    )

    with jsonlines.open(config.get("tasks_file")) as reader:
        tasks = list(reader)

    outputs = list()
    for task in tasks:
        output = simulate_game(
            task,
            speaker,
            listener,
            config.get("target_lengths", config.get("target_lengths", [None])),
            config.get("next_message_policy", "random_correct"),
        )
        outputs.append(output)
        with jsonlines.open(config.get("output_file"), mode="w") as writer:
            writer.write_all(outputs)


def create_remention_games(input_games_path, output_games_path):
    with jsonlines.open(input_games_path) as reader:
        input_games = list(reader)

    output_games = list()
    for x in input_games:
        game = RepeatedReferenceGame.model_validate(x["game"])
        first_mentions = dict()
        for trial in game.trials:
            if trial.target not in first_mentions:
                first_mentions[trial.target] = trial.message

        remention_game = RepeatedReferenceGame(
            context=game.context,
            trials=[
                Trial(
                    target=t.target,
                    message=first_mentions[t.target],
                    selection=t.target,
                    correct=True,
                )
                for t in game.trials
            ],
        )

        next_messages = [first_mentions[t.target] for t in game.trials]

        output_games.append(
            {
                "game": remention_game.model_dump(mode="json"),
                "next_message": next_messages,
                "samples": [[n, *s] for n, s in zip(next_messages, x["samples"])],
                "target_lengths": x["target_lengths"],
            }
        )

    with jsonlines.open(output_games_path, mode="w") as writer:
        writer.write_all(output_games)


def run(config_idx):
    with open("experiment_scripts/listener_belief_configs.json") as f:
        CONFIGS = json.load(f)

    config = CONFIGS[config_idx]

    print(config)
    main(config)
