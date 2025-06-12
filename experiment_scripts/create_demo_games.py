import json
from pathlib import Path
from experiment_scripts.create_drop_words_samples import (
    score_messages,
    get_masked_messages,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from game import RepeatedReferenceGame, Trial
import numpy as np
import spacy
import torch

nlp = spacy.load("en_core_web_sm")


def prepare_demonstration_game_coco(game: RepeatedReferenceGame, COCO_CAPTIONS: dict):
    idx = np.random.permutation(len(game.context))
    selected_trials = [
        Trial(
            target=x,
            message=COCO_CAPTIONS[x][0],
            interpretation={y: 0 if y == x else -float("inf") for y in game.context},
        )
        for i, x in zip(idx, game.context)
    ]

    demo_game_extracted = RepeatedReferenceGame(
        context=game.context, trials=selected_trials
    )

    return RepeatedReferenceGame(
        context=[
            f"val2014/{x}"
            for x in [
                *demo_game_extracted.context,
            ]
        ],
        trials=[
            Trial(
                target=f"val2014/{t.target}",
                message=t.message,
                interpretation={f"val2014/{k}": v for k, v in t.interpretation.items()},
            )
            for t in demo_game_extracted.trials
        ],
    )


def get_ablated_caption(caption: str, model, tokenizer):
    processed_message = [tok for tok in nlp(caption) if not tok.is_punct]
    masked_messages = get_masked_messages(
        processed_message, 2 ** len(processed_message) - 2
    )
    _, scores = score_messages(masked_messages, model, tokenizer)
    # Get top message based on normalized scores
    top_message_idx = np.argmax(scores)
    return masked_messages[top_message_idx]


def prepare_demonstration_game_coco_with_ablated_captions(
    game: RepeatedReferenceGame, COCO_CAPTIONS: dict, model, tokenizer
):
    selected_trials = list()
    for i, x in enumerate(game.context):
        if i < len(game.context) // 2:
            selected_trials.append(
                Trial(
                    target=x,
                    message=COCO_CAPTIONS[x][0],
                    interpretation={
                        y: 0 if y == x else -float("inf") for y in game.context
                    },
                )
            )
        else:
            selected_trials.append(
                Trial(
                    target=x,
                    message=get_ablated_caption(COCO_CAPTIONS[x][0], model, tokenizer),
                    interpretation={
                        y: 0 if y == x else -float("inf") for y in game.context
                    },
                )
            )

    demo_game_extracted = RepeatedReferenceGame(
        context=game.context,
        trials=[
            selected_trials[i] for i in np.random.permutation(len(selected_trials))
        ],
    )

    return RepeatedReferenceGame(
        context=[
            f"val2014/{x}"
            for x in [
                *demo_game_extracted.context,
            ]
        ],
        trials=[
            Trial(
                target=f"val2014/{t.target}",
                message=t.message,
                interpretation={f"val2014/{k}": v for k, v in t.interpretation.items()},
            )
            for t in demo_game_extracted.trials
        ],
    )


def prepare_demonstration_game(game: RepeatedReferenceGame):
    trial_idx_by_target = {x: list() for x in game.context}
    for trial_idx, trial in enumerate(game.trials):
        trial_idx_by_target[trial.target].append(trial_idx)

    idx = np.random.permutation(len(game.context))
    selected_trials = [
        game.trials[trial_idx_by_target[x][i]] for i, x in zip(idx, game.context)
    ]
    demo_game_extracted = RepeatedReferenceGame(
        context=game.context, trials=selected_trials
    )
    return RepeatedReferenceGame(
        context=[
            f"val2014/{x}"
            for x in [
                *demo_game_extracted.context,
            ]
        ],
        trials=[
            Trial(
                target=f"val2014/{t.target}",
                message=t.message,
                interpretation={f"val2014/{k}": v for k, v in t.interpretation.items()},
            )
            for t in demo_game_extracted.trials
        ],
    )


def main(human_human_games_file, coco_annotations_file, output_path):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()

    with open(coco_annotations_file, "r") as f:
        coco_data = json.load(f)

    # Build a mapping from image ID to filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    # Build the dictionary mapping filename to list of captions
    COCO_CAPTIONS = {}

    # Iterate through all annotations and group captions by image
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]

        # Get the filename for this image
        filename = image_id_to_filename[image_id]

        # Add caption to the list for this filename
        if filename not in COCO_CAPTIONS:
            COCO_CAPTIONS[filename] = []
        COCO_CAPTIONS[filename].append(caption)

    with open(human_human_games_file, "r") as f:
        human_human_games = json.load(f)

    selected_game_ids = list(human_human_games.keys())[:5]

    output_path = Path(output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for game_id in selected_game_ids:
        game = RepeatedReferenceGame.model_validate(human_human_games[game_id])
        with open(output_path / f"human_human_{game_id}.json", "w") as f:
            json.dump(prepare_demonstration_game(game).model_dump(), f)

        with open(output_path / f"coco_only_{game_id}.json", "w") as f:
            json.dump(
                prepare_demonstration_game_coco(game, COCO_CAPTIONS).model_dump(), f
            )

        with open(output_path / f"coco_with_ablated_{game_id}.json", "w") as f:
            json.dump(
                prepare_demonstration_game_coco_with_ablated_captions(
                    game, COCO_CAPTIONS, model, tokenizer
                ).model_dump(),
                f,
            )
