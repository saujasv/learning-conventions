from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)
import jsonlines
from game import RepeatedReferenceGame, Trial
from agents.hf_listeners import ScoringListener
from tqdm import tqdm
import numpy as np
import torch
from pydantic_core import to_jsonable_python
import itertools


import spacy

nlp = spacy.load("en_core_web_sm")


def get_masked_messages(processed_message, num_masked_messages):
    mask_ids = (
        np.random.choice(
            2 ** len(processed_message) - 2,
            min(num_masked_messages, 2 ** len(processed_message) - 2),
            replace=False,
        )
        + 1
    )
    masks = [
        list(map(lambda s: bool(int(s)), bin(m)[2:].zfill(len(processed_message))))
        for m in mask_ids
    ]
    return [
        "".join([tok.text_with_ws for i, tok in enumerate(processed_message) if m[i]])
        for m in masks
    ]


def score_messages(masked_messages, model, tokenizer):
    all_sequence_log_probs = list()
    all_normalized_sequence_log_probs = list()

    for batch in itertools.batched(masked_messages, 2048):
        processed_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            outputs = model(**processed_inputs)
        logits = outputs.logits
        # Get log probabilities by applying log softmax to logits
        log_probs = logits.log_softmax(dim=-1)
        # Get sequence log probability by summing token log probs, excluding first (BOS) token
        sequence_log_probs = (
            log_probs[:, :-1]
            .gather(2, processed_inputs["input_ids"][:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        sequence_log_probs = sequence_log_probs.masked_fill(
            processed_inputs["attention_mask"][:, 1:].bool().logical_not(), 0
        )
        sequence_log_probs = sequence_log_probs.sum(dim=-1)
        normalized_sequence_log_probs = sequence_log_probs / processed_inputs[
            "attention_mask"
        ].sum(dim=-1)
        all_sequence_log_probs.extend(sequence_log_probs.tolist())
        all_normalized_sequence_log_probs.extend(normalized_sequence_log_probs.tolist())
    return all_sequence_log_probs, all_normalized_sequence_log_probs


def create_games(
    source_samples_file, save_path, num_samples_per_game, num_masked_messages
):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()

    with jsonlines.open(source_samples_file) as reader:
        data = list(reader)

    games = dict()
    for x in data:
        if x["game_id"] in games:
            games[x["game_id"]].append(x)
        else:
            games[x["game_id"]] = [x]

    data = list()
    for game_id, records in tqdm(games.items(), desc="Creating games"):
        game = records[-1]["game"]
        trials_by_target = {x: list() for x in game["context"]}
        new_game = RepeatedReferenceGame(context=game["context"], trials=[])
        for trial, x in zip(game["trials"], records):
            scores = None
            normalized_scores = None
            if len(trials_by_target[trial["target"]]) >= 1:
                processed_message = [
                    tok
                    for tok in nlp(trials_by_target[trial["target"]][-1].message)
                    if not tok.is_punct
                ]
                masked_messages = get_masked_messages(
                    processed_message, num_masked_messages
                )
                if len(masked_messages) > 0:
                    scores, normalized_scores = score_messages(
                        masked_messages, model, tokenizer
                    )
                    # Get top k messages based on normalized scores
                    top_k_indices = np.argsort(normalized_scores)[
                        -num_samples_per_game:
                    ]
                    top_k_messages = [masked_messages[i] for i in top_k_indices]

                    # Add trials with top scoring messages
                    sampled_trials = [
                        Trial(
                            target=trial["target"],
                            message=message,
                            interpretation=None,
                        )
                        for message in top_k_messages
                    ]

                    next_trial = sampled_trials[0]
                else:
                    next_trial = Trial.model_validate(x["sampled_trials"][0])
            else:
                sampled_trials = [Trial.model_validate(t) for t in x["sampled_trials"]]
                correct_trials = [t for t in sampled_trials if t.get_correct()]
                if len(correct_trials) == 0:
                    next_trial = sampled_trials[0]
                else:
                    next_trial = correct_trials[0]

            new_game = RepeatedReferenceGame(
                context=game["context"],
                trials=[*new_game.trials, next_trial],
            )
            trials_by_target[next_trial.target].append(next_trial)

            data.append(
                {
                    "game_id": game_id,
                    "game": new_game,
                    "sampled_trials": sampled_trials,
                    "scores": scores,
                    "normalized_scores": normalized_scores,
                }
            )

        with jsonlines.open(save_path, "w") as writer:
            writer.write_all(to_jsonable_python(data))


def score_games(games_file, save_path):
    processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-3-12b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()

    listener = ScoringListener(
        model,
        processor,
        "last_shuffle",
        True,
        chat_template_file="agents/gemma-3-chat-template.jinja",
    )

    with jsonlines.open(games_file) as reader:
        data = list(reader)

    results = list()
    for x in data:
        game = RepeatedReferenceGame.model_validate(x["game"])
        scores = listener.batch_score(
            [
                RepeatedReferenceGame(
                    context=game.context, trials=[*game.trials, Trial.model_validate(t)]
                )
                for t in x["sampled_trials"]
            ]
        )
        trials = [
            Trial(target=t["target"], message=t["message"], interpretation=s)
            for t, s in zip(x["sampled_trials"], scores)
        ]
        results.append(
            {
                "game_id": x["game_id"],
                "game": game,
                "sampled_trials": trials,
            }
        )

        with jsonlines.open(save_path, "w") as writer:
            writer.write_all(to_jsonable_python(results))
