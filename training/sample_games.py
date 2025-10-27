from agents.game import RepeatedReferenceGame, Trial
import json
import inflect
import random
import jsonlines

inflect_engine = inflect.engine()


def sample_context(pool, num_items, sampling_algorithm="uniform", **kwargs):
    if sampling_algorithm == "uniform":
        return random.sample(pool, num_items)
    else:
        raise ValueError(f"Unknown sampling algorithm: {sampling_algorithm}")


def extract_cogen_annotations(save_path, *cogen_data_paths):
    games = list()
    for cogen_data_path in cogen_data_paths:
        with open(cogen_data_path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                games.extend(v)

    annotations = dict()

    for g in games:
        if g["selection"] != g["gt_target"]:
            continue

        if g["gt_target"] not in annotations:
            annotations[g["gt_target"]] = {"annotations": list()}
        annotations[g["gt_target"]]["annotations"].append(
            {"whole": {"wholeAnnotation": g["chat"]}}
        )

    with open(save_path, "w") as f:
        json.dump({f"{k}.jpg": v for k, v in annotations.items()}, f)


def sample_kilogram_description(annotations, min_parts=0, max_parts=0):
    sampled_annotation = random.choice(annotations)
    last_word = sampled_annotation["whole"]["wholeAnnotation"].strip().split(" ")[-1]
    if not inflect_engine.singular_noun(last_word):
        whole_description = inflect_engine.a(
            sampled_annotation["whole"]["wholeAnnotation"]
        )
    else:
        whole_description = sampled_annotation["whole"]["wholeAnnotation"]

    if max_parts is None or max_parts > 0:
        unique_parts = list(set(sampled_annotation["part"].values()))
        if max_parts is None:
            max_parts = len(unique_parts)

        if min_parts > max_parts:
            raise ValueError(f"min_parts ({min_parts}) > max_parts ({max_parts})")

        num_parts = random.randint(min_parts, max_parts)

        parts = random.sample(unique_parts, min(len(unique_parts), num_parts))
        part_phrases = list()
        for part in parts:
            last_word = part.split(" ")[-1]
            if not inflect_engine.singular_noun(last_word):
                part_phrase = inflect_engine.a(part)
            else:
                part_phrase = part
            part_phrases.append(part_phrase)

        if len(part_phrases) == 0:
            return whole_description

        description = (
            (
                whole_description
                + " with "
                + ", ".join(part_phrases[:-1])
                + ", and "
                + part_phrases[-1]
            )
            if len(part_phrases) > 1
            else whole_description + " with " + part_phrases[0]
        )
        return description
    else:
        return whole_description


def add_extension(filename, extension="png"):
    if "." in filename:
        return filename
    else:
        return filename + "." + extension


def sample_game(
    referents_pool,
    descriptions,
    num_referents,
    num_trials=None,
    num_blocks=None,
    block_structure=False,
    min_parts=0,
    max_parts=0,
    reuse_descriptions=False,
    oracle_listener=False,
):
    context = sample_context(referents_pool, num_referents)
    context_images = [add_extension(c) for c in context]
    trials = list()
    if block_structure:
        if num_blocks is None:
            raise ValueError("num_blocks must be specified if using block structure")
        for b in range(num_blocks):
            block_targets = random.sample(context, len(context))
            for target_referent in block_targets:
                description = sample_kilogram_description(
                    descriptions[target_referent]["annotations"], min_parts, max_parts
                )
                selection = random.choice(context)
                selection_image = add_extension(selection)
                trials.append(
                    Trial(
                        target=add_extension(target_referent),
                        message=description,
                        interpretation={
                            x: 1 if x == selection_image else 0 for x in context_images
                        },
                    )
                )
    else:
        if num_trials is None:
            raise ValueError(
                "num_trials must be specified if not using block structure"
            )

        if reuse_descriptions:
            selected_annotations = dict()

        for t in range(num_trials):
            target_referent = random.choice(context)

            if reuse_descriptions and not target_referent in selected_annotations:
                selected_annotations[target_referent] = [
                    random.choice(descriptions[target_referent]["annotations"])
                ]

            if reuse_descriptions:
                description = sample_kilogram_description(
                    selected_annotations[target_referent], min_parts, max_parts
                )
            else:
                description = sample_kilogram_description(
                    descriptions[target_referent]["annotations"], min_parts, max_parts
                )

            selection = target_referent if oracle_listener else random.choice(context)
            selection_image = add_extension(selection)
            trials.append(
                Trial(
                    target=add_extension(target_referent),
                    message=description,
                    interpretation={
                        x: 1 if x == selection_image else 0 for x in context_images
                    },
                )
            )

    return RepeatedReferenceGame(context=context_images, trials=trials)


def sample_training_games(
    kilogram_data_path,
    save_path,
    num_games,
    num_referents,
    incremental=False,
    num_trials=None,
    num_blocks=None,
    block_structure=False,
    min_parts=0,
    max_parts=0,
    reuse_descriptions=False,
    oracle_listener_history=False,
    oracle_listener_response=False,
):
    with open(kilogram_data_path) as f:
        kilogram_descriptions = json.load(f)

    referents_pool = list(kilogram_descriptions.keys())
    games = list()
    for _ in range(num_games):
        game = sample_game(
            referents_pool,
            kilogram_descriptions,
            num_referents,
            num_trials,
            num_blocks,
            block_structure,
            min_parts,
            max_parts,
            reuse_descriptions,
            oracle_listener_history,
        )

        if incremental:
            for i, t in enumerate(game.trials):
                if oracle_listener_response:
                    interpretation = {
                        x: 1 if x == t.target else 0 for x in game.context
                    }
                else:
                    selection = random.choice(game.context)
                    selection_image = add_extension(selection)
                    interpretation = {
                        x: 1 if x == selection_image else 0 for x in game.context
                    }

                incremental_game = RepeatedReferenceGame(
                    context=game.context,
                    trials=[
                        *game.trials[:i],
                        Trial(
                            target=t.target,
                            message=t.message,
                            interpretation=interpretation,
                        ),
                    ],
                )
                games.append(incremental_game)
        else:
            if oracle_listener_response != oracle_listener_history:
                raise ValueError(
                    "oracle_listener_response and oracle_listener_history must be the same if games are not incremental"
                )
            games.append(game)

    with jsonlines.open(save_path, "w") as writer:
        writer.write_all(map(lambda g: g.model_dump(mode="json"), games))
