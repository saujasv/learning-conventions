from game import RepeatedReferenceGame, Trial
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


def sample_kilogram_description(annotations, min_parts=0, max_parts=0):
    sampled_annotation = random.choice(annotations)
    last_word = sampled_annotation["whole"]["wholeAnnotation"].split(" ")[-1]
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


def sample_game(
    referents_pool,
    descriptions,
    num_referents,
    num_trials=None,
    num_blocks=None,
    block_structure=False,
    min_parts=0,
    max_parts=0,
):
    context = sample_context(referents_pool, num_referents)
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
                trials.append(
                    Trial(
                        target=f"{target_referent}.png",
                        message=description,
                        selection=f"{selection}.png",
                        correct=target_referent == selection,
                    )
                )
    else:
        if num_trials is None:
            raise ValueError(
                "num_trials must be specified if not using block structure"
            )
        for t in range(num_trials):
            target_referent = random.choice(context)

            description = sample_kilogram_description(
                descriptions[target_referent]["annotations"], min_parts, max_parts
            )
            selection = random.choice(context)
            trials.append(
                Trial(
                    target=f"{target_referent}.png",
                    message=description,
                    selection=f"{selection}.png",
                    correct=target_referent == selection,
                )
            )

    return RepeatedReferenceGame(context=[f"{c}.png" for c in context], trials=trials)


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
        )

        if incremental:
            for i, t in enumerate(game.trials):
                incremental_game = RepeatedReferenceGame(
                    context=game.context, trials=game.trials[: i + 1]
                )
                games.append(incremental_game)
        else:
            games.append(game)

    with jsonlines.open(save_path, "w") as writer:
        writer.write_all(map(lambda g: g.model_dump(mode="json"), games))


if __name__ == "__main__":
    import fire

    fire.Fire(sample_training_games)
