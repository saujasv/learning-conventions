import json
import pandas as pd
import random
from game import RepeatedReferenceGame, Trial


def convert_experiment1_to_icca(game_df, no_control=False):
    repeated_targets = game_df[~game_df.controlled].target.unique().tolist()
    control_targets = (
        game_df[game_df.controlled].sort_values(by="block").target.tolist()
    )

    if no_control:
        context = repeated_targets
    else:
        context = repeated_targets + control_targets

    repeated_reference_game = RepeatedReferenceGame(
        context=[x.replace(".svg", ".png") for x in context]
    )

    selected_rows = game_df[~game_df.controlled] if no_control else game_df

    for i, row in selected_rows.sort_values(by="trial_index").iterrows():
        target = (
            row.target.replace(".svg", ".png") if isinstance(row.target, str) else None
        )

        selection = (
            row.response.replace(".svg", ".png")
            if isinstance(row.response, str)
            else None
        )
        if not selection in repeated_reference_game.context:
            selection = None

        message = row.description if isinstance(row.description, str) else None
        correct = target == selection if not target is None else False

        if not target in repeated_reference_game.context:
            continue

        trial = Trial(
            target=target,
            message=message,
            selection=selection,
            correct=correct,
        )

        repeated_reference_game.trials.append(trial)

    return repeated_reference_game


def main(experiments_data, save_path, no_control=False):
    df = pd.read_csv(experiments_data)

    gameids = df.game_id.unique()

    icca_games = dict()
    for gameid in gameids:
        icca_game = convert_experiment1_to_icca(
            df[df.game_id == gameid], no_control
        ).model_dump(mode="json")
        icca_games[gameid] = icca_game

    with open(save_path, "w") as f:
        json.dump(icca_games, f)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
