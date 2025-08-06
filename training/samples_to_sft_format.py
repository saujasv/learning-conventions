import pandas as pd
from game import Trial


def convert_to_sft_format(input_file, output_file):
    df = pd.read_json(input_file, lines=True)

    df.sort_values(by="game", key=lambda x: [len(g["trials"]) for g in x], inplace=True)
    games_df = df.groupby("game_id").agg("last")
    games_df["sft_game"] = games_df.apply(
        lambda x: {
            "context": x["game"]["context"],
            "trials": [*x["game"]["trials"], x["sampled_trials"][0]],
        },
        axis=1,
    )

    games_df["sft_game"].to_json(output_file, orient="records", lines=True)


def convert_to_sft_format_incremental(input_file, output_file):
    games_df = pd.read_json(input_file, lines=True)

    games_df["sft_game"] = games_df.apply(
        lambda x: [
            {
                "context": x["game"]["context"],
                "trials": [*x["game"]["trials"], t],
            }
            for t in x["sampled_trials"]
            if Trial.model_validate(t).get_correct()
        ],
        axis=1,
    )
    games_df = games_df.explode("sft_game")
    games_df = games_df[games_df.sft_game.notna()]

    games_df["sft_game"].to_json(output_file, orient="records", lines=True)
