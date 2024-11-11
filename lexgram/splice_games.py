from collections import Counter, defaultdict
import random
import json

from game import RepeatedReferenceGame, Trial


def get_message_at_block_index(game, target, block_index):
    block_counter = Counter()
    for trial in game.trials:
        block_counter.update([trial.target])
        if trial.target == target and block_counter[trial.target] - 1 == block_index:
            return trial.message


def splice_games(games, splice_block_index):
    img2game = defaultdict(list)
    for gameid, game in games.items():
        for img in game.context:
            img2game[img].append(gameid)

    spliced_games = dict()
    for gameid, game in games.items():
        splice_games = {
            img: (
                random.choice(
                    [alternate for alternate in img2game[img] if alternate != gameid]
                )
                if len(img2game[img]) > 1
                else img2game[img][0]
            )
            for img in game.context
        }

        spliced_trials = []
        block_counter = Counter()
        for trial in game.trials:
            block_counter.update([trial.target])
            if block_counter[trial.target] - 1 >= splice_block_index:
                message = get_message_at_block_index(
                    games[splice_games[trial.target]],
                    trial.target,
                    block_counter[trial.target] - 1,
                )
            else:
                message = trial.message

            spliced_trials.append(
                Trial(
                    target=trial.target,
                    message=message,
                    selection=trial.selection,
                    correct=trial.correct,
                )
            )

        spliced_games[gameid] = RepeatedReferenceGame(
            context=game.context,
            trials=spliced_trials,
        )

    return spliced_games


def main(games_path, splice_block_index, save_path):
    with open(games_path) as f:
        games = dict()
        for gameid, game in json.load(f).items():
            games[gameid] = RepeatedReferenceGame.model_validate_json(json.dumps(game))

    spliced_games = splice_games(games, splice_block_index)
    with open(save_path, "w") as f:
        json.dump(
            {
                gameid: game.model_dump(mode="json")
                for gameid, game in spliced_games.items()
            },
            f,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
