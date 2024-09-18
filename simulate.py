import json
import random
from pathlib import Path
from PIL import Image
from game import Round, Feedback
from agents import GPTListener
from tqdm import tqdm


def simulate(rounds, speaker=None, listener=None, teacher_force=False, gameid=None):
    simulated_rounds = list()
    for i, round in enumerate(
        tqdm(rounds, desc=f"Simulating {gameid}" if gameid else "Simulating")
    ):
        previous_rounds = rounds[:i] if teacher_force else simulated_rounds[:i]

        if not speaker is None:
            message = speaker.generate(round.context, round.target, previous_rounds)
        else:
            message = round.message

        if not listener is None:
            selection = listener.select(round.context, message, previous_rounds)
        else:
            selection = round.selection

        if selection == None:
            feedback = Feedback.INVALID
        elif selection == round.target:
            feedback = Feedback.CORRECT
        else:
            feedback = Feedback.INCORRECT

        simulated_rounds.append(
            Round(round.context, round.target, message, selection, feedback)
        )

    return simulated_rounds


def load_games(games_path, images_path, shuffle_context=False):
    with open(games_path, "r") as f:
        data = json.load(f)

    games = dict()
    for gameid, rounds_data in data.items():
        rounds = list()
        context = [f"page-{chr(ord('A') + i)}.png" for i in range(12)]
        random.shuffle(context)

        for round_data in rounds_data:
            if shuffle_context:
                random.shuffle(context)
            if (
                round_data["role"] == "director"
                and round_data["trialNum"] == len(rounds) + 1
            ):
                round = Round(
                    [str(Path(images_path) / img) for img in context],
                    context.index(round_data["intendedObj"]),
                    round_data["contents"],
                    context.index(round_data["clickedObj"]),
                    Feedback.CORRECT if round_data["correct"] else Feedback.INCORRECT,
                )
                rounds.append(round)

        games[gameid] = rounds

    return games


def main(
    games_path,
    images_path,
    simulation_save_path,
    api_call_log_file,
    listener_images_once=True,
    listener_history=True,
    shuffle_context=False,
    teacher_force=False,
):
    games = load_games(games_path, images_path, shuffle_context=shuffle_context)
    listener = GPTListener(
        "gpt-4o-mini-2024-07-18",
        images_once=listener_images_once,
        previous_rounds=listener_history,
        response_save_path=api_call_log_file,
    )
    simulated_games = dict()
    for gameid, game in games.items():
        simulated_rounds = simulate(
            game, listener=listener, teacher_force=teacher_force, gameid=gameid
        )
        simulated_games[gameid] = simulated_rounds
        with open(simulation_save_path, "w") as f:
            json.dump(simulated_games, f)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
