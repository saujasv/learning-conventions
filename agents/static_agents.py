from typing import List, Optional
import jsonlines
from game import RepeatedReferenceGame, Trial


class ReplaySpeaker:
    def __init__(self, replay_data_path):
        with jsonlines.open(replay_data_path) as reader:
            replay_data = list(reader)

        self.game_state_to_samples = dict()
        for x in replay_data:
            context = ",".join(sorted(x["game"]["context"]))
            messages = [t["message"] for t in x["game"]["trials"]]
            self.game_state_to_samples[f"{context};{'|'.join(messages)}"] = [
                t["message"] for t in x["sampled_trials"]
            ]

    def batch_generate(
        self,
        game: RepeatedReferenceGame,
        num_return_sequences: Optional[int] = None,
        target_lengths: Optional[List[int]] = None,
    ):
        context = ",".join(sorted(game.context))
        messages = [t.message for t in game.trials if t.message]
        game_state_key = f"{context};{'|'.join(messages)}"
        if num_return_sequences:
            return [self.game_state_to_samples[game_state_key][:num_return_sequences]]
        else:
            return [self.game_state_to_samples[game_state_key]]


class OracleListener:
    def batch_score(
        self,
        games: list[RepeatedReferenceGame],
    ):
        return [
            {x: 1 if x == g.trials[-1].target else 0 for x in g.context} for g in games
        ]
