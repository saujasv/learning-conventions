from typing import List, Optional
import jsonlines
from .game import RepeatedReferenceGame, Trial


class ReplaySpeaker:
    def __init__(self, replay_data_path):
        with jsonlines.open(replay_data_path) as reader:
            replay_data = list(reader)

        self.game_id_to_samples = dict()
        for x in replay_data:
            if x["game_id"] not in self.game_id_to_samples:
                self.game_id_to_samples[x["game_id"]] = list()
            self.game_id_to_samples[x["game_id"]].append(
                [t["message"] for t in x["sampled_trials"]]
            )

    def batch_generate(
        self,
        games: list[RepeatedReferenceGame],
        num_return_sequences: Optional[int] = None,
        target_lengths: Optional[List[int]] = None,
        context_ids: Optional[str] = None,
    ):
        responses = list()
        for game, ctx_id in zip(games, context_ids):
            if num_return_sequences:
                responses.append(
                    self.game_id_to_samples[ctx_id][len(game.trials) - 1][
                        :num_return_sequences
                    ]
                )
            else:
                responses.append(self.game_id_to_samples[ctx_id][len(game.trials) - 1])
        return responses


class OracleListener:
    def batch_score(
        self,
        games: list[RepeatedReferenceGame],
    ):
        return [
            {x: 1 if x == g.trials[-1].target else 0 for x in g.context} for g in games
        ]
