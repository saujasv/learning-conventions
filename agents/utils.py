import numpy as np
import torch
from transformers.generation.logits_process import LogitsProcessor


class FIRELogitsWarper(LogitsProcessor):
    def __init__(
        self,
        num_return_sequences,
        fire_temperature: float = 2.0,
        standard_temperature: float = 0.3,
    ):
        self.fire = torch.ones((num_return_sequences,), dtype=torch.bool)
        self.fire_temperature = fire_temperature
        self.standard_temperature = standard_temperature

    def __call__(self, input_ids, logits):
        non_inf_mask = ~torch.isinf(logits)
        num_non_inf = non_inf_mask.sum(dim=-1)
        output = torch.where(
            self.fire.to(logits.device).unsqueeze(1),
            logits / self.fire_temperature,
            logits / self.standard_temperature,
        )

        self.fire = self.fire & ~(num_non_inf.to(self.fire.device) > 0)

        return output


class TemperatureDecayLogitsWarper(LogitsProcessor):
    def __init__(self, scale: float = 2.0, target_temperature: float = 0.3):
        self.scale = scale
        self.target_temperature = target_temperature
        self.idx = 0

    def __call__(self, input_ids, logits):
        temp = self.scale * np.exp(-self.idx) + self.target_temperature
        self.idx += 1
        return logits / temp
