from transformers.generation.logits_process import LogitsProcessor


class FIRELogitsWarper(LogitsProcessor):
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, input_ids, logits):
        import ipdb

        ipdb.set_trace()
        return logits
