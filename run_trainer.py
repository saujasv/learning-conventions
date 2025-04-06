from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser
from training.trainer import train
import itertools

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config, args = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    print("Script Arguments:", script_args)
    print("Training Arguments:", training_args)
    print("Model Configuration:", model_config)
    print("Remaining Arguments:", args)

    train(script_args, training_args, model_config, **dict(itertools.batched(args, 2)))
