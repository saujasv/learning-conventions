from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser
from training.trainer_unsloth import train_unsloth
from training.configs import DataConfig, LoraConfig
import itertools

if __name__ == "__main__":
    parser = TrlParser(
        (ScriptArguments, SFTConfig, ModelConfig, DataConfig, LoraConfig)
    )
    script_args, training_args, model_config, data_config, lora_config, args = (
        parser.parse_args_and_config(return_remaining_strings=True)
    )

    train_unsloth(script_args, training_args, model_config, data_config, lora_config)
