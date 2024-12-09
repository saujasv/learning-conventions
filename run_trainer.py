# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
pip install pillow

# Tested on 8x H100 GPUs
accelerate launch
    --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir sft-llava-1.5-7b-hf \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing

For LLaVA-NeXT, use: (requires transformers>=4.45)
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf

For meta-llama/Llama-3.2-11B-Vision-Instruct, use: (requires transformers>=4.45.1)
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct
"""

from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser
from training.trainer import train
import itertools

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config, args = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    train(script_args, training_args, model_config, **dict(itertools.batched(args, 2)))
