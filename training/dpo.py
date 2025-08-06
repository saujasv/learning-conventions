import torch
from trl import (
    get_peft_config,
    ModelConfig,
    ScriptArguments,
    DPOConfig,
    TrlParser,
)
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Idefics3Processor,
    PixtralProcessor,
)
import pandas as pd
from game import RepeatedReferenceGame, Trial
from training.dpo_trainer import DPOTrainer
from training.model_constants import get_lora_target_modules
from agents.hf_speakers import GenerateSpeaker
from agents.args import AgentArguments
from pathlib import Path
from datasets import Dataset


def make_game_preference_pair(x, speaker):
    game_history = RepeatedReferenceGame.model_validate(x["game"])
    win_trial = Trial.model_validate(x["preference_pairs"][0])
    lose_trial = Trial.model_validate(x["preference_pairs"][1])
    win_messages, images = speaker.construct_prompt_messages(
        RepeatedReferenceGame(
            context=game_history.context, trials=[*game_history.trials, win_trial]
        ),
        exclude_feedback_on_last=True,
        random_seed=412,
    )
    lose_messages, _ = speaker.construct_prompt_messages(
        RepeatedReferenceGame(
            context=game_history.context, trials=[*game_history.trials, lose_trial]
        ),
        exclude_feedback_on_last=True,
        random_seed=412,
    )

    assert win_messages[:-1] == lose_messages[:-1], "Messages do not match"

    return {
        "prompt": win_messages[:-1],
        "chosen": [win_messages[-1]],
        "rejected": [lose_messages[-1]],
        "images": images,
    }


def train():
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, AgentArguments))
    script_args, training_args, model_config, agent_args, _ = (
        parser.parse_args_and_config(return_remaining_strings=True)
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if isinstance(processor, Idefics3Processor):
        processor.image_processor.do_image_splitting = False

    processor.tokenizer.padding_side = "left"

    if isinstance(processor, PixtralProcessor):
        attn_implementation = {"text_config": "flash_attention_2"}
    else:
        attn_implementation = "flash_attention_2"

    if agent_args.chat_template_file is not None:
        with open(agent_args.chat_template_file, "r") as f:
            processor.chat_template = f.read().strip()
            processor.tokenizer.chat_template = processor.chat_template

    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    model_config.lora_target_modules = get_lora_target_modules(
        model.config, model_config.lora_target_modules
    )

    ################
    # Dataset
    ################
    agent = GenerateSpeaker(
        model_type=agent_args.model_type,
        model=None,
        processor=processor,
        context_presentation=agent_args.context_presentation,
        feedback_label=agent_args.feedback_label,
        chat_template_file=agent_args.chat_template_file,
        demonstration_game=agent_args.demonstration_game,
        max_image_size=agent_args.max_image_size,
    )

    train_df = pd.read_json(
        Path(script_args.dataset_name) / "train.jsonl", lines=True, orient="records"
    )

    train_df = train_df.explode("preference_pairs")
    train_df = train_df[~train_df["preference_pairs"].isnull()]

    train_df[["prompt", "chosen", "rejected", "images"]] = train_df.apply(
        lambda x: make_game_preference_pair(x, agent), axis=1, result_type="expand"
    )
    train_df = train_df.drop(columns=["game", "preference_pairs", "sampled_trials"])

    train_dataset = Dataset.from_pandas(train_df)

    validation_df = pd.read_json(
        Path(script_args.dataset_name) / "validation.jsonl",
        lines=True,
        orient="records",
    )
    validation_df = validation_df.explode("preference_pairs")
    validation_df = validation_df[~validation_df["preference_pairs"].isnull()]
    validation_df[["prompt", "chosen", "rejected", "images"]] = validation_df.apply(
        lambda x: make_game_preference_pair(x, agent), axis=1, result_type="expand"
    )
    validation_df = validation_df.drop(
        columns=["game", "preference_pairs", "sampled_trials"]
    )
    validation_dataset = Dataset.from_pandas(validation_df)

    ################
    # Training
    ################

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(
            validation_dataset if training_args.eval_strategy != "no" else None
        ),
        processing_class=processor,
        peft_config=get_peft_config(model_config),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
