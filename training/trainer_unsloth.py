import torch
from trl import SFTTrainer, SFTConfig, get_peft_config
from unsloth import FastVisionModel, is_bf16_supported
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset
from training.collator import RepeatedReferenceGameCollator
from agents import GenerateSpeaker, GenerateListener
from pathlib import Path


def train_unsloth(
    script_args,
    training_args,
    model_config,
    data_config,
    lora_config,
    **kwargs,
):
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
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
    )

    model, processor = FastVisionModel.from_pretrained(
        model_config.model_name_or_path,
        use_gradient_checkpointing="unsloth",
        load_in_4bit=False,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config.finetune_vision_layers,
        finetune_language_layers=lora_config.finetune_language_layers,
        finetune_attention_modules=lora_config.finetune_attention_modules,
        finetune_mlp_modules=lora_config.finetune_mlp_modules,
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        random_state=lora_config.random_state,
    )

    ################
    # Dataset
    ################
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(script_args.dataset_name) / "train.jsonl"),
            "validation": str(Path(script_args.dataset_name) / "validation.jsonl"),
        },
    )

    if data_config.agent_type == "speaker":
        agent = GenerateSpeaker(
            None,
            processor,
            image_base_path=data_config.images_base_path,
            context_presentation=data_config.context_presentation,
            use_length_token=data_config.use_length_token,
        )
    elif data_config.agent_type == "listener":
        agent = GenerateListener(
            None,
            processor,
            image_base_path=data_config.images_base_path,
            context_presentation=data_config.context_presentation,
        )

    ################
    # Training
    ################

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=RepeatedReferenceGameCollator(agent),
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processor.tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
