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
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=False,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=412,
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

    agent_type = kwargs.get("--agent_type", "speaker")
    if agent_type == "speaker":
        agent = GenerateSpeaker(
            None,
            processor,
            image_base_path=kwargs.get("--images_base_path", None),
            context_presentation=kwargs.get("--context_presentation", "last_shuffle"),
        )
    elif agent_type == "listener":
        agent = GenerateListener(
            None,
            processor,
            image_base_path=kwargs.get("--images_base_path", None),
            context_presentation=kwargs.get("--context_presentation", "last_shuffle"),
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
