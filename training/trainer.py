import torch
import json
from trl import SFTTrainer, get_peft_config
from accelerate import Accelerator
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Idefics3Processor,
    PixtralProcessor,
    Gemma3Processor,
    Gemma3ForConditionalGeneration,
)
from datasets import load_dataset
from game import RepeatedReferenceGame, Trial
import itertools
from PIL import Image
from training.collator import RepeatedReferenceGameCollator
from training.model_constants import get_lora_target_modules
from agents import GenerateSpeaker, ScoringListener
from agents.hf_speakers import BaseVLMGenerateSpeaker
from agents.hf_listeners import BaseVLMScoringListener
from pathlib import Path


def train(
    script_args,
    training_args,
    model_config,
    **kwargs,
):
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

    print(model.config)

    model_config.lora_target_modules = get_lora_target_modules(
        model.config, model_config.lora_target_modules
    )
    print(model_config.lora_target_modules)

    ################
    # Dataset
    ################

    model_type = kwargs.get("--model_type", "base")
    agent_type = kwargs.get("--agent_type", "listener")
    if model_type == "base":
        if agent_type == "speaker":
            agent = BaseVLMGenerateSpeaker(
                None,
                processor,
                context_presentation=kwargs.get("--context_presentation", "once"),
                feedback_label=bool(kwargs.get("--feedback_label", "true")),
                chat_template_file=kwargs.get("--chat_template_file", None),
                demonstration_game=kwargs.get("--demonstration_game", None),
            )
        elif agent_type == "listener":
            agent = BaseVLMScoringListener(
                None,
                processor,
                context_presentation=kwargs.get(
                    "--context_presentation", "last_shuffle"
                ),
                feedback_label=bool(kwargs.get("--feedback_label", "true")),
                chat_template_file=kwargs.get("--chat_template_file", None),
                demonstration_game=kwargs.get("--demonstration_game", None),
            )
    else:
        if agent_type == "speaker":
            agent = GenerateSpeaker(
                None,
                processor,
                context_presentation=kwargs.get(
                    "--context_presentation", "last_shuffle"
                ),
                use_length_token=bool(kwargs.get("--use_length_token", "false")),
                feedback_label=bool(kwargs.get("--feedback_label", "true")),
                chat_template_file=kwargs.get("--chat_template_file", None),
            )
        elif agent_type == "listener":
            agent = ScoringListener(
                None,
                processor,
                context_presentation=kwargs.get(
                    "--context_presentation", "last_shuffle"
                ),
                feedback_label=bool(kwargs.get("--feedback_label", "true")),
                chat_template_file=kwargs.get("--chat_template_file", None),
            )

    max_image_size = kwargs.get("--max_image_size", None)
    collator = RepeatedReferenceGameCollator(
        agent,
        max_image_size=int(max_image_size) if max_image_size else None,
        mask_only_last=(agent.context_presentation != "once"),
    )

    dataset = load_dataset(
        "text",
        data_files={
            "train": str(Path(script_args.dataset_name) / "train.jsonl"),
            "validation": str(Path(script_args.dataset_name) / "validation.jsonl"),
        },
    )

    # Process dataset in a way that avoids schema inference issues
    def process_example(x):
        game = RepeatedReferenceGame.model_validate(json.loads(x["text"]))
        messages, image_paths = agent.construct_prompt_messages(game)
        return {"messages": messages, "image_paths": image_paths}

    dataset = dataset.map(process_example, remove_columns=["text"])

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
