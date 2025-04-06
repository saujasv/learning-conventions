import torch
from trl import SFTTrainer, get_peft_config
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Idefics3Processor,
    PixtralProcessor,
)
from datasets import load_dataset
from game import RepeatedReferenceGame, Trial
import itertools
from PIL import Image
from training.collator import RepeatedReferenceGameCollator
from training.model_constants import get_lora_target_modules
from agents import GenerateSpeaker, GenerateListener
from pathlib import Path


def prepare_game(x, agent):
    game = RepeatedReferenceGame.model_validate(x)
    game_messages = agent.construct_prompt_messages(game)[0]
    message_texts = agent.processor.apply_chat_template(game_messages)
    message_images = list(
        itertools.chain.from_iterable(
            [
                [
                    Image.open(f'{chunk["image_url"]["url"]}').convert("RGB")
                    for chunk in m["content"]
                    if chunk["type"] == "image_url"
                ]
                for m in game_messages
            ]
        )
    )

    processed = agent.processor(
        text=message_texts, images=message_images, return_tensors="pt"
    )

    return processed


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

    model = AutoModelForVision2Seq.from_pretrained(
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

    agent_type = kwargs.get("--agent_type", "speaker")
    if agent_type == "speaker":
        agent = GenerateSpeaker(
            None,
            processor,
            image_base_path=kwargs.get("--images_base_path", None),
            context_presentation=kwargs.get("--context_presentation", "last_shuffle"),
            use_length_token=bool(kwargs.get("--use_length_token", "false")),
            feedback_label=bool(kwargs.get("--feedback_label", "false")),
        )
    elif agent_type == "listener":
        agent = GenerateListener(
            None,
            processor,
            image_base_path=kwargs.get("--images_base_path", None),
            context_presentation=kwargs.get("--context_presentation", "last_shuffle"),
            feedback_label=bool(kwargs.get("--feedback_label", "false")),
        )

    collator = RepeatedReferenceGameCollator(agent)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(script_args.dataset_name) / "train.jsonl"),
            "validation": str(Path(script_args.dataset_name) / "validation.jsonl"),
        },
    )

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
