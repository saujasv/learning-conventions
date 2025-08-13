import torch
import json
from trl import (
    SFTTrainer,
    get_peft_config,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
)
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Idefics3Processor,
    PixtralProcessor,
)
from datasets import load_dataset
from agents.game import RepeatedReferenceGame
from training.collator import RepeatedReferenceGameCollator
from training.model_constants import get_lora_target_modules
from agents import GenerateSpeaker, ScoringListener
from agents.args import AgentArguments
from pathlib import Path


def train():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, AgentArguments))
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

    if not model_config.use_peft and model_config.lora_target_modules == "vision_tower":
        print("Freezing all parameters except vision tower")
        for name, param in model.named_parameters():
            if "vision_tower" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model_config.lora_target_modules = get_lora_target_modules(
        model.config, model_config.lora_target_modules
    )
    print(model_config.lora_target_modules)

    ################
    # Dataset
    ################

    if agent_args.agent_type == "speaker":
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
    elif agent_args.agent_type == "listener":
        agent = ScoringListener(
            model_type=agent_args.model_type,
            model=None,
            processor=processor,
            context_presentation=agent_args.context_presentation,
            feedback_label=agent_args.feedback_label,
            chat_template_file=agent_args.chat_template_file,
            demonstration_game=agent_args.demonstration_game,
            max_image_size=agent_args.max_image_size,
        )

    collator = RepeatedReferenceGameCollator(
        agent,
        max_image_size=agent_args.max_image_size,
        mask_only_last=True,
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

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
