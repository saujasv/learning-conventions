import torch
from trl import SFTTrainer, get_peft_config, DPOTrainer
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Idefics3Processor,
    PixtralProcessor,
    AutoTokenizer
)
from datasets import load_dataset
from game import RepeatedReferenceGame, Trial
import itertools
from PIL import Image
from training.collator import RepeatedReferenceGameCollator
from training.model_constants import get_lora_target_modules
from agents import GenerateSpeaker, GenerateListener
from pathlib import Path
import re
from datasets import features
import os




def load_images(example):
    print("function called!!!")
    print("example", example)
    image_dir = Path('/data/tir/projects/tir7/user_data/ambharad/icca-tangrams')  # base dir that contains the image folder
    loaded = []

    for rel_path in example.get("image_path", []):
        img_path = image_dir / rel_path
        print(f"Trying to load {img_path}")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            loaded.append(img)
        else:
            print(f"Missing: {img_path}")
            loaded.append(None)  # or raise Exception if you want strict checking

    example["images"] = loaded
    return example



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

    if isinstance(processor, PixtralProcessor):
        processor.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<s>[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}"
            " {{ message['content'] }}</s>"
            "{% endif %}"
            "{% endfor %}"
        )
    

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



    model_config.lora_target_modules = get_lora_target_modules(
        model.config, model_config.lora_target_modules
    )

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


    dataset = load_dataset(
        "json",
        data_files={
            "train": str(Path(script_args.dataset_name) / "train.jsonl"),
            "validation": str(Path(script_args.dataset_name) / "validation.jsonl"),
        },
    )


    dataset = dataset.map(load_images, batched=False)
    for split in dataset.keys():
        f = dataset[split].features
        f["images"] = features.Sequence(features.Image(decode=True))
        dataset[split] = dataset[split].cast(f)

    ################
    # Training
    ################

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processor,
        peft_config=get_peft_config(model_config),
    )
    

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)