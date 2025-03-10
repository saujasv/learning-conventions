import itertools
from PIL import Image
from transformers import HfArgumentParser, AutoProcessor
from training.qwen_finetune.training.train import train
from training.collator import RepeatedReferenceGameCollator
from training.configs import DataConfig
from training.qwen_finetune.training.params import TrainingArguments, ModelArguments
from game import RepeatedReferenceGame
from agents import GenerateSpeaker, GenerateListener
from datasets import load_dataset


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


def main(config_file):
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataConfig))
    model_config, training_args, data_config = parser.parse_yaml_file(config_file)

    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)

    if data_config.agent_type == "speaker":
        agent = GenerateSpeaker(
            None,
            processor,
            image_base_path=data_config.images_base_path,
            context_presentation=data_config.context_presentation,
            use_length_token=data_config.use_length_token,
            feedback_label=data_config.feedback_label,
        )
    elif data_config.agent_type == "listener":
        agent = GenerateListener(
            None,
            processor,
            image_base_path=data_config.images_base_path,
            context_presentation=data_config.context_presentation,
            feedback_label=data_config.feedback_label,
        )

    collator = RepeatedReferenceGameCollator(agent)

    dataset = load_dataset(
        "json",
        data_files={
            "train": data_config.train_dataset_path,
            "validation": data_config.validation_dataset_path,
        },
    )

    dataset = dataset.map(
        lambda x: prepare_game(x, agent),
        remove_columns=["context", "trials"],
        writer_batch_size=50,
    )

    train(
        model_config, training_args, dataset["train"], dataset["validation"], collator
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
