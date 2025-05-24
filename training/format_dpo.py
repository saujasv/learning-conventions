import json
import random
import os
from PIL import Image

BASE_IMAGE_PATH = "/data/tir/projects/tir7/user_data/ambharad/icca-tangrams"

def generate_random_string(length=10):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=length)).capitalize() + '.'

def is_valid_turn(turn):
    # A valid turn should not be just an image URL or contain image paths
    if not isinstance(turn, dict):
        return False
    # If the turn is just an image URL or has non-empty image_paths, it's invalid
    if turn.get("type") == "image_url":
        return False
    if "image_paths" in turn and turn["image_paths"]:
        return False
    if "image_url" in turn and turn["image_url"]:
        return False
    return True

def clean_turn(turn):
    # Create a copy to clean the turn
    turn = dict(turn)
    return turn

def process_dialogue(dialogue_obj):
    dialogue = dialogue_obj.get("text", [])
    print(dialogue_obj,"\n\n\n\n")
    if not isinstance(dialogue, list):
        raise ValueError("Expected 'conversation' to be a list of turns.")

    # Filter out invalid turns (those with just image URL or non-empty image_paths)
    filtered = [clean_turn(turn) for turn in dialogue if is_valid_turn(turn)]

    # Choose a random assistant turn
    random_assistant_turn = random.choice([turn for turn in filtered if turn.get('role') == 'assistant'])

    # Find the index of this random assistant turn in the filtered list
    i = filtered.index(random_assistant_turn)

    # Prepare the samples for the chosen assistant turn
    prompt = filtered[:i]  # All turns before the assistant's turn
    chosen = [random_assistant_turn]  # The assistant's caption
    rejected = [{"role": "assistant", "content": generate_random_string()}]  # Random rejected string

    # Return the sample
    return [{
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "image_path": dialogue_obj['image_paths']
    }]

def convert_jsonl_for_dpo(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for lineno, line in enumerate(infile, 1):
            try:
                obj = json.loads(line)
                samples = process_dialogue(obj)
                for sample in samples:
                    json.dump(sample, outfile)
                    outfile.write('\n')
            except Exception as e:
                print(f"Error on line {lineno}: {e}")
                continue

# Example usage
convert_jsonl_for_dpo("validation_formatted.jsonl", "/data/tir/projects/tir7/user_data/ambharad/icca-tangrams/training/dpo_test/train.jsonl")