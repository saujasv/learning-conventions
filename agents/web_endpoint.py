import json
from pathlib import Path
import modal
import torch
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForImageTextToText
from agents.hf_speakers import GenerateSpeaker
from agents.game import RepeatedReferenceGame

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]",
        "torch==2.6.0",
        "transformers==4.51.3",
        "accelerate==1.4.0",
        "vllm==0.8.4",
        "packaging",
        "setuptools",
        "wheel",
        "ninja",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    .apt_install("wget", "unzip", "tree")
    .add_local_file(
        "./square-black-imgs-jpg.zip",
        remote_path="/square-black-imgs-jpg.zip",
        copy=True,
    )
    .run_commands(
        "wget -q http://images.cocodataset.org/zips/val2014.zip && unzip -q val2014.zip && mkdir /train2014",
        # load only demo images from train set
        *[
            f"wget http://images.cocodataset.org/train2014/{x} && mv {x} /train2014/"
            for x in [
                "COCO_train2014_000000386973.jpg",
                "COCO_train2014_000000032997.jpg",
                "COCO_train2014_000000276694.jpg",
                "COCO_train2014_000000480495.jpg",
            ]
        ],
        "unzip -q /square-black-imgs-jpg.zip -d / && mv /square-black-imgs/*.jpg /",
    )
    .env(
        {
            "IMAGE_BASE_PATH": "/",
        }
    )
)
app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

MODEL_CONFIG_MAP = {
    "gemma_instruct": {
        "model": "google/gemma-3-12b-it",
        "processor": "google/gemma-3-12b-it",
        "model_type": "chat",
        "generation_config": {"do_sample": False, "max_new_tokens": 64},
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-chat-template.jinja",
        "demonstration_game": None,
    },
    "gemma_hard_correctness_or_cost": {
        "model": "saujasv/gemma-hard-correctness-or-cost-ipo-random",
        "processor": "google/gemma-3-12b-pt",
        "model_type": "base",
        "generation_config": {"do_sample": False, "max_new_tokens": 64},
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-pt-chat-template.jinja",
        "demonstration_game": "agents/coco-demo.json",
    },
    "gemma_hard_correctness": {
        "model": "saujasv/gemma-hard-correctness-ipo-random",
        "processor": "google/gemma-3-12b-pt",
        "model_type": "base",
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 64,
        },
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-pt-chat-template.jinja",
        "demonstration_game": "agents/coco-demo.json",
    },
    "gemma_kilogram_hard_correctness_or_cost": {
        "model": "saujasv/gemma-kilogram-hard-correctness-or-cost-ipo-random-ensemble-1",
        "processor": "google/gemma-3-12b-pt",
        "model_type": "base",
        "generation_config": {"do_sample": True, "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.95},
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-pt-chat-template.jinja",
        "demonstration_game": "agents/kilogram-demo.json",
    },
    "gemma_kilogram_hard_correctness": {
        "model": "saujasv/gemma-kilogram-hard-correctness-ipo-random-ensemble-1",
        "processor": "google/gemma-3-12b-pt",
        "model_type": "base",
        "generation_config": {"do_sample": True, "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.95},
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-pt-chat-template.jinja",
        "demonstration_game": "agents/kilogram-demo.json",
    },
    "gemma_kilogram_base": {
        "model": "saujasv/kilogram-cogen_human_pretrain-speaker-demo-vision_tower",
        "processor": "google/gemma-3-12b-pt",
        "model_type": "base",
        "generation_config": {"do_sample": True, "max_new_tokens": 64, "temperature": 1.0, "top_p": 0.95},
        "context_presentation": "once",
        "feedback_label": True,
        "chat_template_file": "agents/gemma-3-pt-chat-template.jinja",
        "demonstration_game": "agents/kilogram-demo.json",
    },
}


@app.cls(
    gpu="L40S",
    scaledown_window=30 * 60,
    max_containers=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class WebEndpoint:
    model_name: str = modal.parameter()

    @modal.enter()
    def init(self):
        processor = AutoProcessor.from_pretrained(
            MODEL_CONFIG_MAP[self.model_name]["processor"]
        )
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_CONFIG_MAP[self.model_name]["model"],
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        self.speaker = GenerateSpeaker(
            model_type=MODEL_CONFIG_MAP[self.model_name]["model_type"],
            model=model,
            processor=processor,
            generation_config=MODEL_CONFIG_MAP[self.model_name]["generation_config"],
            context_presentation=MODEL_CONFIG_MAP[self.model_name][
                "context_presentation"
            ],
            feedback_label=MODEL_CONFIG_MAP[self.model_name]["feedback_label"],
            chat_template_file=MODEL_CONFIG_MAP[self.model_name]["chat_template_file"],
            demonstration_game=MODEL_CONFIG_MAP[self.model_name]["demonstration_game"],
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, game: RepeatedReferenceGame):
        print(game)
        message = self.speaker.generate([game])[0][0]
        return JSONResponse(content={"message": message})
