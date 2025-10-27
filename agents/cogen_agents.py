from typing import Optional, List
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from pathlib import Path
from accelerate import find_executable_batch_size
import itertools

import os
import sys

# Add cogen directories to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
cogen_dir = os.path.join(script_dir, "cogen")

sys.path.insert(0, os.path.abspath(cogen_dir))
sys.path.insert(0, os.path.abspath(script_dir))

from cogen.models.joint_inference import IdeficsJointInferenceModel
from cogen.continual_learning.train_utils import filter_targets
from transformers import Idefics2ForConditionalGeneration, AutoProcessor
from .game import RepeatedReferenceGame, Trial


class CoGenAgent:
    def __init__(
        self,
        checkpoint: str = None,
        mode: str = None,
        anno_len_threshold: int = 40,
        comprehension_prompt: str = "verbose_instruction",
        context_size: int = 10,
        evaluation_type: str = "joint",
        from_scratch: bool = True,
        generation_prompt: str = "information_after",
        listener_filter: str = "no_neg_gen",
        listener_lambda: float = 0.5,
        load_from_checkpoint: bool = False,
        lora_dropout: float = 0.05,
        lora_r: int = 16,
        lora_subset: str = "vision_resampler",
        model_family_name: str = "full",
        no_lora: bool = False,
        no_shuffling: bool = False,
        noise_filter: str = "",
        num_samples: int = 10,
        only_seed: bool = False,
        sampling_type: str = "nucleus",
        seed: int = 133395,
        shared_parameters: bool = True,
        speaker_filter: str = "no_neg_comp",
        speaker_lambda: float = 0.5,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 1.0,
        training_type: str = "multitask",
        weight_decay: float = 0.1,
        index_to_token_path: str = None,
        padding_images: List[str] = None,
    ):
        self.config = {
            "anno_len_threshold": anno_len_threshold,
            "comprehension_prompt_type": comprehension_prompt,
            "context_size": context_size,
            "evaluation_type": evaluation_type,
            "from_scratch": from_scratch,
            "generation_prompt_type": generation_prompt,
            "listener_filter": listener_filter,
            "listener_lambda": listener_lambda,
            "load_from_checkpoint": load_from_checkpoint,
            "lora_dropout": lora_dropout,
            "lora_r": lora_r,
            "lora_subset": lora_subset,
            "model_family_name": model_family_name,
            "no_lora": no_lora,
            "no_shuffling": no_shuffling,
            "noise_filter": noise_filter,
            "num_samples": num_samples,
            "only_seed": only_seed,
            "sampling_type": sampling_type,
            "seed": seed,
            "shared_parameters": shared_parameters,
            "speaker_filter": speaker_filter,
            "speaker_lambda": speaker_lambda,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "training_type": training_type,
            "weight_decay": weight_decay,
            "index_to_token_path": index_to_token_path,
        }
        self.image_base_path = Path(os.environ["IMAGE_BASE_PATH"])
        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False,
            size={"longest_edge": 448, "shortest_edge": 224},
        )
        model = self.initialize_idefics()
        model.load_adapter(checkpoint, model.active_adapter)
        self.model = IdeficsJointInferenceModel(
            listener_lambda, speaker_lambda, model=model
        )

        # hardcoded values for Idefics2
        self.index_to_token = [
            28734,
            28740,
            28750,
            28770,
            28781,
            28782,
            28784,
            28787,
            28783,
            28774,
        ]
        self.padding_images = padding_images
        self.mode = mode

    def initialize_idefics(self):
        # Initialize the model
        checkpoint = "HuggingFaceM4/idefics2-8b"
        model = Idefics2ForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ).cuda()
        if self.config["no_lora"]:
            return model

        # Add LoRA adapters
        if self.config["lora_subset"] == "all":
            target_modules = r".*(text_model|vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        elif self.config["lora_subset"] == "theirs":  # from IDEFICS-2 tutorial notebook
            target_modules = r".*(text_model|vision_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        elif self.config["lora_subset"] == "vision_resampler":
            target_modules = r"(.*(vision_model|modality_projection|perceiver_resampler).*(out_proj|fc1|fc2|down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)|(.*(k_proj|q_proj|v_proj).*$)"
        elif self.config["lora_subset"] == "standard":
            target_modules = r".*(k_proj|q_proj|v_proj).*$"
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=8,
            lora_dropout=self.config["lora_dropout"],
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )

        return get_peft_model(model, lora_config)

    def construct_speaker_full_prompt(self, target_anno, target_idx):
        messages = self.construct_speaker_base_prompt(target_idx)

        # Assistant response
        target_anno = target_anno.lower().strip()
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_anno}]}
        )

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        ).strip()

    def construct_speaker_base_prompt(self, target_idx, process=False):
        messages = []

        if self.config["generation_prompt_type"] == "information_after":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f". Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "information_after_strict":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your must produce a caption that describes your target image and only your target image. ",
                        },
                        {
                            "type": "text",
                            "text": "When someone reads your description, they should be able to pick your target image from the set of 10 images. ",
                        },
                        {
                            "type": "text",
                            "text": f"Your target for this turn will be Image {target_idx}.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f". Your target image is Image {target_idx}. Produce your distinctive caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "reversed_image_description":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n"}
                    )

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif (
            self.config["generation_prompt_type"]
            == "reversed_image_description_letters"
        ):
            image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == target_idx:
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Target image start: "}
                    )

                messages[0]["content"].append({"type": "image"})
                messages[0]["content"].append(
                    {"type": "text", "text": f"Image {image_labels[i]}. "}
                )

                if i == target_idx:
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Target image end\n"}
                    )
                else:
                    messages[0]["content"].append({"type": "text", "text": f"\n"})

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {image_labels[target_idx]}. Produce your caption now.",
                }
            )
        elif (
            self.config["generation_prompt_type"]
            == "reversed_image_description_repetition"
        ):
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description.\n",
                        },
                        {
                            "type": "text",
                            "text": f"Your target will be Image {target_idx}.\n",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n"}
                    )

            # User side: Target assignment
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Your target image is Image {target_idx}. Produce your caption now.",
                }
            )
        elif self.config["generation_prompt_type"] == "information_before":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and be assigned a target image. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to produce a caption for your target image such that anyone could guess the image from your description. ",
                        },
                        {
                            "type": "text",
                            "text": f"Your target image is Image {target_idx}. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Target assignment
            messages[0]["content"].append(
                {"type": "text", "text": f". Produce your caption now."}
            )
        else:
            print("Invalid prompt type")
            assert "False"

        if process:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            ).strip()
            return prompt
        else:
            return messages

    def construct_listener_full_prompt(
        self,
        target_anno,
        target_idx=0,
    ):
        target_anno = target_anno.lower().strip()
        messages = []

        if self.config["comprehension_prompt_type"] == "information_after":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f". Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {"type": "text", "text": f" Which image does this caption describe?"}
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "information_before":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                        {"type": "text", "text": f"Your caption is: {target_anno}"},
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption

            messages[0]["content"].append(
                {"type": "text", "text": f" Which image does this caption describe?"}
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "verbose_instruction":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be presented with a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                if i == 0:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}: "}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f", Image {i}: "}
                    )
                messages[0]["content"].append({"type": "image"})

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f". Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "reversed_image_description":
            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will given a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n "}
                    )

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f"Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image 0, 1, 2, 3, 4, 5, 6, 7, 8 or 9?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        elif (
            self.config["comprehension_prompt_type"]
            == "reversed_image_description_letters"
        ):
            image_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

            # User side: Intro
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will given a sequence of 10 images and a caption describing exactly one of them. ",
                        },
                        {
                            "type": "text",
                            "text": "Your task is to guess which image the caption describes. ",
                        },
                    ],
                }
            )

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                messages[0]["content"].append(
                    {"type": "text", "text": f" Image {image_labels[i]}.\n"}
                )

            # User side: Caption
            messages[0]["content"].append(
                {"type": "text", "text": f"Caption: {target_anno}"}
            )
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f" Does this caption describe Image A, B, C, D, E, F, G, H, I or J?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {image_labels[target_idx]}",
                        }
                    ],
                }
            )
        elif self.config["comprehension_prompt_type"] == "no_instruction":
            # User side: Intro
            messages.append({"role": "user", "content": []})

            # User side: Images
            for i in range(10):
                messages[0]["content"].append({"type": "image"})
                if i == 9:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i}.\n"}
                    )
                else:
                    messages[0]["content"].append(
                        {"type": "text", "text": f" Image {i},\n "}
                    )

            # User side: Caption
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": f"Which image does the caption '{target_anno}' describe?",
                }
            )

            # Model side: Guess
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The caption describes Image {target_idx}",
                        }
                    ],
                }
            )
        else:
            print("Invalid prompt type")
            assert "False"

        return self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        ).strip()

    def prepare_inputs_listener(self, contexts, messages):
        # Generate the raw image sequence and sim idx
        raw_images = [
            [Image.open(self.image_base_path / img) for img in context]
            for context in contexts
        ]
        # if not self.config["no_shuffling"]:
        #     random.shuffle(raw_images)

        # Create the prompt and inputs for the listener
        prompts = [self.construct_listener_full_prompt(message) for message in messages]
        outputs = self.processor(
            text=prompts,
            images=raw_images,
            padding=True,
            return_tensors="pt",
        )
        l_input_tokens = outputs["input_ids"][:, :-2]
        l_attn_mask = outputs["attention_mask"][:, :-2]
        l_attn_mask[(l_input_tokens == 0).bool()] = 0
        images = outputs["pixel_values"]
        l_image_attn_mask = outputs["pixel_attention_mask"]

        # base_prompt_outputs = self.processor(
        #     text=[
        #         self.construct_speaker_base_prompt(i, process=True) for i in range(10)
        #     ],
        #     images=[raw_images] * 10,
        #     padding=False,
        # )
        # prompt_outputs = self.processor(
        #     text=[self.construct_speaker_full_prompt(message, i) for i in range(10)],
        #     images=[raw_images] * 10,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # padding_tokens = (prompt_outputs.input_ids == 0).sum(dim=1).tolist()
        # n_prefix_tokens = [
        #     len(base_tokens) + n_padding_tokens
        #     for base_tokens, n_padding_tokens in zip(
        #         base_prompt_outputs.input_ids, padding_tokens
        #     )
        # ]

        # s_input_tokens = prompt_outputs["input_ids"][:, :-1]
        # s_attn_mask = prompt_outputs["attention_mask"][:, :-1]
        # s_attn_mask[(s_input_tokens == 0).bool()] = 0
        # s_image_attn_mask = prompt_outputs["pixel_attention_mask"]
        # s_target_tokens = prompt_outputs["input_ids"][:, 1:]
        # s_target_mask = torch.ones_like(s_attn_mask)
        # for i, n_prefix in enumerate(n_prefix_tokens):
        #     s_target_mask[i, :n_prefix] = 0
        return (
            images.to(self.model.model.device),
            l_input_tokens.to(self.model.model.device),
            l_attn_mask.to(self.model.model.device),
            l_image_attn_mask.to(self.model.model.device),
            # s_input_tokens.unsqueeze(0).to("cuda"),
            # s_attn_mask.unsqueeze(0).to("cuda"),
            # s_image_attn_mask.unsqueeze(0).to("cuda"),
            # s_target_mask.unsqueeze(0).to("cuda"),
            # s_target_tokens.unsqueeze(0).to("cuda"),
        )

    def score(
        self,
        repeated_reference_games: List[RepeatedReferenceGame],
        return_logits: bool = False,
        seed: Optional[int] = None,
    ):
        (
            images,
            l_input_tokens,
            l_attn_mask,
            l_image_attn_mask,
            # s_input_tokens,
            # s_attn_mask,
            # s_image_attn_mask,
            # s_target_mask,
            # s_target_tokens,
        ) = self.prepare_inputs_listener(
            [
                [*game.context, *self.padding_images]
                for game in repeated_reference_games
            ],
            [game.trials[-1].message for game in repeated_reference_games],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.no_grad():
                if self.mode == "split_listener":
                    logits = self.model.forward(
                        "comprehension",
                        [
                            l_input_tokens,
                            l_attn_mask,
                            images,
                            l_image_attn_mask,
                        ],
                    )
                    target_logits = filter_targets(
                        logits[:, -1],
                        self.index_to_token[: len(repeated_reference_games[0].context)],
                    )
                    listener_log_probs = F.log_softmax(target_logits, dim=1)
                elif self.mode == "joint_listener":
                    raise NotImplementedError("Joint listener mode not tested")

        interpretations = list()
        for game, log_probs in zip(repeated_reference_games, listener_log_probs):
            interpretations.append(
                {t: l.item() for t, l in zip(game.context, log_probs)}
            )

        return interpretations

    @find_executable_batch_size(starting_batch_size=64)
    def batch_score(batch_size, self, repeated_reference_games, return_logits=False):
        return list(
            itertools.chain.from_iterable(
                [
                    self.score(batch, return_logits=return_logits)
                    for batch in itertools.batched(repeated_reference_games, batch_size)
                ]
            )
        )

    def prepare_inputs_speaker(self, context, target):
        raw_images = [Image.open(self.image_base_path / img) for img in context]
        target_idx = context.index(target)
        base_prompt = self.construct_speaker_base_prompt(target_idx, process=True)

        # Process the speaker inputs
        outputs = self.processor(
            text=[base_prompt],
            images=[raw_images],
            padding=True,
            return_tensors="pt",
        ).to(self.model.model.device, self.model.model.dtype)
        input_tokens = outputs["input_ids"]  # T
        attn_mask = outputs["attention_mask"]  # T
        attn_mask[(input_tokens == 0).bool()] = 0
        images = outputs["pixel_values"]  # 10x3x224x224
        image_attn_mask = outputs["pixel_attention_mask"]  # Tx10

        return (
            images,
            input_tokens,
            attn_mask,
            image_attn_mask,
            torch.tensor([target_idx], dtype=torch.long),
        )

    def generate(
        self,
        repeated_reference_games: List[RepeatedReferenceGame],
        num_return_sequences: int = 1,
        target_lengths: Optional[List[int]] = None,
    ):
        inputs = [
            self.prepare_inputs_speaker(
                [*game.context, *self.padding_images], game.trials[-1].target
            )
            for game in repeated_reference_games
        ]

        for images, input_tokens, attn_mask, image_attn_mask, target_idx in inputs:
            if self.mode == "split_speaker":
                utterances = self.model.split_generate(
                    input_tokens.expand(num_return_sequences, -1).contiguous(),
                    attn_mask.expand(num_return_sequences, -1).contiguous(),
                    images.expand(num_return_sequences, -1, -1, -1, -1).contiguous(),
                    image_attn_mask.expand(
                        num_return_sequences, -1, -1, -1
                    ).contiguous(),
                    self.processor,
                    32,
                    "nucleus",
                    self.config["temperature"],
                    self.config["top_k"],
                    self.config["top_p"],
                    num_samples=1,
                )
            elif self.mode == "joint_speaker":
                raise NotImplementedError("Joint speaker mode not tested")
                # outputs = self.model.split_generate(
                #     images,
                #     input_tokens,
                #     attn_mask,
                #     image_attn_mask,
                #     target_idx,
                #     [context],
                #     self.processor,
                #     self.image_base_path,
                #     self.index_to_token,
                #     sampling_type="nucleus",
                #     temperature=self.config["temperature"],
                #     top_k=self.config["top_k"],
                #     top_p=self.config["top_p"],
                #     num_samples=10,
                # )
            return utterances

    @find_executable_batch_size(starting_batch_size=64)
    def batch_generate(
        batch_size: int,
        self,
        repeated_reference_games: list[RepeatedReferenceGame],
        num_return_sequences: Optional[int] = None,
        target_lengths=None,
    ):
        if not target_lengths is None:
            raise NotImplementedError("target_lengths is not supported for CoGenAgent")

        if len(repeated_reference_games) != 1:
            raise ValueError(
                "batch_generate only supports exactly one game for CoGenAgent"
            )

        game = repeated_reference_games[0]
        total_sequences = num_return_sequences or 1

        # Split num_return_sequences into batches of batch_size
        all_outputs = []
        remaining_sequences = total_sequences

        while remaining_sequences > 0:
            current_batch_size = min(batch_size, remaining_sequences)

            # Generate current_batch_size sequences for the game
            batch_outputs = self.generate(
                [game], num_return_sequences=current_batch_size
            )
            all_outputs.extend(batch_outputs)

            remaining_sequences -= current_batch_size

        return [all_outputs]  # Return as list of lists to match expected format
