from transformers import (
    PixtralProcessor,
    Idefics3Processor,
    Qwen2_5_VLProcessor,
    Gemma3Processor,
)
from agents.hf_speakers import GenerateSpeaker
from agents.hf_listeners import ScoringListener


def get_lora_target_modules(model_config, lora_targets):
    if lora_targets == "all-linear":
        return "all-linear"
    elif lora_targets == "vision":
        if (
            model_config.model_type == "llava"
            and model_config.vision_config.model_type == "pixtral"
        ):
            return r"(vision_tower|multi_modal_projector).*(linear_1|linear_2|down_proj|gate_proj|up_proj|q_proj|k_proj|v_proj|o_proj).*$"
        else:
            raise ValueError("Vision LoRA not supported for this model")
    elif lora_targets == "text":
        if (
            model_config.model_type == "llava"
            and model_config.vision_config.model_type == "pixtral"
        ):
            return r"(language_model|multi_modal_projector).*(linear_1|linear_2|down_proj|gate_proj|up_proj|q_proj|k_proj|v_proj|o_proj).*$"
        else:
            raise ValueError("Text LoRA not supported for this model")
    else:
        return lora_targets


def get_chat_template_features(agent):
    processor = agent.processor

    if isinstance(processor, PixtralProcessor):
        if agent.model_type == "chat":
            return "[/INST]", "[INST]"
        else:
            raise ValueError(f"Unsupported agent type for Pixtral: {agent.model_type}")
    elif isinstance(processor, Gemma3Processor):
        if agent.model_type == "chat":
            return "<start_of_turn>model\n", "<start_of_turn>user\n"
        elif agent.model_type == "base" and isinstance(agent, GenerateSpeaker):
            return " description:\n", "\nFeedback"
        elif agent.model_type == "base" and isinstance(agent, ScoringListener):
            return "Image:\n", "<eos>"
        else:
            raise ValueError(f"Unsupported agent type for Gemma 3: {agent.model_type}")
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")


def get_image_sizes_field(processor):
    if isinstance(processor, PixtralProcessor):
        return "image_sizes"
    elif isinstance(processor, Qwen2_5_VLProcessor):
        return "image_grid_thw"
    elif isinstance(processor, Gemma3Processor):
        return None
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")
