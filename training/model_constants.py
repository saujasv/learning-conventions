from transformers import PixtralProcessor, Idefics3Processor, Qwen2_5_VLProcessor


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


def get_chat_template_features(processor):
    if isinstance(processor, PixtralProcessor):
        return "[/INST]", "[INST]"
    elif isinstance(processor, Idefics3Processor):
        return "Assistant:", "User:"
    elif isinstance(processor, Qwen2_5_VLProcessor):
        return "<|im_start|>assistant", "<|im_start|>user"
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")


def get_image_sizes_field(processor):
    if isinstance(processor, PixtralProcessor):
        return "image_sizes"
    elif isinstance(processor, Qwen2_5_VLProcessor):
        return "image_grid_thw"
    else:
        raise ValueError(f"Unsupported processor type: {type(processor)}")
