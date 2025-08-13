from typing import Tuple, List
from io import BytesIO
import base64
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import apply_hf_chat_template, parse_chat_messages
from vllm.inputs import TokensPrompt
from .base_agent import BaseListener, BaseSpeaker
from .utils import CHAT_TEMPLATE, QWEN_CHAT_TEMPLATE
from pathlib import Path
from .game import RepeatedReferenceGame, Trial


class vLLMAgent:
    def __init__(self, model, tensor_parallel_size=1, image_base_path: str = ""):
        if isinstance(model, str):
            self.llm = LLM(
                model=model,
                limit_mm_per_prompt={"image": 256},
                max_model_len=16384,
                tensor_parallel_size=tensor_parallel_size,
            )
        elif isinstance(model, LLM):
            self.llm = model

        self.text_only_assistant = False
        self.image_base_path = image_base_path

    def api_call(self, messages):
        outputs = self.llm.chat(
            messages=messages, sampling_params=self.generation_config
        )
        return outputs[0].outputs[0].text.strip("\"'")

    def encode_image(self, image_path):
        if self.tangrams:
            image_format = "png"
        else:
            image_format = "jpg"
        with open(Path(self.image_base_path) / image_path, "rb") as image_file:
            return f"data:image/{image_format};base64,{base64.b64encode(image_file.read()).decode("utf-8")}"


class vLLMSpeaker(vLLMAgent, BaseSpeaker):
    def __init__(
        self,
        model,
        image_base_path,
        context_presentation="last_no_shuffle",
        feedback_label=False,
        tensor_parallel_size=1,
        generation_config=None,
        prompt_type="standard",
        tangrams=True,
    ):
        vLLMAgent.__init__(self, model, tensor_parallel_size, image_base_path)

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        if generation_config is None:
            self.generation_config = {
                "max_tokens": 64,
                "temperature": 0.3,
            }
        else:
            self.generation_config = generation_config

        self.prompt_type = prompt_type
        self.tangrams = tangrams

    def generate(self, repeated_reference_game, num_return_sequences=1):
        messages, _ = self.construct_prompt_messages(repeated_reference_game)
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=SamplingParams(
                **{**self.generation_config, "n": num_return_sequences}
            ),
            chat_template=(
                QWEN_CHAT_TEMPLATE
                if "qwen" in self.llm.llm_engine.get_model_config().model.lower()
                else CHAT_TEMPLATE
            ),
            chat_template_content_format="string",
            use_tqdm=False,
        )
        return [x.text.strip().strip('"') for x in outputs[0].outputs]


class vLLMListener(vLLMAgent, BaseListener):
    def __init__(
        self,
        model,
        image_base_path: str = "",
        context_presentation="last_shuffle",
        feedback_label=True,
        generation_config=None,
        tensor_parallel_size=1,
        tangrams=True,
    ):
        vLLMAgent.__init__(self, model, tensor_parallel_size, image_base_path)

        if generation_config is None:
            generation_config = {
                "max_tokens": 8,
                "temperature": 0,
            }

        self.context_presentation = context_presentation
        self.feedback_label = feedback_label

        self.generation_config = SamplingParams(**generation_config)
        self.tangrams = tangrams

    def get_prompt_prefix(self, repeated_reference_game):
        counterfactual_games = [
            RepeatedReferenceGame(
                context=repeated_reference_game.context,
                trials=[
                    *repeated_reference_game.trials[:-1],
                    Trial(
                        target=repeated_reference_game.trials[-1].target,
                        message=repeated_reference_game.trials[-1].message,
                        selection=c,
                        correct=True,
                    ),
                ],
            )
            for c in repeated_reference_game.context
        ]

        counterfactual_prompt_messages, counterfactual_prompt_contexts = zip(
            *[
                self.construct_prompt_messages(
                    cg, exclude_feedback_on_last=True, random_seed=412
                )
                for cg in counterfactual_games
            ]
        )

        assert all(
            [
                counterfactual_prompt_contexts[0] == ctx
                for ctx in counterfactual_prompt_contexts[1:]
            ]
        ), "Contexts for counterfactual games should be the same."

        counterfactual_prompt_tokens = list()
        for cfpm in counterfactual_prompt_messages:
            conversation, mm_data = parse_chat_messages(
                cfpm,
                self.llm.llm_engine.get_model_config(),
                self.llm.get_tokenizer(),
                "string",
            )

            prompt_data = apply_hf_chat_template(
                self.llm.get_tokenizer(),
                trust_remote_code=self.llm.llm_engine.get_model_config().trust_remote_code,
                conversation=conversation,
                chat_template=(
                    QWEN_CHAT_TEMPLATE
                    if "qwen" in self.llm.llm_engine.get_model_config().model.lower()
                    else CHAT_TEMPLATE
                ),
                add_generation_prompt=True,
                continue_final_message=False,
            )

            tokens = self.llm.get_tokenizer().encode(prompt_data)
            counterfactual_prompt_tokens.append(tokens)

        idx = (
            min(len(x) for x in counterfactual_prompt_tokens) - 2
        )  # -2 to exclude the last token which is the eos token

        while not all(
            [
                x[idx] == counterfactual_prompt_tokens[0][idx]
                for x in counterfactual_prompt_tokens
            ]
        ):
            idx -= 1

        assert all(
            [
                x[: idx + 1] == counterfactual_prompt_tokens[0][: idx + 1]
                for x in counterfactual_prompt_tokens
            ]
        ), "Prefixes for counterfactual games should be the same."

        prompt = TokensPrompt(
            prompt_token_ids=counterfactual_prompt_tokens[0][: idx + 1]
        )
        prompt["multi_modal_data"] = mm_data

        return prompt, dict(
            [
                (counterfactual_prompt_tokens[i][idx + 1], c)
                for i, c in enumerate(repeated_reference_game.context)
            ]
        )

    def score(self, repeated_reference_games):
        if isinstance(repeated_reference_games, RepeatedReferenceGame):
            repeated_reference_games = [repeated_reference_games]

        prompts, context_maps = zip(
            *[self.get_prompt_prefix(game) for game in repeated_reference_games]
        )

        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                n=1,
                logprobs=20,
                max_tokens=1,
            ),
            use_tqdm=False,
        )

        logprobs = [dict() for _ in range(len(repeated_reference_games))]
        for i, context_map in enumerate(context_maps):
            for r in context_map:
                if r in outputs[i].outputs[0].logprobs[0]:
                    logprobs[i][context_map[r]] = (
                        outputs[i].outputs[0].logprobs[0][r].logprob
                    )
                else:
                    logprobs[i][context_map[r]] = -float("inf")

        return logprobs if len(logprobs) > 1 else logprobs[0]
