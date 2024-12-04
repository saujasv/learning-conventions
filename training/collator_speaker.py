from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import random
import itertools
from game import RepeatedReferenceGame, Trial
import json
from agents.chat_speaker import ChatSpeaker
import torch
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, all_game_messages, processor):
        self.all_game_messages = all_game_messages
        self.processor = processor

    def __len__(self):
        return len(self.all_game_messages)

    def __getitem__(self, idx):
        game = self.all_game_messages[idx]
        fm = self.processor.apply_chat_template(game[0])
        processed = self.processor(
            text=fm,
            images=list(
                itertools.chain.from_iterable(
                    [
                        [
                            Image.open("square-black-imgs/" + chunk["image_url"]["url"]).convert("RGB")
                            for chunk in m["content"]
                            if chunk["type"] == "image_url"
                        ]
                        for m in game[0]
                    ]
                )
            ),
            return_tensors="pt",
        )
        return {'input_ids': processed.input_ids[0]} 
    

class CombinedDataCollator:
    def __init__(self, processor, hf_collator):
        self.processor = processor
        self.hf_collator = hf_collator

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        hf_batch = [{'input_ids': ids} for ids in padded_input_ids]

        hf_output = self.hf_collator(hf_batch)

        return {
            'input_ids': hf_output['input_ids'],  
            'attention_mask': hf_output['attention_mask'],
        }



with open("lexgram/exp1_data-icca-no_control.json", "r") as f:
    data = json.load(f)

    games = dict()
    for gameid, rrg_data in data.items():
        rrg = RepeatedReferenceGame.model_validate_json(json.dumps(rrg_data))
        games[gameid] = rrg


processor = AutoProcessor.from_pretrained("saujasv/pixtral-12b")

speaker = ChatSpeaker()

all_game_messages = [speaker.construct_prompt_messages(game) for gameid, game in games.items()]

instruction_template = "[INST]"
response_template = "[\INST]"
hf_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=processor.tokenizer, mlm=False)

dataset = CustomDataset(all_game_messages, processor)
collator = CombinedDataCollator(processor,hf_collator=hf_collator)

dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

for batch in dataloader:
    print(batch['input_ids'].shape)
    break
