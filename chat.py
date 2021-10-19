import argparse
import json
import numpy as np
import os
import random
import torch
from pprint import pprint

from typing import Any, Dict, Tuple
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tokenizers import BertWordPieceTokenizer
from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel


CURR_DIR = os.path.abspath(os.path.dirname(__file__))
CHAT_ARGS = {
    "data_path": os.path.join(CURR_DIR, "STAR/dialogues/"),
    "schema_path": os.path.join(CURR_DIR, "STAR/tasks/"),
    "token_vocab_path": os.path.join(CURR_DIR, "bert-base-uncased-vocab.txt"),
    "output_dir": os.path.join(CURR_DIR, "sam_task_transfer/"),
    "task": "action",
    "max_seq_length": 100,
    "dropout": 0.5,
    "schema_max_seq_length": 50,
    "model_path": os.path.join(CURR_DIR, "final_standard_sam"),
    "use_schema": True
}
TOKEN_VOCAB_NAME = os.path.basename(CHAT_ARGS["token_vocab_path"]).replace(".txt", "")


def get_schema_tokenizer():
    sc_tokenizer = BertWordPieceTokenizer(
        CHAT_ARGS["token_vocab_path"],
        lowercase=True
    )
    sc_tokenizer.enable_padding(
        length=CHAT_ARGS["schema_max_seq_length"]
    )
    return sc_tokenizer

def get_schema_dataset(sc_tokenizer, action_label_to_id):
    schema = NextActionSchema(
        CHAT_ARGS["schema_path"],
        sc_tokenizer,
        CHAT_ARGS["schema_max_seq_length"],
        action_label_to_id,
        TOKEN_VOCAB_NAME
    )
    return schema

def get_schema_dataloader(schema_dataset):
    schema_dataloader = DataLoader(
        dataset=schema_dataset,
        batch_size=len(schema_dataset),
        pin_memory=True,
        shuffle=True
    )
    return schema_dataloader

def get_tokenizer():
    tokenizer = BertWordPieceTokenizer(
        CHAT_ARGS["token_vocab_path"],
        lowercase=True
    )
    tokenizer.enable_padding(
        length=CHAT_ARGS["max_seq_length"]
    )
    return tokenizer

def get_dataset(tokenizer):
    dataset = NextActionDataset(
        CHAT_ARGS["data_path"],
        tokenizer,
        CHAT_ARGS["max_seq_length"],
        TOKEN_VOCAB_NAME
    )
    return dataset

def get_dataloader(dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=CHAT_ARGS["train_batch_size"],
        pin_memory=True
    )             

DOMAIN_STR = "hotel"
TASK_STR = "hotel_book"
DOMAIN = json.load(open(os.path.join(CURR_DIR, "STAR", "tasks", TASK_STR, f"{TASK_STR}.json"), "r"))
# from pprint import pprint
# pprint(DOMAIN)

###################################################

model = SchemaActionBertModel("bert-base-uncased", 0.5, 166).cuda()
ckpt = torch.load("/blue/boyer/amogh.mannekote/sds-project/final_domaintransfer_zeroshot_sam_best_fixed/ride/model.pt")
model.load_state_dict(ckpt)

orig_tokenizer = get_tokenizer()
orig_dataset = get_dataset(orig_tokenizer)

schema_tokenizer = get_schema_tokenizer()
schema_dataset = get_schema_dataset(
    schema_tokenizer,
    orig_dataset.action_label_to_id
)
schema_dataloader = get_schema_dataloader(schema_dataset)

def chat():
    history = ""
    while True:
        # 1. get user input
        # 2. update history
        # 3. pass to model + get sys resp
        # 4. update history
        # 5. go back to 1


        user_input = input("USER: >> ")
        history += "[{}] {} [SEP] ".format("User", user_input.strip())
        system_response = get_next_utterance(user_input)
        output_reply = DOMAIN["replies"][system_response]
        print(f"SYS: >> {output_reply}")
        history+="[{}] {} [SEP] ".format("Agent", output_reply.strip())

        # TODO: handle query later
        


def get_next_utterance(history, device=0):
    dataset = history_to_dataset(history)
    eval_dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
    
    id_map = orig_dataset.action_label_to_id
    label_map = sorted(id_map, key=id_map.get)

    sentence = []
    preds = []

    model.eval()
    batch = next(iter(eval_dataloader))
    
    # Get schema pooled outputs
    with torch.no_grad():
        sc_batch = next(iter(schema_dataloader))
        if torch.cuda.is_available():
            for key, val in sc_batch.items():
                if type(sc_batch[key]) is list:
                    continue
                sc_batch[key] = sc_batch[key].to(device)

        try:
            sc_all_output, sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                                attention_mask=sc_batch["attention_mask"],
                                                token_type_ids=sc_batch["token_type_ids"],
                                                return_dict=False)
        except Exception as e:
            print(e)
        sc_action_label = sc_batch["action"]
        sc_tasks = sc_batch["task"]

        # Move to GPU
        if torch.cuda.is_available():
            for key, val in batch.items():
                if type(batch[key]) is list:
                    continue
                batch[key] = batch[key].to(device)

        print("======MAIN MODEL INPUT PARAMS======")
        action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        token_type_ids=batch["token_type_ids"],
                                        tasks=batch["tasks"],
                                        sc_all_output=sc_all_output,
                                        sc_pooled_output=sc_pooled_output,
                                        sc_tasks=sc_tasks,
                                        sc_action_label=sc_action_label)
        # Argmax to get predictions
        action_preds = torch.argmax(action_logits, dim=1).cpu().tolist()
        preds += action_preds
        sentence += [orig_tokenizer.decode(e.tolist(), skip_special_tokens=False).replace(" [PAD]", "") for e in batch["input_ids"]]

    # Perform evaluation
    return label_map[preds[0]]

def history_to_dataset(history):
    max_seq_length = 100
    
    # history += "[{}] {} [SEP] ".format("User", utt_text.strip())
    processed_history = ' '.join(history.strip().split()[:-1])
    encoded_history  = orig_tokenizer.encode(processed_history)

    examples = [{
        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
        "dialog_id": 75, # keep it constant
        "domains": DOMAIN_STR,
        "tasks": TASK_STR,
        "happy": True, # shouldn't matter
        "multitask": False,
        "orig_history": processed_history,
    }]

    return SingleUtteranceDataset(examples)

class SingleUtteranceDataset(torch.utils.data.Dataset):
    
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    pass