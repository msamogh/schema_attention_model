import argparse
import json
import numpy as np
import os
import random
import torch

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

def chat():
    # # Load dialogue stuff.
    tokenizer = get_tokenizer()
    dataset = get_dataset(tokenizer)
    # dataloader = get_dataloader(dataset)

    # Load schema stuff.
    schema_tokenizer = get_schema_tokenizer()
    schema_dataset = get_schema_dataset(
        schema_tokenizer,
        dataset.action_label_to_id
    )
    schema_dataloader = get_schema_dataloader(schema_dataset)

    # Load model.
    model = SchemaActionBertModel(
        CHAT_ARGS["model_path"],
        dropout=CHAT_ARGS["dropout"],
        num_action_labels=len(dataset.action_label_to_id)
    )

###################################################

model = SchemaActionBertModel("bert-base-uncased", 0.5, 166).cuda()
ckpt = torch.load("/blue/boyer/amogh.mannekote/sds-project/final_standard_sam/model.pt")
model.load_state_dict(ckpt)

orig_tokenizer = get_tokenizer()
orig_dataset = get_dataset(orig_tokenizer)

schema_tokenizer = get_schema_tokenizer()
schema_dataset = get_schema_dataset(
    schema_tokenizer,
    orig_dataset.action_label_to_id
)
schema_dataloader = get_schema_dataloader(schema_dataset)

def chat(user_utterance, device=0):
    dataset = utterance_to_dataset(user_utterance)
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

        sc_all_output, sc_pooled_output = model.bert_model(input_ids=sc_batch["input_ids"],
                                            attention_mask=sc_batch["attention_mask"],
                                            token_type_ids=sc_batch["token_type_ids"],
                                            return_dict=False)
        sc_action_label = sc_batch["action"]
        sc_tasks = sc_batch["task"]

        # Move to GPU
        if torch.cuda.is_available():
            for key, val in batch.items():
                if type(batch[key]) is list:
                    continue

                batch[key] = batch[key].to(device)

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

def utterance_to_dataset(utt_text):
    max_seq_length = 100

    turn_idx = 0
    history = ""
    events = []

    examples = []

    history += "[{}] {} [SEP] ".format("User", utt_text.strip())
    processed_history = ' '.join(history.strip().split()[:-1])
    encoded_history  = orig_tokenizer.encode(processed_history)

    examples.append({
        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
        "dialog_id": 1, # keep it constant
        "domains": ["bank"],
        "tasks": ["bank_fraud_report"],
        "happy": True, # shouldn't matter
        "multitask": False,
        "orig_history": processed_history,
    })
    
    return SingleUtteranceDataset(examples)

class SingleUtteranceDataset(torch.utils.data.Dataset):
    
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)


if __name__ == "__main__":
    chat("hi someone is transferring money on my account of over 10 dollars over the past week. my name is katarina miller")
