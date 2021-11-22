import argparse
import json
import numpy as np
import os
import random
import re
import torch
from pprint import pprint

from typing import Any, Dict, Tuple
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tokenizers import BertWordPieceTokenizer
# from transformers import pipeline

from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel
from STAR.apis import api

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
    "use_schema": True
}
TOKEN_VOCAB_NAME = os.path.basename(CHAT_ARGS["token_vocab_path"]).replace(".txt", "")

# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

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

def get_entity(slot_name, sys_query, usr_response):
    from dataclasses import dataclass
    from typing import List
    import re

    @dataclass
    class PrimingExample:
      slot_name: str
      sys_query: str
      usr_response: str
      result: str = None

      def __post_init__(self):
        self.gpt_result_prompt = f"The {self.slot_name} is {self.result}."

      def get_input(self):
        if self.result:
          return f"History: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput: {self.result}"
        else:
          return f"Objective: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput:"
        # return self.sys_query + " " + self.usr_response + " " + self.gpt_result_prompt



    ex1 = PrimingExample("number of people coming to the party", "How many people are coming to the party?", "We might be like 8 people. I'm not too sure.", "8")
    ex2 = PrimingExample("destination", "Where would you like to go?", "I would like to go home, please", "home")
    ex3 = PrimingExample("destination", "Where would you like to go?", "I would like to go home, please", "home")
    # ex3 = PrimingExample("start time", "Who is the driver", "Philip", "")
    # ex4 = PrimingExample("driver", "Who is the driver", "Philip", "Philip")
    # ex3 = PrimingExample("dpcty", "Who is the driver", "Philip", "Philip" )
    ex4 = PrimingExample("name", "What is your name?", "My name is Max", "Max" )
    # ex5 = PrimingExample("name", "What is your name?", "My name is Max", "Max" )

    # ex5 = PrimingExample("want to go", "Who is the driver", "Philip", "Philip" )
    # ex6 = PrimingExample("dpcty", "Who is the driver", "Philip", "Philip" )
    prime = ex1.get_input() + "\n" + ex2.get_input() + "\n" + ex3.get_input() + "\n"+ ex4.get_input() #+ "\n"+ ex5.get_input()
    input_example = PrimingExample(slot_name, sys_query, usr_response)
    
    gpt_input = prime + "\n" + input_example.get_input()
#     print(gpt_input)
    out = generator(
        gpt_input,
        do_sample=False,
        max_length=30,
        min_length=3
    )[0]["generated_text"][len(gpt_input) + 1:]
    entity = re.sub(r'\W+', '', usr_response[usr_response.find(out):].split(" ")[0])
    return entity

UNCERTAINTY_THRESHOLD = -7
# from pprint import pprint
# pprint(DOMAIN)

###################################################

model = SchemaActionBertModel("bert-base-uncased", 0.5, 171).cuda()
# ckpt = torch.load("/blue/boyer/amogh.mannekote/sds-project/final_standard_sam/model.pt")
ckpt = torch.load("/blue/boyer/amogh.mannekote/sds-project/final_tasktransfer_zeroshot_sam_best_fixed/ride_book/model.pt")
model.load_state_dict(ckpt)

orig_tokenizer = get_tokenizer()
orig_dataset = get_dataset(orig_tokenizer)

schema_tokenizer = get_schema_tokenizer()
schema_dataset = get_schema_dataset(
    schema_tokenizer,
    orig_dataset.action_label_to_id
)
schema_dataloader = get_schema_dataloader(schema_dataset)

def print_sys_response(response):
    print(f"SYS: >> {response.replace('%', '')}")

def print_sys_clarification(prev_response):
    print(f"SYS: >> Sorry, I didn't catch that. Could you rephrase that more explicitly?\n{prev_response}")

def get_augmented_user_input(prev_sys_response):
    user_input = input("USER: >> ").strip().lower()
    if len(user_input) < 8 and ("hello" in user_input or "hi" in user_input or "hey" in user_input):
        user_input = "hello hello hello"
    user_input, entity = explicitize_user_input(user_input, prev_sys_response)
    return user_input, entity

def explicitize_user_input(user_input, prev_sys_response):
    search_result = re.search(r"%(.*)%", prev_sys_response)
    entity = None
    if search_result is not None:
        entity = search_result.group(1)
        user_input = f"The {entity} is {user_input}."
    return user_input, entity

def get_turn_str(speaker, utterance):
    return "[{}] {} [SEP] ".format(speaker, utterance.strip())

def make_api_call(slots):
    def to_title_case(key):
        return key.title().replace(" ", "").strip()
    return api.call_api(
        "bank_balance",
        constraints=[{
            to_title_case(k): v for k, v in slots.items()
        }],
    )

def chat(domain, task):
    DOMAIN = json.load(
        open(os.path.join(CURR_DIR, "STAR", "tasks", task, f"{task}.json"), "r")
    )
    history = ""
    prev_sys_response = ""
    slots = dict()
    while True:
        user_input, entity = get_augmented_user_input(prev_sys_response)
    
        user_turn_str = get_turn_str("User", user_input)
        history += user_turn_str

        system_response, is_ambiguous = get_next_utterance(user_input, domain, task)
        if is_ambiguous:
            print_sys_clarification(prev_sys_response)
            history = history[:-len(user_turn_str)]  # Undo appending of the latest user utterance.
            continue
            
        if entity is not None:
            pass
            # slots[entity] = get_entity(entity, system_response, user_input)
        
        output_reply = DOMAIN["replies"][system_response]
        if output_reply == "[QUERY]":
            print(f"You provided: {slots}")
            db_response = make_api_call(slots)
            print(db_response)
            
        prev_sys_response = output_reply
        print_sys_response(output_reply)

        sys_turn_str = get_turn_str("Agent", output_reply.strip().replace("%", ""))
        history += sys_turn_str

        
def get_next_utterance(history, domain_str, task_str, device=0):
    dataset = history_to_dataset(history, domain_str, task_str)
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

#         print("======MAIN MODEL INPUT PARAMS======")
        action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        token_type_ids=batch["token_type_ids"],
                                        tasks=batch["tasks"],
                                        sc_all_output=sc_all_output,
                                        sc_pooled_output=sc_pooled_output,
                                        sc_tasks=sc_tasks,
                                        sc_action_label=sc_action_label)
        # action_logits = torch.nn.functional.softmax(action_logits)
        # Argmax to get predictions
        prediction_scores = [(label_map[i], action_logits[0][i].item()) for i in range(action_logits.size(1))]
        prediction_scores = list(sorted(prediction_scores, key=lambda x: x[1], reverse=True))
#         pprint(list(sorted(prediction_scores, key=lambda x: x[1], reverse=True)))
        action_preds = torch.argmax(action_logits, dim=1).cpu().tolist()
        preds += action_preds
        sentence += [orig_tokenizer.decode(e.tolist(), skip_special_tokens=False).replace(" [PAD]", "") for e in batch["input_ids"]]
        if prediction_scores[0][1] < UNCERTAINTY_THRESHOLD:
            return label_map[preds[0]], True
    # Perform evaluation
    return label_map[preds[0]], False

def history_to_dataset(history, domain_str, task_str):
    max_seq_length = 100
    
    # history += "[{}] {} [SEP] ".format("User", utt_text.strip())
    processed_history = ' '.join(history.strip().split()[:-1])
    encoded_history  = orig_tokenizer.encode(processed_history)

    examples = [{
        "input_ids": np.array(encoded_history.ids)[-max_seq_length:],
        "attention_mask": np.array(encoded_history.attention_mask)[-max_seq_length:],
        "token_type_ids": np.array(encoded_history.type_ids)[-max_seq_length:],
        "dialog_id": 75, # keep it constant
        "domains": domain_str,
        "tasks": task_str,
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