import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from tokenizers import BertWordPieceTokenizer

from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel


UNCERTAINTY_THRESHOLD = -7


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


class SingleUtteranceDataset(torch.utils.data.Dataset):
    
    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)

    
def load_saved_model(task):
    model = SchemaActionBertModel("bert-base-uncased", 0.5, 171).cuda()
    ckpt = torch.load(f"/blue/daisyw/v.pathak/zs-bot/sds-zs-bot/SAM/schema_attention_model/sam_task_transfer/{task}/model.pt")
    model.load_state_dict(ckpt)
    return model
    

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


def history_to_dataset(history, domain_str, task_str):
    max_seq_length = 100

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


def get_system_action(model, history, domain_str, task_str, device=0):
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

        action_logits, _ = model.predict(input_ids=batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        token_type_ids=batch["token_type_ids"],
                                        tasks=batch["tasks"],
                                        sc_all_output=sc_all_output,
                                        sc_pooled_output=sc_pooled_output,
                                        sc_tasks=sc_tasks,
                                        sc_action_label=sc_action_label)
        # Argmax to get predictions
        prediction_scores = [(label_map[i], action_logits[0][i].item()) for i in range(action_logits.size(1))]
        prediction_scores = list(sorted(prediction_scores, key=lambda x: x[1], reverse=True))

        action_preds = torch.argmax(action_logits, dim=1).cpu().tolist()
        preds += action_preds
        sentence += [orig_tokenizer.decode(e.tolist(), skip_special_tokens=False).replace(" [PAD]", "") for e in batch["input_ids"]]
        if prediction_scores[0][1] < UNCERTAINTY_THRESHOLD:
            return label_map[preds[0]], True
    # Perform evaluation
    return label_map[preds[0]], False


orig_tokenizer = get_tokenizer()
orig_dataset = get_dataset(orig_tokenizer)
schema_tokenizer = get_schema_tokenizer()
schema_dataset = get_schema_dataset(
    schema_tokenizer,
    orig_dataset.action_label_to_id
)
schema_dataloader = get_schema_dataloader(schema_dataset)
