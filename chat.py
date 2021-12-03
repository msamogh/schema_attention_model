import argparse
import json
import numpy as np
import os
import yaml
from dataclasses import dataclass
from typing import List
import re
import torch
from pprint import pprint

from typing import Any, Dict, Tuple
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tokenizers import BertWordPieceTokenizer

from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel
from STAR.apis import api

from data_model_utils import CURR_DIR, get_system_action, load_saved_model
from slot_extraction import get_entity, to_db_result_string


with open('messages.yaml', 'r') as f:
    MESSAGES = yaml.load(f)['templates']

def user_utterance_to_model_input(user_input, requested_entity_name):
    if len(user_input) < 8 and ("hello" in user_input or "hi" in user_input or "hey" in user_input):
        user_input = "hello hello"
    user_input = explicitize_user_input(user_input, requested_entity_name)
    user_input = get_turn_str("User", user_input)
    return user_input

def get_requested_entity_if_exists(system_response):
    search_result = re.search(r"%%(.*)%%", system_response)
    if search_result is not None:
        requested_entity = search_result.group(1)
        return requested_entity
    search_result = re.search(r"%(.*)%", system_response)
    if search_result is not None:
        requested_entity = search_result.group(1)
        return requested_entity
    return None

def remove_entity_annotations(system_response):
    system_response = re.sub(r"%%(.*)%%", "", system_response)
    system_response = system_response.replace("%", "")
    return system_response

def explicitize_user_input(user_input, requested_entity_name):
    if requested_entity_name is not None:
        user_input = f"The {requested_entity_name} is {user_input}."
    return user_input

def get_turn_str(speaker, utterance):
    return "[{}] {} [SEP] ".format(speaker, utterance.strip())


def get_db_result_string(task, request_type, slots):
    assert request_type in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]
    
    # Sets RequestType to "", "Book", or "Check".
    slots["request type"] = request_type[1:-1][len("QUERY_"):].title()
    
    def to_title_case(key):
        return key.title().replace(" ", "").strip()

    api_response = api.call_api(
        task,
        constraints=[{
            to_title_case(k): v for k, v in slots.items()
        }],
    )[0]

    db_result_string = to_db_result_string(api_response)
    return db_result_string, api_response

def title_to_snake_case(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

    
def chat(domain, task):
    DOMAIN = json.load(
        open(os.path.join(CURR_DIR, "STAR", "tasks", task, f"{task}.json"), "r")
    )
    MODEL = load_saved_model(task=task)

    history = ""
    prev_sys_response = ""
    slots = dict()
    
    db_results_so_far = {}
    db_result_string = None
    db_result_dict = None

    print(MESSAGES['welcome'])

    while True:
        # Fetch user input
        if db_result_string is not None:
            history += db_result_string
        else:
            user_input_raw = input("USER: >> ").strip().lower()
            requested_entity_name = get_requested_entity_if_exists(prev_sys_response)
            user_input_model = user_utterance_to_model_input(
                user_input_raw,
                requested_entity_name
            )
            if requested_entity_name is not None:
                entity_name = requested_entity_name
                entity_value = get_entity(requested_entity_name, system_response, user_input_raw)
                slots[entity_name] = entity_value
            history += user_input_model

        # Get system action and ask user to rephrase if necessary
        system_action, is_ambiguous = get_system_action(MODEL, history, domain, task)
        if is_ambiguous:
            print(f"SYS: >> Sorry, I didn't catch that. "
                  f"Could you rephrase that more explicitly?\n"
                  f"{remove_entity_annotations(prev_sys_response)}")
            # Undo appending of the latest user utterance.
            history = history[:-len(user_input_model)]
            continue
        system_response = DOMAIN["replies"][system_action]
        if db_result_string is not None:
            system_response = system_response.format(
                **{title_to_snake_case(k): v for k, v in db_results_so_far.items()}
            )

        
        # Reset database result variables
        db_result_string = None
        db_result_dict = None

        # Handle DB calls separately
        if system_response in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]:
            db_result_string, db_result_dict = get_db_result_string(task=task, request_type=system_response, slots=slots)
            db_results_so_far.update(**db_result_dict)
        else:
            print(f"SYS: >> {remove_entity_annotations(system_response)}")
        prev_sys_response = system_response
        system_response_model = get_turn_str(
            "Agent", remove_entity_annotations(prev_sys_response)
        )
        history += system_response_model


if __name__ == "__main__":
    pass
