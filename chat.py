import argparse
import json
import numpy as np
import os
import random
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
from slot_extraction import get_entity


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

def make_api_call(task, slots):
    def to_title_case(key):
        return key.title().replace(" ", "").strip()
    return api.call_api(
        task,
        constraints=[{
            to_title_case(k): v for k, v in slots.items()
        }],
    )

def handle_api_call(task, request_type, slots):
    assert request_type in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]
    
    # Sets RequestType to "", "Book", or "Check".
    slots["RequestType"] = request_type[1:-1][len("QUERY_"):].title()
    print(f"You provided: {slots}")

    db_response = make_api_call(task, slots)
    return db_response

    
def chat(domain, task):
    import os
    print(os.path.dirname(os.path.abspath(__file__)))
    DOMAIN = json.load(
        open(os.path.join(CURR_DIR, "STAR", "tasks", task, f"{task}.json"), "r")
    )
    MODEL = load_saved_model(task=task)

    history = ""
    prev_sys_response = ""
    slots = dict()

    while True:
        # Fetch user input
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

        # Handle DB calls separately
        if system_response in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]:
            api_response = handle_api_call(task=task, request_type=system_response, slots=slots)
            print(f"API: {api_response}")
            
        prev_sys_response = system_response
        print(f"SYS: >> {remove_entity_annotations(system_response)}")

        system_response_model = get_turn_str(
            "Agent", remove_entity_annotations(system_response)
        )
        history += system_response_model


if __name__ == "__main__":
    pass