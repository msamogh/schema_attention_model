import asyncio
import json
import numpy as np
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List
import re
import torch
from pprint import pprint

from typing import Any, Dict, Tuple, Text
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tokenizers import BertWordPieceTokenizer

from data_readers import filter_dataset, NextActionDataset, NextActionSchema
from models import ActionBertModel, SchemaActionBertModel
from STAR.apis import api

from data_model_utils import CURR_DIR, get_system_action, load_saved_model
from slot_extraction import get_entity, to_db_result_string


with open("messages.yaml", "r") as f:
    MESSAGES = yaml.load(f)["templates"]


def user_utterance_to_model_input(user_input, requested_entity_name):
    if len(user_input) < 8 and (
        "hello" in user_input or "hi" in user_input or "hey" in user_input
    ):
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
    slots["request type"] = request_type[1:-1][len("QUERY_") :].title()

    def to_title_case(key):
        return key.title().replace(" ", "").strip()

    api_response = api.call_api(
        task,
        constraints=[{to_title_case(k): v for k, v in slots.items()}],
    )[0]

    db_result_string = to_db_result_string(api_response)
    return db_result_string, api_response


def title_to_snake_case(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


@dataclass
class DialogueContext(object):
    history: List[Text] = field(default_factory=list)
    slots: Dict[Text, Any] = field(default_factory=dict)
    prev_sys_response: Text = field(default="")

    db_results_so_far: Dict[Text, Any] = field(default_factory=dict)


def load_schema_json(task):
    return json.load(
        open(os.path.join(CURR_DIR, "STAR", "tasks", task, f"{task}.json"), "r")
    )


async def handle_web_message(
    context: Dict[Text, Any],
    new_message: Text,
    model: SchemaActionBertModel,
    schema: Dict,
    task: Text,
    domain: Text,
) -> Text:
    ctx = DialogueContext(**context)

    waiting_for_user_input = False
    db_result_string = None
    db_result_dict = None

    while True:
        if db_result_string is not None:
            ctx.history.append(db_result_string)
        else:
            if waiting_for_user_input:
                break
            requested_entity_name = get_requested_entity_if_exists(
                ctx.prev_sys_response
            )
            if requested_entity_name is not None:
                ctx.slots[requested_entity_name] = get_entity(
                    requested_entity_name, ctx.prev_system_response, new_message
                )
            ctx.history.append(
                user_utterance_to_model_input(new_message, requested_entity_name)
            )

        # Get system action and ask user to rephrase if necessary
        system_action, is_ambiguous = await get_system_action(
            model, ctx.history, domain, task
        )
        # Check if db_result_string is None, meaning that the system action is not a query
        if db_result_string is None and is_ambiguous:
            del ctx.history[-1]
            return json(
                {
                    "response": f"{MESSAGES['rephrase']}{remove_entity_annotations(ctx.prev_sys_response)}\n",
                    "updated_context": asdict(ctx),
                }
            )

        # Get system response
        system_response = schema["replies"][system_action]
        if db_result_string is not None:
            system_response = system_response.format(
                **{title_to_snake_case(k): v for k, v in ctx.db_results_so_far.items()}
            )
        prev_sys_response = system_response
        ctx.history.append(
            get_turn_str("Agent", remove_entity_annotations(prev_sys_response))
        )

        # Reset database result variables
        db_result_string = None
        db_result_dict = None

        # Handle DB calls separately
        if system_response in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]:
            db_result_string, db_result_dict = get_db_result_string(
                task=task, request_type=system_response, slots=ctx.slots
            )
            ctx.db_results_so_far.update(**db_result_dict)
        else:
            waiting_for_user_input = True  # as opposed to waiting for an API response
            return json(
                {
                    "response": f"{remove_entity_annotations(system_response)}\n",
                    "updated_context": asdict(ctx),
                }
            )


def chat(domain, task):
    schema = load_schema_json(task)
    model = load_saved_model(task=task)

    ctx = DialogueContext()

    print(MESSAGES["welcome"])

    while True:
        # Fetch user input
        if db_result_string is not None:
            ctx.history.append(db_result_string)
        else:
            user_input_raw = input("USER: >> ").strip().lower()
            requested_entity_name = get_requested_entity_if_exists(prev_sys_response)
            user_input_model = user_utterance_to_model_input(
                user_input_raw, requested_entity_name
            )
            if requested_entity_name is not None:
                entity_name = requested_entity_name
                entity_value = get_entity(
                    requested_entity_name, ctx.prev_system_response, user_input_raw
                )
                ctx.slots[entity_name] = entity_value
            ctx.history.append(user_input_model)

        # Get system action and ask user to rephrase if necessary
        system_action, is_ambiguous = asyncio.run(
            get_system_action(model, ctx.history, domain, task)
        )
        if is_ambiguous:
            print(
                f"SYS: >> Sorry, I didn't catch that. "
                f"Could you rephrase that more explicitly?\n"
                f"{remove_entity_annotations(prev_sys_response)}"
            )
            # Undo appending of the latest user utterance.
            del ctx.history[-1]
            continue
        system_response = schema["replies"][system_action]
        if db_result_string is not None:
            system_response = system_response.format(
                **{title_to_snake_case(k): v for k, v in ctx.db_results_so_far.items()}
            )

        # Reset database result variables
        db_result_string = None
        db_result_dict = None

        # Handle DB calls separately
        if system_response in ["[QUERY]", "[QUERY_BOOK]", "[QUERY_CHECK]"]:
            db_result_string, db_result_dict = get_db_result_string(
                task=task, request_type=system_response, slots=ctx.slots
            )
            ctx.db_results_so_far.update(**db_result_dict)
        else:
            print(f"SYS: >> {remove_entity_annotations(system_response)}")
        prev_sys_response = system_response
        system_response_model = get_turn_str(
            "Agent", remove_entity_annotations(prev_sys_response)
        )
        ctx.history.append(system_response_model)


if __name__ == "__main__":
    pass
