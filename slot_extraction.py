import re
from dataclasses import dataclass

from transformers import pipeline

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

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
            return f"Objective: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput: {self.result}"
        else:
            return f"Objective: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput:"


def get_entity(slot_name, sys_query, usr_response):
    ex1 = PrimingExample("number of people", "How many people are coming to the party?", "The number of people is we might be like 8 people. I'm not too sure.", "8")
    ex2 = PrimingExample("destination", "Where would you like to go?", "The destination is I would like to go home, please", "home")
    ex3 = PrimingExample("destination", "Where would you like to go?", "The destination is home", "home")
    ex4 = PrimingExample("name", "What is your name?", "The My name is Max", "Max" )

    prime = ex1.get_input() + "\n" + ex2.get_input() + "\n" + ex3.get_input() + "\n"+ ex4.get_input()
    
    input_example = PrimingExample(slot_name, sys_query, usr_response)
    
    gpt_input = prime + "\n" + input_example.get_input()
    out = generator(
        gpt_input,
        do_sample=False,
        max_length=30,
        min_length=3
    )[0]["generated_text"][len(gpt_input) + 1:]
    
    entity = re.sub(r'\W+', '', usr_response[usr_response.find(out):].split(" ")[0])
    return entity


def to_db_query_string(slots):
    utt_text = "[QUERY] "
    for slot_key, slot_value in slots.items():
        utt_text += f"{slot_key} = {slot_value} ; "
    return utt_text

def to_db_result_string(result):
    utt_text = "[RESULT] "
    if 'Item' not in utt:
        utt_text += "NO RESULT"
    else:
        for key,val in utt['Item'].items():
            utt_text += "{} = {} ; ".format(key, val)
