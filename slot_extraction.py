from transformers import pipeline
from dataclasses import dataclass


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
            return f"Objective: Extract {self.slot_name}\nInput: {self.sys_query} {self.usr_response}\nOutput: {self.result}"
        else:
            return f"Objective: Extract {self.slot_name}\nInput: {self.sys_query} {self.usr_response}\nOutput:"


ex1 = PrimingExample("number of people coming to the party", "How many people are coming to the party?", "We might be like people. I'm not too sure.", "8")
ex2 = PrimingExample("destination", "Where would you like to go?", "I would like to go home, please", "home")
ex3 = PrimingExample("driver", "Who is the driver", "Philip", "Philip" )

PRIME = ex1.get_input() + "\n" + ex2.get_input() + "\n" + ex3.get_input()

def extract_slots(sys_query, usr_response, *slot_names):
    slots = dict()
    for slot_name in slot_names:
        input_example = PrimingExample(slot_name, sys_query, usr_response)
        gpt_input = prime + "\n" + input_example.get_input()
        slots[slot_name] = generator(
            gpt_input,
            do_sample=False,
            max_length=5,
            min_length=1
        )[0]["generated_text"][len(gpt_input) + 1:]
    return slots

