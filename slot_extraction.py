import re
from dataclasses import dataclass

# from transformers import pipeline

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


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
    """
    Here "num_tokes" define how many tokens [words] we want as a output for the given dialogue
    """

    # Add more relevant examples
    ex1 = PrimingExample("number of people", "How many people are coming to the party?",
                         "The number of people is we might be like 8 people. I'm not too sure.", "8")
    ex2 = PrimingExample("destination", "Where would you like to go?",
                         "The destination is I would like to go home, please", "home")
    ex3 = PrimingExample(
        "destination", "Where would you like to go?", "The destination is home", "home")
    ex4 = PrimingExample("name", "What is your name?",
                         "The My name is Max", "Max")
    prime = ex1.get_input() + "\n" + ex2.get_input() + "\n" + \
        ex3.get_input() + "\n" + ex4.get_input()

    num_tokes = 3  # Use this some how to make it more better.
    input_example = PrimingExample(slot_name, sys_query, usr_response)
    sequence = prime + "\n" + input_example.get_input()

    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(
        inputs, max_length=inputs.shape[1]+num_tokes, do_sample=True, num_beams=5, early_stopping=True)
    text = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # We get the whole string back along with the entity at the end. So, we need to extract the entity.
    entity = text.split("Output:")[-1].split("\n")[0].strip()
    return entity


def to_db_result_string(slots):
    utt_text = "[RESULT] "
    for slot_key, slot_value in slots.items():
        utt_text += f"{slot_key} = {slot_value} ; "
    return utt_text

# def to_db_result_string(result):
#     utt_text = "[RESULT] "
#     if 'Item' not in utt:
#         utt_text += "NO RESULT"
#     else:
#         for key,val in utt['Item'].items():
#             utt_text += "{} = {} ; ".format(key, val)

# def get_entity(slot_name, sys_query, usr_response):
#     ex1 = PrimingExample("number of people", "How many people are coming to the party?",
#                          "The number of people is we might be like 8 people. I'm not too sure.", "8")
#     ex2 = PrimingExample("destination", "Where would you like to go?",
#                          "The destination is I would like to go home, please", "home")
#     ex3 = PrimingExample(
#         "destination", "Where would you like to go?", "The destination is home", "home")
#     ex4 = PrimingExample("name", "What is your name?",
#                          "The My name is Max", "Max")

#     prime = ex1.get_input() + "\n" + ex2.get_input() + "\n" + \
#         ex3.get_input() + "\n" + ex4.get_input()

#     input_example = PrimingExample(slot_name, sys_query, usr_response)

#     gpt_input = prime + "\n" + input_example.get_input()
#     out = generator(
#         gpt_input,
#         do_sample=False,
#         max_length=30,
#         min_length=3
#     )[0]["generated_text"][len(gpt_input) + 1:]

#     entity = re.sub(
#         r'\W+', '', usr_response[usr_response.find(out):].split(" ")[0])
#     return entity
