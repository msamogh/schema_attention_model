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
      return f"History: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput: {self.result}"
    else:
      return f"Objective: Extract {self.slot_name}\nSystem: {self.sys_query}\nUser: {self.usr_response}\nOutput:"


def get_entity(slot_name, sys_query, usr_response):
    ex1 = PrimingExample("number of people coming to the party", "How many people are coming to the party?", "We might be like 8 people. I'm not too sure.", "8")
    ex2 = PrimingExample("destination", "Where would you like to go?", "I would like to go home, please", "home")
    ex3 = PrimingExample("destination", "Where would you like to go?", "I would like to go home, please", "home")
    ex4 = PrimingExample("name", "What is your name?", "My name is Max", "Max" )

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
    return out, entity

