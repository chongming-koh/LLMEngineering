# -*- coding: utf-8 -*-
"""
Created on Thu May 22 21:56:24 2025

@author: Chong Ming
"""

import os
import matplotlib.pyplot as plt
from tqdm import tqdm # Creates progress bars for cycles

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

from openai import OpenAI

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

llama_8b_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def prettify_string(text, max_line_length=80):
    """Prints a string with line breaks at spaces to prevent horizontal scrolling.

    Args:
        text: The string to print.
        max_line_length: The maximum length of each line.
    """

    output_lines = []
    lines = text.split("\n") #Split the chunk of text retrieved from LLM into lines
    for line in lines:       #Loop all the lines
        current_line = ""
        words = line.split() #Split the lines into words separate by whitespace
        for word in words:
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += word + " "
            else:
                output_lines.append(current_line.strip())
                current_line = word + " "
        output_lines.append(current_line.strip())  # Append the last line
    return "\n".join(output_lines)

def answer_with_llm(prompt: str,
                    system_prompt="You are a helpful assistant",
                    max_tokens=512,
                    client=nebius_client,
                    model=llama_8b_model,
                    prettify=True,
                    temperature=0.7) -> str:

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    if prettify:
        return prettify_string(completion.choices[0].message.content)
    else:
        return completion.choices[0].message.content

'''    
# Section 1: adjusting the prompt, we are able to change both the focus and the style of generation
result = answer_with_llm("How to create a great villain?")
print(result)

result = answer_with_llm("""How to create a great villain?
Tell this by explaining what not to do to avoid creating a bad villain that wouldn't capture the audience.""")
print(result)

result = answer_with_llm("""How to create a great villain?
Do NOT use words: villain, character, create.""")
print("3rd Prompt\n")
print(result)

result = answer_with_llm("""How do I create a compelling RPG quest?
Don't be too wordy.""")
print("\n")
print("4rd Prompt\n")
print("==============================================================\n")
print(result)

result = answer_with_llm("""How do I create a compelling RPG quest?
Answer in 2-3 sentences.""")
print("\n")
print("5th Prompt\n")
print("==============================================================\n")
print(result)

# Section 2:  influencing LLM's generation with role assignment, describing whom or what it should impersonate in the discussion.
result = answer_with_llm("""You are an experienced dungeon master.
Explain how stealth works in RPGs.""")
print("\n")
print("6th Prompt\n")
print("==============================================================\n")
print(result)

result = answer_with_llm("""You are a cheerful pirate from Baldur's Gate.
Explain how the stealth skill works.""")
print("\n")
print("7th Prompt\n")
print("==============================================================\n")
print(result)

result = answer_with_llm("""You are a game theory expert with a PhD in this topic from Stanford.
Explain how the stealth skill works in RPGs.""")
print("\n")
print("8th Prompt\n")
print("==============================================================\n")
print(result)

'''



# Section 3:  LLM will try to follow your numerical guidelines, but only to a certain degree of accuracy.

'''
n_trials = 20

n_sents = []
for _ in tqdm(range(n_trials)):
    result = answer_with_llm("""How to create a relatable villain?
    Answer in exactly 3 sentences""", prettify=False)

    # We need to subtract 1; otherwise we count the empty substring after the last "."
    n_sents.append(len(result.strip().split(".")) - 1)

plt.hist(n_sents)
plt.show()

n_sents = []
for _ in tqdm(range(n_trials)):
    result = answer_with_llm("""How to create a relatable villain?
    Answer in exactly 50 words""", prettify=False)

    n_sents.append(len(result.split()))

plt.hist(n_sents)
plt.show()

n_sents = []
for _ in tqdm(range(n_trials)):
    result = answer_with_llm("""How to create a relatable villain?
    Answer in between 50 and 100 words""", prettify=False)

    n_sents.append(len(result.split()))

plt.hist(n_sents)
plt.show()

n_sents = []
for _ in tqdm(range(n_trials)):
    result = answer_with_llm("""Create an engaging speech about creating a compelling villain
    The length of the speech should be up to 500 words""",
                             max_tokens=3096, prettify=False)

    n_sents.append(len(result.split()))

plt.hist(n_sents)
plt.show()




# Section 4:  Give a math task to one of the today's LLM, LLM does not just show an answer, but instead produces a solution
result = answer_with_llm("""In the fantasy world of Xu, they have unique math system:
- "a + b" means min(a,b)
- "a*b" means a + b
Solve the equation x*x + 2*x + 1 = 0""",
                         model="meta-llama/Meta-Llama-3.1-405B-Instruct")
print(result)



# Section 5: Extracting answers
result = answer_with_llm("""Saruman is mass-producing steel in the depths of Isengard, aiming for maximum efficiency.
His underground furnaces consume 3.5 tons of wood per hour to sustain the forging and breeding pits.
The Orc lumberjacks can chop 28 tons of wood per day and work for 14 hours a day.
Saruman needs to know if he can keep the furnaces running continuously or if they will run out of fuel.
Question: What is the net surplus of wood per hour?
Provide the step by step solution.
In the end, output only the net surplus after #ANSWER:
If there is deficit instead of surplus, output it as a negative number.
You should output the net surplus as a floating point number with two decimal places, like: 2.31 or -7.00""",
                         model="meta-llama/Meta-Llama-3.1-405B-Instruct")
print(result)

try:
    answer = float(result.split("#ANSWER:")[1].strip())
except:
    answer = None
answer
'''

# Section 6: answer in \boxed{}

result = answer_with_llm(
    "What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \sqrt{x^2 + 18x + 45}$ ?",
    model="meta-llama/Meta-Llama-3.1-405B-Instruct"
    )
print(result)