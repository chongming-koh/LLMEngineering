# -*- coding: utf-8 -*-
"""
Created on Sat May 24 22:39:08 2025

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

    print("=================================================\n")
    print (messages)
    print(model)
    print(max_tokens)

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
    
    
result = answer_with_llm(
    r"""What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \sqrt{x^2 + 18x + 45}$ ?""",
    model="meta-llama/Meta-Llama-3.1-405B-Instruct",
    
    )
print(result)