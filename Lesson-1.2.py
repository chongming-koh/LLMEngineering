# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:15:33 2025

@author: Chong Ming
"""

import tiktoken
import transformers
encoding = tiktoken.encoding_for_model('gpt-4o-mini')

initial_string = 'Darth Vader was born on Tatooine.'

# Tokenization
encoded_string = encoding.encode(initial_string)
print('After tokenization (encoded string): ', encoded_string)

# Tokenization
print('Decoding back: ', encoding.decode(encoded_string))

# Decoding each token:
for token in encoded_string:
    print(f'{token}: "{encoding.decode([token])}"')
    
print("\n")
initial_string = 'D D!D'

# Tokenization
encoded_string = encoding.encode(initial_string)
print('After tokenization (encoded string): ', encoded_string)

# Tokenization
print('Decoding back: ', encoding.decode(encoded_string))

# Decoding each token:
for token in encoded_string:
    print(f'{token}: "{encoding.decode([token])}"')

print("\n")
python_code = """import pandas as pd
import matplotlib.pyplot as plt

# Plotting the histogram
plt.hist(df['values'], bins=5, edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.show()"""

# Tokenization
encoded_string = encoding.encode(python_code)

# Decoding each token:
for token in encoded_string[:40]:
    print(f'{token}: {encoding.decode([token])}')
    
print("\n")
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Darth Vader was born on Tatooine."

tokenizer.tokenize(prompt)

print("\n")
initial_string = '12345678'

# Tokenization
encoded_string = encoding.encode(initial_string)
print('After tokenization (encoded string): ', encoded_string)

# Decoding each token:
for token in encoded_string:
    print(f'{token}: {encoding.decode([token])}')
    

import os

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()
os.environ["NEBIUS_API_KEY"] = nebius_api_key

with open("openai_api_key", "r") as file:
    openai_api_key = file.read().strip()
os.environ["OPENAI_API_KEY"] = openai_api_key

with open("hf_api_key", "r") as file:
    hf_api_key = file.read().strip()
os.environ["HF_TOKEN"] = hf_api_key

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from openai import OpenAI

# Nebius uses the same OpenAI() class, but with additional details
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
completion = client.chat.completions.create(
    model=model,
    messages=[
    {
        "role": "system",
        "content": """You're an expert in floating point computations."""
    },
    {
        "role": "user",
        "content": """7.24*19.13 ="""
    },
    ],
    max_tokens=512
)

completion.choices[0].message.content
print(completion.choices[0].message.content)

''' Hugging Face connection codes
import requests

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

payload = {
    "inputs": "7.24*19.13 =",
    "parameters": {"max_new_tokens": 32}
}

response = requests.post(API_URL, headers=headers, json=payload)
print("Status code:", response.status_code)
print("Raw response:", response.text)
result = response.json()

print(result)
'''
