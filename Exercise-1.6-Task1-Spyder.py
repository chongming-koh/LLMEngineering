# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 00:06:57 2025

@author: Chong Ming
"""

import os
from collections import Counter
from tqdm import tqdm

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

from openai import OpenAI

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# llm_model = ""
'''
Step 1: I choose my LLM to run this case. Using 3 models
'''
llm_models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct"
]

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
                    system_prompt,
                    max_tokens=512,
                    client=nebius_client,
                    model=llm_models,
                    prettify=False,
                    temperature=0.7) -> str:

    messages = []
    #print("\nModel Type: "+model+"\n")

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
    
#To find the log Probability of the suggestions after my prompt

def answer_with_logprobs(prompt: str,
                         system_prompt="You are a helpful assistant",
                         max_tokens=512,
                         client=nebius_client,
                         model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                         temperature=0.6):
    # Adapt to your logprobs extraction logic as needed
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=5,
    )
    return completion

'''
Step 2: Define my prompts for this exercise
Using 2 temperatures
Each promt will run 20 times (n_trails) to temp 0.6 first follow by 1 (temperatures)

'''
system_prompt="You are a helpful assistant"

prompts = {
    "Name": "Suggest a name for a Disney cartoon character. Only output the name.",
    "Occupation": "Suggest an occupation for a Disney cartoon character. Only output the name of the occupation.",
    "Hobby": "Suggest a hobby for a Disney cartoon character. Only output the name of the hobby."
}
'''
For each model and prompt, count how many unique answers I get. 
Question 1: Are certain names/occupations always being repeated?
Question 2: Does the model have “favorites” it repeats a lot?
'''

n_trials = 5
temperatures = [0.6, 1.0]

all_results = {} #Creates an empty dictionary called results.

for model in llm_models:
    all_results[model] = {} #For each model, adds a new dictionary
    for temp in temperatures:
        all_results[model][temp] = {} #For each temperature set up another dictionary layer to keep results separate for each temperature.
        print(f"\nRunning model: {model} | temperature: {temp}\n")
        for prompt_type, prompt in prompts.items(): #Loop through each prompt
            answers = [] #empty list called answers to store the LLM’s responses for that prompt
            for _ in tqdm(range(n_trials), desc=f"{prompt_type} (T={temp})"):
                response = answer_with_llm(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    model=model,
                    temperature=temp
                )
                answers.append(response.strip()) #Store the returned (and stripped) response in answers variable.
            all_results[model][temp][prompt_type] = answers #save the list of responses under each model, temperature, and prompt type in all_results

'''
analyzes and prints how often each answer appears for every model, temperature, and prompt type
To see if some names/occupations/hobbies are being repeated a lot or if there is good variety
'''
print("\n==== Diversity Counts ====\n")
for model in all_results:
    for temp in temperatures:
        for prompt_type in prompts:
            responses = all_results[model][temp][prompt_type]
            counts = Counter(responses)
            print(f"Model: {model} | Temperature: {temp} | Prompt: {prompt_type}")
            print(counts)
            print("-" * 40)    
        
        
'''
Function Prepares lists to collect the folowing data:

generated_tokens list: the tokens (words or parts of words) the LLM actually generated.
generated_logprobs: their log probabilities (how likely the model thought they were).

top_tokens and top_logprobs: lists to collect the top alternative tokens and their probabilities at each step, besides the generated one.

Fills the lists by:
For each generated token, it adds the token and its log probability.
Creates a pandas DataFrame and puts the generated tokens and logprobs into columns.
Adds extra columns for each of the top alternative tokens and their logprobs (e.g., 0_token, 0_logp, 1_token, 1_logp, etc.).
Returns the DataFrame for easy analysis or display.
'''
# ------ Logprobs Analysis ------
def logprobs_to_table(logprobs_content):
    generated_tokens = [] #store the actual tokens the LLM generated.
    generated_logprobs = [] #will store the log probability (a measure of "confidence") for each generated token.

    #Creates lists of empty lists to hold the alternative ("top") tokens and their log probabilities for each position in the sequence.
    # The list contains the select token which i already captured. I just want to collect and store the alternative ones. Hence I "-1"
    top_tokens = [[] for _ in range(len(logprobs_content[0].top_logprobs) - 1)]
    top_logprobs = [[] for _ in range(len(logprobs_content[0].top_logprobs) - 1)]

    '''
    Loops through each token position in the generated sequence.
    1-Adds the actual generated token and its log probability to their respective lists.
    2-For the alternative tokens at this position (entry.top_logprobs[1:]):\
    3-Adds each alternative token (top_logprob.token) to the appropriate sub-list in top_tokens.
    4-Adds the alternative's log probability (top_logprob.logprob) to the appropriate sub-list in top_logprobs.
    '''
    for entry in logprobs_content:
        generated_tokens.append(entry.token)
        generated_logprobs.append(entry.logprob)
        for j, top_logprob in enumerate(entry.top_logprobs[1:]):
            top_tokens[j].append(top_logprob.token)
            top_logprobs[j].append(top_logprob.logprob)
    import pandas as pd
    df = pd.DataFrame({
        "gen_token": generated_tokens,
        "gen_logp": generated_logprobs
    })
    for j in range(len(top_tokens)):
        df[f"{j}_token"] = top_tokens[j]
        df[f"{j}_logp"] = top_logprobs[j]
    return df

'''
Call function to analyze LLM model outputs for different prompts.
Loops through all your models and prompts:

For each model and each prompt type (like "name", "occupation", "hobby"):
Calls answer_with_logprobs() to get the model’s output with log probability info using temperature 0.6.
Process the output into a table using your function:

If successful, prints the DataFrame (so you can see which words/tokens were chosen and how likely they were).
If there’s an error (e.g., if logprob data isn’t present), prints an error message.
'''

print("\n==== Logprob Analysis (temperature=0.6) ====\n")
for model in llm_models:
    for prompt_type, prompt in prompts.items():
        logprob_completion = answer_with_logprobs(
            prompt=prompt,
            model=model,
            temperature=0.6
        )
        print(f"\nModel: {model} | Prompt: {prompt_type}")
        try:
            logprobs_df = logprobs_to_table(logprob_completion.choices[0].logprobs.content)
            print(logprobs_df)
        except Exception as e:
            print("Error processing logprobs:", e)
            
print("\n==== Logprob Analysis (temperature=1) ====\n")
for model in llm_models:
    for prompt_type, prompt in prompts.items():
        logprob_completion = answer_with_logprobs(
            prompt=prompt,
            model=model,
            temperature=1
        )
        print(f"\nModel: {model} | Prompt: {prompt_type}")
        try:
            logprobs_df = logprobs_to_table(logprob_completion.choices[0].logprobs.content)
            print(logprobs_df)
        except Exception as e:
            print("Error processing logprobs:", e)            