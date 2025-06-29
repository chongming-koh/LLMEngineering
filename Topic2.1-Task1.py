# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 20:09:07 2025

@author: Chong Ming
"""

import pandas as pd
import json
import os

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

from openai import OpenAI

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

#chosenLLMModel = "meta-llama/Meta-Llama-3.1-8B-Instruct"
chosenLLMModel = "meta-llama/Meta-Llama-3.1-70B-Instruct"
#chosenLLMModel ="Qwen/Qwen3-32B"

'''
Function to parse the excel using into Nebius API
'''
def parse_press_release(text):
   
    system_prompt = """You are an assistant that extracts structured data from press releases. 
    Your output must be a valid JSON object with the following keys: name, date, n_speakers, n_participants, price."""
    
    user_prompt = f"""
    Extract the following fields from the press release below:
    - name
    - date
    - n_speakers
    - n_participants
    - price
    
    Output format Example 1 (JSON only):
    
    {{
      "name": "...",
      "date": "...",
      "n_speakers": ...,
      "n_participants": ...,
      "price": "..."
    }}
    
    Output format Example 2 (JSON only):
    
    {{
      "name": "...",
      "date": "08.10.2023-09.10.202",
      "n_speakers": 2,
      "n_participants": 2,
      "price": "140.00"
    }}
    
    Output format Example 3 (JSON only):
    
    {{
      "name": "XXX",
      "date": "15.10.2023",
      "n_speakers": 20,
      "n_participants": 20,
      "price": "140"
    }}
    
    Press Release:
    {text}
    """

    # Get raw response from LLM
    #I want the model to stick strictly to the expected format. So temperate is set to zero
    response_text = answer_with_llm(prompt=user_prompt, system_prompt=system_prompt, temperature=0)

    # Try to parse JSON response from model
    try:
        parsed_output = json.loads(response_text)
        return parsed_output
    except json.JSONDecodeError:
        #print("================Exception======================\n")
        print("Failed to parse JSON:", response_text)
        return {}
    
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
                    model=chosenLLMModel,
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
    
    # Add this line to inspect the LLM output
    #print("LLM response:\n", completion.choices[0].message.content)

    if prettify:
        return prettify_string(completion.choices[0].message.content)
    else:
        return completion.choices[0].message.content

# Load the dataset
pr_df = pd.read_csv("press_release_extraction - press_release_extraction.csv")

# Preview the first row
#print(pr_df.head())
#print(pr_df.pr_parsed[0])


'''
The code set below given by the Task is to run the evaluation loop to check results
'''
parsed_list = []
fields = {
    "name": str,
    "date": str,
    "n_speakers": int,
    "n_participants": int,
    "price": str
}
correct_fields = 0

for row in pr_df.itertuples():
    parsed_release = parse_press_release(row.pr_text)
    parsed_list.append(json.dumps(parsed_release, indent=4))
    golden = json.loads(row.pr_parsed)
    
    #Golden field holds the row values under "pr_parsed" column while  'parsed_field" holds the json format received from LLM after the prompt
    for field, field_type in fields.items():
        golden_field = golden[field]
        parsed_field = parsed_release.get(field)

        try:
            parsed_field = field_type(parsed_field)
        except (ValueError, TypeError):
            pass

        if golden_field == parsed_field:
            correct_fields += 1
        else:
            print(f"For {golden['name']} {field} {parsed_field} doesn't match {golden_field}")

print(f"\n✅ Correctly extracted {correct_fields} out of {5 * len(pr_df)} fields")

'''
Analyze Results
by saving the outputs for visual comparison:
'''
pr_df["parsed"] = parsed_list
pr_df.to_csv("with_results.csv", index=False)

