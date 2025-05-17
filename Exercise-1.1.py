# -*- coding: utf-8 -*-
"""
Created on Sat May 17 11:37:26 2025

@author: Chong Ming
"""

######################Task 1

import re
from typing import Callable

import os

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

with open("openai_api_key", "r") as file:
    openai_api_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_api_key

from openai import OpenAI

class LLMPrivacyWrapper:
    def __init__(self, replacement_map: dict):
        """
        Initializes the wrapper with a mapping of words to their replacements.

        replacement_map: Dictionary where keys are sensitive words and values are their innocent replacements.
        """
        self.replacement_map = replacement_map
        self.reverse_map = {v: k for k, v in replacement_map.items()}  # Reverse for decoding
        
       

    def encode(self, text: str) -> str:
        """
        Replaces sensitive words with innocent alternatives.

        text: Input text containing sensitive words.

        return: Encoded text with innocent replacements.
        """
        # <YOUR CODE HERE>
        
        for sensitive, innocent in self.replacement_map.items():
        # Use regex to match whole words only (case-sensitive)
            pattern = r'\b' + re.escape(sensitive) + r'\b'
            text = re.sub(pattern, innocent, text)
            
        return text


    def decode(self, text: str) -> str:
        """
        Restores original sensitive words in the text.

        :param text: Encoded text with innocent replacements.
        :return: Decoded text with original words restored.
        """
        # <YOUR CODE HERE>
        for innocent, sensitive in self.reverse_map.items():
            pattern = r'\b' + re.escape(innocent) + r'\b'
            text = re.sub(pattern, sensitive, text)
            
        return text
     

    
    def answer_with_llm_DUMMY(self, text: str, client, model: str) -> str:
        """
        Encodes text, sends it to the LLM, and then decodes the response.

        :param text: The original input text.
        :param llm_call: A callable function simulating an LLM response.
        :return: The final processed text with original words restored.
        """
        # <YOUR CODE HERE>
        
        encoded_text = self.encode(text)
       
        llm_response = client(encoded_text, model)
        decoded_response = self.decode(llm_response)
        return decoded_response
    
    def answer_with_llm(self, text: str, client, model: str) -> str:
        """
        Encodes text, sends it to the LLM, and then decodes the response.

        :param text: The original input text.
        :param llm_call: A callable function simulating an LLM response.
        :return: The final processed text with original words restored.
        """
        # <YOUR CODE HERE>
        encoded_text = self.encode(text)
        print("\n")
        print("4: Inside answer_with_llm function and display the encoded text: ")
        print("=========================================================================")
        print(encoded_text)
        
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": encoded_text}]
        

        
        )
        print("\n")
        print("5: This is model response before decrypting: ")
        print("=========================================================================")
        print(response.choices[0].message.content)
        return self.decode(response.choices[0].message.content)

def dummy_llm_call(text, model):
    # Simulate an LLM response by appending a message
    return f"LLM({model}) received: {text}"
    
        
# Step 1: Define replacement mapping
replacement_map = {
    "Apple": "Banana",
    "apple":"banana",
    "Apple": "Bananas",
    "apple":"bananas",
    "John": "Jane",
}

# Step 2: Create wrapper instance
wrapper = LLMPrivacyWrapper(replacement_map)

# Step 3: Input text
prompt = "John picked a ripe apple from the tree, savoring its crisp sweetness under the autumn sun." 
        
# Step 4: Test encode
encoded = wrapper.encode(prompt)
print("1: Encoded:", encoded)
# Output: "Jane from Banana is visiting Narnia. Pineapple is not a company. Johnny likes apples."

# Step 5: Test decode (should bring back original)
decoded = wrapper.decode(encoded)
print("2: Decoded:", decoded)
# Output: "John from Apple is visiting Singapore. Pineapple is not a company. Johnny likes apples."

# Step 6: Full roundtrip through LLM
response = wrapper.answer_with_llm_DUMMY(prompt, dummy_llm_call, "gpt-4")   
print("\n")
print("3: Let's get a response from a dummy model. Getting response from LLM Model")
print(response) 


client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

result = wrapper.answer_with_llm(prompt,client=client, model=model)
print("\n")
print("6: This is the decoded message after communicating with LLM Model")
print("=========================================================================")
print(result)    