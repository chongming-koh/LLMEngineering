# -*- coding: utf-8 -*-
"""
Created on Sat May 17 16:05:23 2025

@author: Chong Ming
"""

import os

with open("nebius_api_key", "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

with open("openai_api_key", "r") as file:
    openai_api_key = file.read().strip()

os.environ["OPENAI_API_KEY"] = openai_api_key

from openai import OpenAI

##### This is for Image support and generating using prompt
from PIL import Image
from io import BytesIO
import base64
import json
import matplotlib.pyplot as plt

client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

# Action 1: Create function to generate image
def generate_image(prompt):
    response = client.images.generate(         # this line will return an ImagesResponse object, not a Python dictionary. Need to use JSON method to parse
        model="black-forest-labs/flux-dev",
        prompt=prompt,
        n=1,
        size="512x512",
        response_format="b64_json"
    )
    response_json = response.to_json()  # Get JSON string
    response_data = json.loads(response_json)  # Convert to dict
    img_data = response_data['data'][0]['b64_json']
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    return img

# Action 2: Create function to describe image
def describe_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct",  
        messages=messages,
        max_tokens=60,
    )
    return response.choices[0].message.content

# Action 3: Create loop to loop image and decibing image based on number of rounds I want when I call this function
def broken_telephone(starting_prompt, n_rounds):
    history = []  # To store tuples of (text, image) Action 6: Define history variable
    current_text = starting_prompt
    print(f"Round 0 (Start Prompt): {current_text}")

    for round_num in range(1, n_rounds + 1):
        # Step 1: Generate an image from the current text
        img = generate_image(current_text)
        img.show()  # Optional: to see each image
        img.save(f"round_{round_num}.png")  # Save each round's image for reference
        history.append((None, img))  # Store image only. Action 4: Added this code to collect all history of images

        # Step 2: Describe the generated image to produce new text
        current_text = describe_image(img)
        print(f"Round {round_num} (Generated Description): {current_text}")
        history.append((current_text, None))  # Store description only. Action 5: Added this code to collect all history of descriptions

    print("Final Output:", current_text)
    
    # Action 6: Print all history I collected earlier
    print("\n--- Summary of All Rounds ---")
    for i, (text, image) in enumerate(history):
        if text:
            print(f"Round {i}: TEXT: {text}")
        if image:
            print(f"Round {i}: IMAGE: [Saved as round_{i//2 + 1}.png]")


    # Optionally: Return history for further analysis
    return history

# Action 7: Display all history of image and description for easier comparisons
def show_history(history):
    for idx, (text, img) in enumerate(history):
        print(f"Round {idx}:")
        if text:
            print("Text:", text)
        if img:
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
# These codes are for testing purpose
'''
myprompt="A cup with imprinted image of a crocodile"
img = generate_image(myprompt)
img.show()  # View the generated image

description = describe_image(img)
print("LLM Description:", description)
'''

history = broken_telephone("Adventure awaits on every winding road.", 3)
show_history(history)
