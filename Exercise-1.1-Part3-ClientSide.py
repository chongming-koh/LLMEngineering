# -*- coding: utf-8 -*-
"""
Created on Sat May 17 20:29:49 2025

@author: Chong Ming
"""

import requests

api_url = "https://2776-34-46-208-119.ngrok-free.app/broken-telephone"

payload = {
    "prompt": "A robot playing the piano on the moon",
    "rounds": 3
}

response = requests.post(api_url, json=payload)
print(response.json())