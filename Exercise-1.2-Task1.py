# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:29:42 2025

@author: Chong Ming
"""

import tiktoken

#------------------------------Step 1--------------------------
# Option 1: Use the encoding for GPT-4/GPT-3.5 models
# encoding = tiktoken.get_encoding("cl100k_base")

# Option 2: Use model-specific encoding (if you know your model)
encoding = tiktoken.encoding_for_model("gpt-4")

#------------------------------Step 2--------------------------
token_index = 1011
subword = encoding.decode([token_index])
print(f'Token {token_index} corresponds to: "{subword}"')

token_ids = [0, 1, 100, 50256, 10000, 10101, 15000, 20000, 25000, 30000]
for token_id in token_ids:
    subword = encoding.decode([token_id])
    print(f'Token ID {token_id}: "{subword}"')
    
