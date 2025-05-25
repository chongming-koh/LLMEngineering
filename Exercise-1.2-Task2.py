# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:51:28 2025

@author: Chong Ming
"""

import tiktoken

#------------------------------Step 1--------------------------
# Option 1: Use the encoding for GPT-4/GPT-3.5 models
# encoding = tiktoken.get_encoding("cl100k_base")

# Option 2: Use model-specific encoding (if you know your model)
encoding = tiktoken.encoding_for_model("gpt-4")

excerpts = {
    'English': '''
Once upon a time, in a small village, there lived some kind peasants. They worked hard every day, taking care of their fields and animals. The peasants had cows, chickens, goats, and a big brown dog named Buddy.

Every morning, the peasants woke up early. They fed the chickens, who clucked happily. They milked the cows, who gave them fresh, sweet milk. The goats liked to jump and play, making everyone laugh. Buddy, the dog, helped keep the animals safe.

One day, a storm came. The wind blew hard, and the rain fell heavily. The peasants quickly brought all the animals into the big barn. Inside, it was warm and dry. The cows lay down on the soft hay. The chickens snuggled together on their perch. The goats found a corner to rest, and Buddy lay by the door, watching over everyone.

The storm lasted all night, but the peasants and their animals were safe in the barn. In the morning, the sun came out, and the storm was gone. The peasants opened the barn doors, and the animals went outside, happy and free.

The peasants fixed the fences and cleaned the barn. They worked together, helping each other and their animals. At the end of the day, they sat under a big tree, watching the sunset. Buddy lay at their feet, and the animals grazed nearby.

The village was peaceful, and the peasants were happy. They knew that as long as they took care of their animals and each other, they would always have a good life.

And so, they lived happily ever after.
''',
    'French': '''
Il était une fois, dans un petit village, de gentils paysans. Ils travaillaient dur chaque jour, prenant soin de leurs champs et de leurs animaux. Les paysans avaient des vaches, des poules, des chèvres et un gros chien brun nommé Buddy.

Chaque matin, les paysans se réveillaient tôt. Ils nourrissaient les poules qui glousaient joyeusement. Ils traitaient les vaches qui leur donnaient du lait frais et sucré. Les chèvres aimaient sauter et jouer, faisant rire tout le monde. Buddy, le chien, a aidé à assurer la sécurité des animaux.

Un jour, une tempête est arrivée. Le vent soufflait fort et la pluie tombait abondamment. Les paysans ont rapidement amené tous les animaux dans la grande grange. À l’intérieur, il faisait chaud et sec. Les vaches se couchent sur le foin moelleux. Les poules se blottissaient les unes contre les autres sur leur perchoir. Les chèvres trouvèrent un coin pour se reposer et Buddy resta allongé près de la porte, veillant sur tout le monde.

La tempête a duré toute la nuit, mais les paysans et leurs animaux étaient en sécurité dans la grange. Le matin, le soleil s'est levé et la tempête s'est calmée. Les paysans ouvrirent les portes de la grange et les animaux sortirent, heureux et libres.

Les paysans réparèrent les clôtures et nettoyèrent la grange. Ils ont travaillé ensemble, s'entraidant ainsi que leurs animaux. À la fin de la journée, ils étaient assis sous un grand arbre et regardaient le coucher du soleil. Buddy gisait à leurs pieds et les animaux paissaient à proximité.

Le village était paisible et les paysans étaient heureux. Ils savaient que tant qu’ils prendraient soin de leurs animaux et les uns des autres, ils auraient toujours une belle vie.

Et ainsi, ils vécurent heureux pour toujours.
''',
    'Swahili': '''
Hapo zamani za kale, katika kijiji kidogo waliishi wakulima wa aina fulani. Walifanya kazi kwa bidii kila siku, wakitunza mashamba na wanyama wao. Wakulima walikuwa na ng'ombe, kuku, mbuzi, na mbwa mkubwa wa kahawia aliyeitwa Buddy.

Kila asubuhi, wakulima waliamka mapema. Walilisha kuku, ambao waliruka kwa furaha. Wakakamua ng'ombe, ambaye aliwapa maziwa safi, matamu. Mbuzi walipenda kuruka na kucheza, na kufanya kila mtu acheke. Buddy, mbwa, alisaidia kuweka wanyama salama.

Siku moja, dhoruba ilikuja. Upepo ulivuma kwa nguvu, na mvua ikanyesha sana. Wakulima haraka wakaleta wanyama wote kwenye zizi kubwa. Ndani, kulikuwa na joto na kavu. Ng'ombe walilala kwenye nyasi laini. Kuku walijibana kwenye sangara wao. Mbuzi walipata kona ya kupumzika, na Buddy akalala karibu na mlango, akiangalia kila mtu.

Dhoruba hiyo ilidumu usiku kucha, lakini wakulima na wanyama wao walikuwa salama ghalani. Asubuhi, jua lilitoka, na dhoruba ikatoweka. Wakulima walifungua milango ya ghalani, na wanyama wakatoka nje, wakiwa na furaha na huru.

Wakulima walitengeneza ua na kusafisha ghala. Walifanya kazi pamoja, kusaidiana na wanyama wao. Mwisho wa siku walikaa chini ya mti mkubwa wakitazama machweo ya jua. Rafiki alilala miguuni mwao, na wanyama walilisha karibu.

Kijiji kilikuwa na amani, na wakulima walikuwa na furaha. Walijua kwamba maadamu wanatunza wanyama wao na kila mmoja wao, wangekuwa na maisha mazuri kila wakati.

Na hivyo, waliishi kwa furaha milele.
''',
    'Chinese': '''
從前，在一個小村莊裡，住著一些善良的農夫。他們每天努力工作，照顧他們的田地和動物。農民養了牛、雞、山羊和一隻名叫巴迪的棕色大狗。

每天早上，農民們都早早起床。他們餵雞，雞高興地咯咯叫。他們給乳牛擠奶，乳牛給它們新鮮、甜的牛奶。山羊們喜歡跳來跳去，玩耍，逗得大家哈哈大笑。巴迪狗幫助保護動物的安全。

有一天，一場暴風雨來了。風刮得很大，雨也下得很大。農夫很快就把所有的牲畜都帶進了大穀倉。裡面溫暖而乾燥。奶牛躺在柔軟的乾草上。雞們在棲木上依偎在一起。山羊們找到了一個角落休息，巴迪躺在門口，看著大家。

暴風雨持續了一夜，但農民和他們的動物在穀倉裡很安全。早上，太陽出來了，暴風雨也過去了。農民打開穀倉的門，動物們快樂自由地走出去。

農民們修好了柵欄，打掃了穀倉。他們一起工作，互相幫助，也幫助他們的動物。一天結束時，他們坐在一棵大樹下，看著日落。巴迪躺在他們腳邊，動物們在附近吃草。

村子裡太平了，農夫們都幸福了。他們知道，只要照顧好自己的動物和彼此，他們就會永遠過著美好的生活。

就這樣，他們從此過上了幸福的生活。
'''
}

for lang, text in excerpts.items():
    tokens = encoding.encode(text)
    print(f"{lang}: {len(tokens)} tokens")
    

chinese_text = excerpts["Chinese"]
encoded_tokens = encoding.encode(chinese_text)

# Print out token IDs and their corresponding strings
for token in encoded_tokens[:40]:  # Show first 40 tokens as a sample
    print(f'{token}: "{encoding.decode([token])}"')
    

from transformers import AutoTokenizer

# Load Chinese-optimized tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

chinese_text = excerpts["Chinese"]

# Tokenize and print tokens (show first 40 for brevity)
tokens = tokenizer.tokenize(chinese_text)
for idx, token in enumerate(tokens[:40]):
    print(f"{idx}: {token}")