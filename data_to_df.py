import pandas as pd
import re
from functools import reduce

filename = 'nlu.md'
with open(filename, 'r') as f:
    content = f.read()

content = content.split("## ")
content = [a for a in content if (re.findall("intent:", a))]

def remove_entities(texts):
    temp = {}
    new_texts = []
    for text in texts:
        entity = re.findall(r'\[(.*?)\]', text)
        to_remove = re.findall(r'(\[.*?\]\(.*?\))', text)
        
        for original, replacement in zip(to_remove, entity):
            temp[original] = replacement

    for text in texts:
        new_text = translator(text, temp)
        new_texts.append(new_text)

    return new_texts

def translator(s, temp):
    for original in temp:
        s = s.replace(original, temp[original])
    
    return s

df = {
    "text": [],
    "intent": []
}

for intent in content:
    intention = re.findall(r'intent:(.*?)\n', intent)[0]

    texts = re.findall(r'- (.*)', intent)
    texts = remove_entities(texts)
    
    for each_text in texts:
        df['text'].append(each_text)
        df['intent'].append(intention)

df = pd.DataFrame(df)
df.to_csv("nlu.csv", index=False)
