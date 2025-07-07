import re
import pandas as pd

def remove_accent(word):
    if word[0] == "È":
        return "E" + word[0:] 
    else:
        return word

with open("tassini_clean\\tassini_only_indexes.txt", 'r', encoding='utf-8') as f:
    text = f.read()


page_pattern = r'(.*?)(?=  |\n)'
matches = re.findall(page_pattern, text)

matches = [i.strip() for i in matches if i.strip() != ""]

matches = sorted(matches, key=lambda s: remove_accent(s))

print(matches)

matches = pd.DataFrame(matches)

matches.to_csv("out/places.csv", index=False)
