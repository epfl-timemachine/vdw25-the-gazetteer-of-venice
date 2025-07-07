import re

with open("tassini_clean\\tassini_pages_without_index.txt", 'r', encoding='utf-8') as f:
    text = f.read()


""" #Segement per page
page_pattern = r'[=]+?.+?[=]+?\n([\s\S]*?)(?=[=]+?.+?[=]+?)'
matches = re.findall(page_pattern, text)

print(matches[0]) """

def is_valid_group1(text, first_place):
    if text.split(' ', 1)[0] == first_place:
        return True
    return False

def extract_valid_matches(text, pattern, first_place):
    valid_matches = []
    for match in re.findall(pattern, text):
        group1 = match[0]
        group2 = match[1]

        if is_valid_group1(group1, first_place):
            valid_matches.append((group1, group2))
        else:
            print("Skipping invalid group1:", repr(group1))
    return valid_matches



#For each page, segement per index (there will be still within certain indexes two or more places).
#Each entity is linked with page number(s)

entity_pattern_1 = r'[A-Z]\s+?indice\s+?\n([\s\S]+?)(?=[A-Z]\s+?indice)'
matches = re.findall(entity_pattern_1, text)


#Separate inside the matches if there are two of them
all_entities = []

#matches = matches[1:2]

for match in matches:
    entity_pattern_2 = r'(.+\(.+\)[\s\S]+?\.[\s]*\n)([\s\S]+?)(?=.+\(.+\)[\s\S]+?\.[\s]*\n|\Z)'

    first_place = match.split(' ', 1)[0]
    print(first_place)
    place_matches = extract_valid_matches(match, entity_pattern_2, first_place)

    for place_match in place_matches:
        with open("out/text_with_places.txt", "a", encoding='utf-8') as text_file:
            text_file.write("%" + place_match[0].strip() + "%\n")
            text_file.write(place_match[1].strip() + "\n&\n")

