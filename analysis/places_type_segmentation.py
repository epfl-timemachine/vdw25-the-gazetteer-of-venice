import json


with open("out/place_entity_dict.json", "r", encoding='utf-8') as fp:
    place_entity_dict = json.load(fp)

final_dict = dict()

for key, item in place_entity_dict.items():

    if "(" in key:

        print(f"------------{key}------------")
        places_type = key.split("(")[1].split(")")[0].split(",")
        for place_type in places_type:
            final_dict[f"{place_type.strip()} {key.split("(")[0].strip()}"] = {
                "original_name": key,
                "typology": place_type.strip(),
                "location": {

                },
                "years": item,
                "parish?": key.split("(")[1].split(")")[1].strip()[2:-1].strip(),
                "entities": [],
                "page": 0,
            }
    else:
        final_dict[key.strip()[:-1].strip()] = {
            "original_name": key,
            "typology": "unknown",
            "location": {

            },
            "years": item,
            "parish?": "unknown",
            "entities": [],
            "page": 0,
        }

with open("out/places_type_segmentation.json", "w", encoding='utf-8') as fp:
    json.dump(final_dict , fp, ensure_ascii=False) 