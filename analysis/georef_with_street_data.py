import json
import Levenshtein
from math import sqrt

def closest_string(target, candidates, max_distance=None):
    if not candidates:
        return None, None

    closest = None
    closest_dist = None

    for candidate in candidates:
        dist = Levenshtein.distance(target.lower(), candidate.lower())
        if max_distance is not None and dist > max_distance:
            continue
        if closest is None or dist < closest_dist:
            closest = candidate
            closest_dist = dist

    return closest, closest_dist

def compute_centroid(coords):
    x_sum = sum(p[0] for p in coords)
    y_sum = sum(p[1] for p in coords)
    n = len(coords)
    return (x_sum / n, y_sum / n)

def closest_to_church(locations, church_name):
    closest_church_lev = closest_string(church_name, all_buildings)

    for building in geo_data_buildings_filtered:
        if building["properties"].get("aulic_name") == closest_church_lev[0]:
            closest_church_geometry = building["geometry"]
    
    if closest_church_geometry["type"] == "Polygon":
        closest_church_coordinates = compute_centroid(closest_church_geometry["coordinates"][0])
    elif closest_church_geometry["type"] == "MultiPolygon":
        closest_church_coordinates = compute_centroid(closest_church_geometry["coordinates"][0][0])

    current_loc= (None,None)
    for i, location in enumerate(locations):
        if location["type"] == "Polygon":
            loc_coordinates = compute_centroid(location["coordinates"][0])
        elif location["type"] == "MultiPolygon":
            loc_coordinates = compute_centroid(location["coordinates"][0][0])
        distance_to_church = sqrt((closest_church_coordinates[0] - loc_coordinates[0])**2 + (closest_church_coordinates[1] - loc_coordinates[1])**2)

        if i == 0:
            current_loc = (location, distance_to_church)
        else:
            if distance_to_church < current_loc[1]:
                current_loc = (location, distance_to_church)

    return current_loc[0]


with open("out\places_type_segmentation.json", "r", encoding='utf-8') as fp:
    place_data = json.load(fp)

with open("tassini\\2024_Streets_EPSG32633.geojson", "r", encoding='utf-8') as fp:
    geo_data = json.load(fp)

with open("tassini\\2024_Edifici_EPSG32633.geojson", "r", encoding='utf-8') as fp:
    geo_data_buildings = json.load(fp)


geo_data_filtered = []
all_names = list()

for entry in geo_data["features"]:
    if entry["properties"]["ISOLA"] == None or entry["properties"]["ISOLA"] == "Giudecca":
        geo_data_filtered.append(entry)
        all_names.append(entry["properties"]["NAME"])

        
geo_data_buildings_filtered = []
all_buildings = list()

for entry in geo_data_buildings["features"]:
    if entry["properties"]["ISOLA"] == None or entry["properties"]["ISOLA"] == "Giudecca":
        if entry["properties"]["aulic_name"]:
            geo_data_buildings_filtered.append(entry)
            all_buildings.append(entry["properties"]["aulic_name"])

a = 0
i = 0
j = 0
k = 0
for key, item in place_data.items():
    print("------------------------------------")
 
    closest_place = closest_string(key, all_names, 2)
    if closest_place[0]:
        j+=1
        #add location data:
        num_places = []
        for feature in geo_data_filtered:
            if feature["properties"].get("NAME") == closest_place[0]:
                num_places.append(feature["geometry"])
        place_data[key]["location"] = num_places[0]

        #check for homonyms:
        if len(num_places) > 1:
            place_data[key]["location"] = closest_to_church(num_places, place_data[key]["parish?"])
            if num_places[0] == place_data[key]["location"]:
                i+=1
            else: 
                print(key)
                a+=1
    else:
        k += 1

print(f"num_homonyms_not_changed(were true) : {i}\nnum_homonyms_changes(were not true) : {a}\nlevenshtein: {j}\nnon-matches: {k}")


with open("out/test_geo_ref.json", "w", encoding='utf-8') as fp:
    json.dump(place_data , fp, ensure_ascii=False) 