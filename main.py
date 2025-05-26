"""
In: python list of skills (target set)
In: python list of course descriptions (query set)


Out: txt listing all metadata
Out: JSON list with each course and coresponding skills with cosine similarity (raw data)
Out: JSON list with each course and n coresponding skills with higherst consine similartiy (clean data)
Each output file is backed up every 40 courses, looks like "matches_raw[40].json"
"""


from data_loader import csv_column_to_list, json_to_list 
from text_preprocessing import remove_boilerplate

import stanza
from sentence_transformers import SentenceTransformer, util
import os
import torch
import json
from tqdm import tqdm
from pathlib import Path


# Variables
BASE_PATH = Path(__file__).parent # can be changed if required
BEST_MATCHES_N_SENTENCE = 3 # this will indicate the number matches which will be retrieved for every sentence, which is usually about 4 to 6
BEST_MATHCES_THRESHOLD = 0.3
RUN_FOLDER_NAME = "testing_run"
COURSES_PATH = ("university_data\\sample_data.json")
SKILLS_PATH = ("ONET_data\\ONET_DWA.csv", "DWA Title") # (path, column name)


RUN_FOLDER_PATH = BASE_PATH / RUN_FOLDER_NAME

try:
    os.mkdir(RUN_FOLDER_PATH)
    print(f"Folder '{RUN_FOLDER_NAME}' created successfully.")
except FileExistsError:
    print(f"Folder '{RUN_FOLDER_NAME}' already exists.")
except Exception as e:
    print(f"An error occurred: {e}")
print(f"The working folder will be {RUN_FOLDER_PATH}")

# Initialization
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')
model = SentenceTransformer('all-mpnet-base-v2')

def segment_text(text):
    doc = nlp(text)
    sentences = [sentence.text for sentence in doc.sentences]
    return sentences

def save_backup():
    pass


# Data
onet_skills = csv_column_to_list(SKILLS_PATH[0], SKILLS_PATH[1])
skill_embeddings = model.encode(onet_skills, convert_to_tensor=True)

courses = json_to_list(COURSES_PATH)
courses = [[course["name"], course["code"], course["university"], course["description"]] for course in courses]
print(courses)


# Processing
retrieved_data_courses = []
retrieved_data_courses_sorted = []
for i, course in enumerate(courses):
    print(f"PROCESSING ({i+1}/{len(courses)})")

    course_name, course_code, course_university, course_description = course[0], course[1], course[2], course[3]

    clean_course_description = remove_boilerplate(course_description)
    
    sentences = segment_text(clean_course_description)
    print("Segmented Sentences:", sentences)

    

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)


    similarity_matrix = util.cos_sim(sentence_embeddings, skill_embeddings)
    all_clean_matches = []
    all_all_matches = []

    for j, sentence in tqdm(enumerate(sentences), total=len(sentences), desc='Processing Sentences'):
        similarities = similarity_matrix[j].tolist()
        sorted_indices = sorted(range(len(similarities)), key=lambda x: similarities[x], reverse=True)

        # TOP
        top_matches = [{'skill': onet_skills[idx], 'score': round(similarities[idx], 4)} for idx in sorted_indices[:BEST_MATCHES_N_SENTENCE]]
        top_matches_sorted = sorted(top_matches, key=lambda x: x['score'])
        
        for skill in top_matches_sorted:
            if skill["score"] > BEST_MATHCES_THRESHOLD:
                all_clean_matches.append(skill["skill"])


        # ALL
        all_mathces = [{'skill': onet_skills[idx], 'score': round(similarities[idx], 4)} for idx in sorted_indices[:30]]
        all_mathces_sorted = sorted(all_mathces, key=lambda x: x['score'])
    
        all_all_matches.append({"sentence": sentence, "matches": all_mathces})
    
    retrieved_data_course = {
        "name": course_name,
        "code": course_code,
        "university": course_university,
        "description": course_description,
        "clear_description": clean_course_description,
        "sentences": sentences,
        "matches": all_all_matches
    }

    retrieved_data_courses.append(retrieved_data_course)

    # insure all_clean_matches has no duplicates
    all_clean_matches = list(set(all_clean_matches))

    retrieved_data_course_sorted = {
        "name": course_name,
        "code": course_code,
        "university": course_university,
        "description": course_description,
        "matches": all_clean_matches
    }

    retrieved_data_courses_sorted.append(retrieved_data_course_sorted)

    if (i+1) % 20 == 0:
        with open(RUN_FOLDER_PATH / f"matches_raw[{(i+1)}].json", "w", encoding="utf-8") as file:
            json.dump(retrieved_data_courses, file, indent=4)

        with open(RUN_FOLDER_PATH / f"matches_sorted[{(i+1)}].json", "w", encoding="utf-8") as file:
            json.dump(retrieved_data_courses_sorted, file, indent=4)
    
with open(RUN_FOLDER_PATH / f"matches_raw[{len(courses)}].json", "w", encoding="utf-8") as file:
    json.dump(retrieved_data_courses, file, indent=4)

with open(RUN_FOLDER_PATH / f"matches_sorted[{len(courses)}].json", "w", encoding="utf-8") as file:
        json.dump(retrieved_data_courses_sorted, file, indent=4)
    

