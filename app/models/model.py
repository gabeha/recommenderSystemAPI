from flask import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'sampleOutput.json')

def recommend_courses(keywords, bloom_action_verbs):
    # Your recommender model logic here
    # Use keywords and blooms_action_verbs to generate recommended courses
    with open(FILE_PATH, 'r') as f:
        recommended_courses = json.load(f)
    return recommended_courses
