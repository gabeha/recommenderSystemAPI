from flask import json
import os
from example_RecSys import recommend_courses

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'sampleOutput.json')

def recommend_courses_routeHandler(student_input):
    recommended_courses = recommend_courses(student_input=student_input)
    return recommended_courses
