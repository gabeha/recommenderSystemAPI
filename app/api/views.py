import random
from threading import Thread
import time
from flask import jsonify, request

from app.models.model import recommend_courses_routeHandler

from . import api_blueprint

def dummy_llm_function(course):
    # Simulate some processing delay
    time.sleep(random.randint(1, 3))
    # Return a dummy explanation
    return f"This is a dummy explanation for {course['title']}."

def generate_explanations_and_emit(courses, socketio):
    for course in courses:
        explanation = dummy_llm_function(course)
        socketio.emit('explanation', {'course_id': course['id'], 'explanation': explanation})


@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    student_input = request.get_json()
    recommended = recommend_courses_routeHandler(student_input)
    Thread(target=generate_explanations_and_emit, args=(courses, socketio)).start()
    return jsonify({'recommended_courses': recommended})
