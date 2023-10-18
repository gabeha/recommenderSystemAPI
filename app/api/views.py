from flask import request, jsonify
from app.models.model import recommend_courses_routeHandler
from helpers.cast_int_to_float import recursive_cast_to_float

from . import api_blueprint

@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    student_input = request.get_json()
    student_input = recursive_cast_to_float(student_input)
    print(student_input)
    recommended = recommend_courses_routeHandler(student_input)
    return jsonify({'recommended_courses': recommended})
