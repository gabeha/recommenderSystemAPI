from flask import request, jsonify
from app.models.model import recommend_courses_routeHandler

from . import api_blueprint

@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    student_input = request.get_json(force=True)
    recommended = recommend_courses_routeHandler(student_input)
    return jsonify({'recommended_courses': recommended})
