from flask import request, jsonify
from app.models.model import recommend_courses

from . import api_blueprint

@api_blueprint.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    keywords = data.get('keywords')
    bloom_action_verbs = data.get('bloom')
    recommended = recommend_courses(keywords=keywords, bloom_action_verbs=bloom_action_verbs)
    return jsonify({'recommended_courses': recommended})
