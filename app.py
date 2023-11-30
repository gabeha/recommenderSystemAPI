from flask import Flask, jsonify, request
from instances.recommender_instances import rs
from instances.socketio_instance import socketio
from flask_cors import CORS
from threading import Thread
import time

from helpers.cast_int_to_float import recursive_cast_to_float


app = Flask(__name__)
CORS(app)
socketio.init_app(app)

def dummy_llm_function():
    for i in range(10):
        time.sleep(1)
        socketio.emit('explanation', {'explanation': f"This is a dummy explanation for course {i}."})
    socketio.emit('disconnect', {'disconnected': True})

def generate_explanations_and_emit(rs, student_info):
    # Get Explanation of top_n courses based on StudentNode object
    if rs.explanation:
        rs.generate_explanation(student_info)
    socketio.emit('disconnect', {'disconnected': True})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    student_input = request.get_json()
    student_input = recursive_cast_to_float(student_input)
    output = {}
    # Check source code of recSys.recommender_system
    try:
        # Get the StudentNode object
        student_info = rs.get_recommendation(student_input, system_student_data=student_input['config']['imitate_student'] == 1.0)

        results = student_info.results

        output = {'structured_recommendation': results.get('structured_recommendation'),
                  'student_mongo_id': str(student_info.id), "student_input": student_input}
        
        print("emit recommendations")
        socketio.emit('recommendations', {'recommended_courses': output})

    except Exception as e:
        output = {'error': str(e)}
        socketio.emit('error', {'error': str(e)})
    finally:
        # Thread(target=generate_explanations_and_emit, args=(rs, student_info,)).start()
        
        Thread(target=dummy_llm_function, args=()).start()
        return jsonify({'recommended_courses': output})

@app.route('/api/explanation', methods=['POST'])
def get_explanation():
    input = request.get_json()
    student_mongo_id = input['student_mongo_id']
    course_code = input['course_code']

    print(f"student_mongo_id: {student_mongo_id}")

    explanation = rs.generate_explanation(student_mongo_id, course_code)

    return jsonify({'explanation': explanation})


if __name__ == '__main__':
    socketio.run(app, debug=True)

# input = {
#     "config": {"model_name": "all-MiniLM-L12-v2", "seed_help": True, "domain_adapt": True, "zero_adapt": True},
#     "keywords":{'math': 0.5, 'artificial intelligence': 0.5, 'data analyze': 0.5, 'statistics': 0.5,},
#     "blooms":{'create': 0.0,
#               'understand': 0.0,
#               'apply': 0.0,
#               'analyze': 0.0,
#               'evaluate': 0.0,
#               'remember': 0.0}
# }
# _, student_info, _ = recommend_courses(input)
# print_recommendation(student_info.results, include_keywords=True, include_blooms=False, include_score=True)