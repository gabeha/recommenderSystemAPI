from flask import Flask, jsonify, request
from socketio_instance import socketio
from flask_cors import CORS
from threading import Thread
import time
import random

from helpers.cast_int_to_float import recursive_cast_to_float
from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBasedRecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni._helpers_rec_sys import sort_by_periods

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

    print("get_recommendations")

    student_input = request.get_json()

    model_name = student_input["config"]["model_name"]
    seed_help = student_input["config"]["seed_help"]
    domain_adapt = student_input["config"]["domain_adapt"]
    zero_adapt = student_input["config"]["zero_adapt"]

    print(model_name, seed_help, domain_adapt, zero_adapt)
    student_input = recursive_cast_to_float(student_input)

    # TODO: Should be created before the function call (below)
    course_based = CourseBasedRecSys(model_name=model_name,
                                     seed_help=seed_help,
                                     domain_adapt=domain_adapt,
                                     zero_adapt=zero_adapt,
                                     domain_type='title',
                                     seed_type='title',
                                     zero_type='title',
                                     precomputed_course=True)
    # Print a setting of the course_based system
    course_based.print_config()
    explanation = LLM(
        url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        token="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",
        model_id='HuggingFaceH4/zephyr-7b-beta',
        model_name="all-MiniLM-L12-v2"
    )
    # explanation = None
    bloom_based = BloomBasedRecSys()
    # bloom_based = None
    rs = RecSys(course_based=course_based,
                bloom_based=bloom_based,
                explanation=explanation)
    # Print a setting of the rec_sys object
    rs.print_config()
    # TODO: Should be created before the function call (above)

    output = {}

    # Check source code of recSys.recommender_system
    try:
        # Get the StudentNode object
        student_info = rs.get_recommendation(student_input)


        # Sort recommended courses by score without keywords and blooms in the output
        sort_by_periods(rs, student_info, max=rs.top_n)

        results = student_info.results

        output = {'structured_recommendation': results.get('structured_recommendation'),
                  'explanation': results.get('explanation'), "student_input": student_input}
        
        print("emit recommendations")
        socketio.emit('recommendations', {'recommended_courses': output})

        # print (output)
    except Exception as e:
        output = {'error': str(e)}
        socketio.emit('error', {'error': str(e)})
    finally:
        Thread(target=generate_explanations_and_emit, args=(rs, student_info,)).start()
        
        # Thread(target=dummy_llm_function, args=()).start()
        return jsonify({'recommended_courses': output})


if __name__ == '__main__':
    socketio.run(app, debug=True)
