from helpers.cast_int_to_float import recursive_cast_to_float
from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys
from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBasedRecSys
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni._helpers_rec_sys import print_recommendation


def recommend_courses(student_input):
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
                                     score_alg='sum',
                                     scaler=True,
                                     sent_splitter=False,
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
                explanation=explanation,
                top_n=7,)
    # Print a setting of the rec_sys object
    rs.print_config()
    # TODO: Should be created before the function call (above)

    output = {}

    # Check source code of recSys.recommender_system
    try:
        # Get the StudentNode object
        student_info = rs.get_recommendation(student_input)

        # maybe move this to somewhere else to make sure that the recommendations are sent to frontend asap
        # Get Explanation of top_n courses based on StudentNode object
        if rs.explanation:
            rs.generate_explanation(student_info)

        results = student_info.results

        output = {'structured_recommendation': results.get('structured_recommendation'), # results['structured_recommendation']['semester_1']['period_1']
                  'explanation': results.get('explanation'), "student_input": student_input}
        # print (output)
    except Exception as e:
        output = {'error': str(e)}
        print(e)
    finally:
        return output, student_info, rs


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

#%%
