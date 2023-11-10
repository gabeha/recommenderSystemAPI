from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBasedRecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni.recommender_system import RecSys

# TODO: Should be created before the function call (below)
course_based = CourseBasedRecSys(model_name="all-MiniLM-L12-v2",
                                 seed_help=True,
                                 domain_adapt=True,
                                 zero_adapt=True,
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
    model_name="all-MiniLM-L12-v2",
    chat_gpt=False,
    thread_mode=False,
)
# explanation = None
bloom_based = BloomBasedRecSys()
# bloom_based = None
rs = RecSys(course_based=course_based,
            bloom_based=bloom_based,
            explanation=explanation,
            top_n=7, )
# Print a setting of the rec_sys object
rs.print_config()
