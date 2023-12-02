from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBasedRecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys
from rec_sys_uni.rec_systems.warning_model.warning_model import WarningModel
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.planners.ucm_planner import UCMPlanner

"""
Asymmetric search:
    1. sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco  (keyBert, Dot Product)
    2. msmarco-distilbert-base-v4 (keyBert, Cosine Similarity)
    3. intfloat/e5-base-v2 (keyBert, Cosine Similarity)
Symmetric search:
    1. all-MiniLM-L12-v2 (keyBert, Cosine Similarity)
    2. Intel/bge-base-en-v1.5-int8-static (Intel, Cosine Similarity)
"""

course_based = CourseBasedRecSys(model_name="all-MiniLM-L12-v2",
                                 seed_help=True,
                                 domain_adapt=True,
                                 zero_adapt=True,
                                 domain_type='title',
                                 seed_type='title',
                                 zero_type='title',
                                 score_alg='sum',
                                 distance='cos',
                                 backend='keyBert',
                                 scaler=True,
                                 sent_splitter=False,
                                 precomputed_course=True)
# Print a setting of the course_based system
course_based.print_config()
explanation = LLM(
    url="https://api-inference.huggingface.co/models/openchat/openchat_3.5",
    token="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",
    model_id='openchat/openchat_3.5',
    model_name="all-MiniLM-L12-v2",
    chat_gpt=True,
    thread_mode=False,
)
# explanation = None
bloom_based = BloomBasedRecSys()
# bloom_based = None
warning_model = WarningModel()
rs = RecSys(course_based=course_based,
            bloom_based=bloom_based,
            explanation=explanation,
            warning_model=warning_model,
            top_n=7, 
            planner=UCMPlanner('rec_sys_uni/planners/catalog.json'))
# Print a setting of the rec_sys object
rs.print_config()
