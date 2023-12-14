from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBased
from rec_sys_uni.rec_systems.keyword_based_sys.keyword_based import KeywordBased
from rec_sys_uni.rec_systems.warning_model.warning_model import WarningModel
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.rec_systems.planners.ucm_planner import UCMPlanner

"""
Asymmetric search:
    1. sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco  (keyBert, Dot Product)
    2. msmarco-distilbert-base-v4 (keyBert, Cosine Similarity)
    3. intfloat/e5-base-v2 (keyBert, Cosine Similarity)
Symmetric search:
    1. all-MiniLM-L12-v2 (keyBert, Cosine Similarity)
    2. Intel/bge-base-en-v1.5-int8-static (Intel, Cosine Similarity)
"""
# Keyword Based
keyword_based = KeywordBased(model_name="all-MiniLM-L12-v2",
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
# Print a setting of the Keyword Based system
keyword_based.print_config()

# Explanation model
explanation = LLM(
    token="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"
)
# explanation = None

# Bloom Based
bloom_based = BloomBased(score_alg='sum')
# bloom_based = None

# Warning model
warning_model = WarningModel()
# warning_model = None

# Planner model
# planner = UCMPlanner('rec_sys_uni/datasets/data/planners/catalog.json')
planner = None

rs = RecSys(keyword_based=keyword_based,
            bloom_based=bloom_based,
            explanation=explanation,
            warning_model=warning_model,
            top_n=7,
            planner=planner)
# Print a setting of the rec_sys object
rs.print_config()
