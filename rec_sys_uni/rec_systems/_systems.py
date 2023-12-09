
def compute_recommendation(recSys, student_info):
    recSys.keyword_based.recommend(student_info) # Change scores of each course based on the keywords

    if recSys.bloom_based:
        recSys.bloom_based.recommend(student_info)

def compute_warnings(recSys, student_info):
    recSys.warning_model.predict(student_info) # Predict the pass or fail for courses and give warning recommendations
