
def compute_recommendation(recSys, student_info):
    recSys.course_based.recommend(student_info) # Change scores of each course based on the keywords

    if recSys.bloom_based:
        recSys.bloom_based.recommend(student_info)



def compute_constraints(recSys, student_info):
    pass



def compute_warnings(recSys, student_info):
    pass
