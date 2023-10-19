
def compute_recommendation(recSys):
    recSys.course_based.recommend(recSys) # Change scores of each course based on the keywords

    if recSys.bloom_based is not None:
        recSys.bloom_based.recommend(recSys)



def compute_constraints(recSys):
    pass



def compute_warnings(recSys):
    pass
