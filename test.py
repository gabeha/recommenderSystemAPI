from instances.recommender_instances import rs
from rec_sys_uni._rec_sys_helpers import print_recommendation


def recommend(student_input):
    student_info = rs.get_recommendation(student_input)

    return student_info


input = {
    "keywords": {
        "physics": 1.0,
        "maths": 1.0,
        "statistics": 1.0,
        "ai": 1.0,
        "computer science": 1.0,
        "chem": 1.0,
    },
    "blooms": {'create': 0.0,
               'understand': 0.0,
               'apply': 0.0,
               'analyze': 0.0,
               'evaluate': 0.0,
               'remember': 0.0}
}

student_info = recommend(input)
print_recommendation(rs,
                     student_info.results["recommended_courses"],
                     student_info.course_data,
                     max=7,
                     include_keywords=True,
                     include_blooms=False,
                     include_score=True,
                     include_warning=True,
                     percentage=True)
