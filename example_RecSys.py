from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys

rs = RecSys()

# Print settings of RecSys
rs._settings_()

rs.course_based_settings = CourseBasedRecSys(
    top_n=20,
    min_df=1,
    use_maxsum=False,
    use_mmr=False,
    diversity=0.5,
    nr_candidates=20,
    force_keywords=True,
    precomputed_course=False
)

# Get list of recommended courses
student_input = { "keywords": { 'artificial': 1.0, 'math': 1.0,  'statistics': 1.0, 'data analyze': 1.0, 'law': 1.0, 'human rights': 1.0},
                  "blooms": { "create": 0.5, "understand": 0.75, "apply": 0.25, "analyze": 0.5, "evaluate": 0.0, "remember": 1.0 } }

# Check source code of recSys.recommender_system
results = rs.get_recommendation(student_input)


# Example of the output
print(str(results) + "\n")

for i in results['sorted_recommended_courses'][:10]:
    print(i + ": " + str(rs.course_data[i]['course_name']) + " || Score: "+ str(results['recommended_courses'][i]['score']))

#%%

#%%
