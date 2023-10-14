from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys


course_based = CourseBasedRecSys(seed_help=True,
                                 domain_adapt=True,
                                 zero_adapt=True,
                                 domain_type='title',
                                 seed_type='title',
                                 zero_type='title',
                                 precomputed_course=True)



rs = RecSys(course_based=course_based)

# Print settings of RecSys
rs._settings_()


# Get list of recommended courses
student_input = { "keywords": { 'artificial': 1.0, 'math': 1.0,  'statistics': 1.0, 'data analyze': 1.0, 'law': 1.0, 'human rights': 1.0},
                  "blooms": { "create": 0.0, "understand": 0.0, "apply": 0.0, "analyze": 0.0, "evaluate": 0.0, "remember": 0.0 },
                  "semester": 1.0}

# Check source code of recSys.recommender_system
results = rs.get_recommendation(student_input)


# Example of the output
# print(str(results) + "\n")

# Example of top 10 recommended courses
# for i in results['sorted_recommended_courses'][:10]:
#     print(i + ": " + str(rs.course_data[i]['course_name']) + " || Score: "+ str(results['recommended_courses'][i]['score']))

# Example of structured recommended courses
for period in results['structured_recommendation']:
    print(period)
    for i in results['structured_recommendation'][period]:
        print(i + ": " + str(rs.course_data[i]['course_name']) + " || Score: "+ str(results['recommended_courses'][i]['score']))
    print()
#%%

#%%
