from rec_sys_uni.recommender_system import RecSys
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys

def recommend_courses(student_input):
    course_based = CourseBasedRecSys(model_name='all-MiniLM-L12-v2', # for ui
                                    seed_help=True, # for ui
                                    domain_adapt=True, # for ui
                                    zero_adapt=True, # for ui
                                    domain_type='title',
                                    seed_type='title',
                                    zero_type='title',
                                    precomputed_course=True)



    rs = RecSys(course_based=course_based)
    
    output = {}

    # Check source code of recSys.recommender_system
    try:
        results = rs.get_recommendation(student_input)
        output = {'structured_recommendation': results.get('structured_recommendation'),
        'explanation': results.get('explanation'), "student_input": student_input}
        print (output)
    except Exception as e:
        output = {'error': str(e)}
    finally:
        return output



    # Example of the output
    # print(str(results) + "\n")

    # Example of top 10 recommended courses
    # for i in results['sorted_recommended_courses'][:10]:
    #     print(i + ": " + str(rs.course_data[i]['course_name']) + " || Score: "+ str(results['recommended_courses'][i]['score']))

    # Example of structured recommended courses
    # for period in results['structured_recommendation']:
    #     print(period)
    #     for i in results['structured_recommendation'][period]:
    #         print(i + ": " + str(rs.course_data[i]['course_name']) + " || Score: "+ str(results['recommended_courses'][i]['score']))
    #     print()

# recommend_courses()