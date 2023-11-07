def make_results_template(results, course_data):
    for course in course_data:
        results["recommended_courses"][course] = {"score": 0,
                                                  "period": course_data[course]["period"],
                                                  "warning": False,
                                                  "keywords": {},
                                                  "blooms": {}}
    return results


def semester_course_cleaning(course_data, semester):
    """
    course_data: dictionary of course data
    semester: int
    return: new_course_data: dictionary of course data with only courses in the semester
    """
    range_semester = (1, 2, 3) if semester == 1.0 else (4, 5, 6)
    new_course_data = {}
    for course in course_data:
        period = course_data[course]["period"]
        for i in period:
            if isinstance(i, list):
                for j in i:
                    if j in range_semester:
                        new_course_data[course] = course_data[course]
                        break
            else:
                if i in range_semester:
                    new_course_data[course] = course_data[course]
    return new_course_data


def sort_by_periods(recSys, student_info, max=6, include_keywords=False, include_blooms=False):
    # Sort recommended courses by score
    sorted_recommendation_list = sorted(student_info.results['recommended_courses'].items(),
                                        key=lambda x: x[1]['score'], reverse=True)

    # Append code of sorted courses to the list
    final_recommendation_list = []

    # structured recommendation dictionary
    structured_recommendation = {'period_1': [],
                                 'period_2': []
                                 }

    for i in sorted_recommendation_list:
        period = student_info.course_data[i[0]]['period']
        for j in period:
            course_tmp = {'course_code': i[0], 'course_name': student_info.course_data[i[0]]['course_name']}
            if include_keywords and recSys.course_based:
                course_tmp['keywords'] = student_info.results['recommended_courses'][i[0]]['keywords']
            if include_blooms and recSys.bloom_based:
                course_tmp['blooms'] = student_info.results['recommended_courses'][i[0]]['blooms']

            if (j == 1 or j == 4) and len(structured_recommendation['period_1']) < max:
                structured_recommendation['period_1'].append(course_tmp)
                break
            if (j == 2 or j == 5) and len(structured_recommendation['period_2']) < max:
                structured_recommendation['period_2'].append(course_tmp)
                break
        final_recommendation_list.append(i[0])

    student_info.results['structured_recommendation'] = structured_recommendation
    student_info.results['sorted_recommended_courses'] = final_recommendation_list
    # for i in final_recommendation_list[:20]:
    #     print(student_info.course_data[i]['course_name'])


class StudentNode:
    def __init__(self, results, student_intput, course_data, student_data):
        self.results = results
        self.student_input = student_intput
        self.course_data = course_data
        self.student_data = student_data

# %%
