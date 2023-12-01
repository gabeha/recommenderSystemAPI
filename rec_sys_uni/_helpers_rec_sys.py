def make_results_template(results, course_data):
    for course in course_data:
        results["recommended_courses"][course] = {"score": 0,
                                                  "period": course_data[course]["period"],
                                                  "warning": False,
                                                  "warning_recommendation": [],
                                                  "keywords": {},
                                                  "blooms": {}}
    return results


def semester_course_cleaning(course_data, semester): #TODO: DEPRECATED
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


def sort_by_periods(recSys, student_info, max, include_keywords=False,
                    include_blooms=False, include_score=False, percentage=True):
    # Sort recommended courses by score
    sorted_recommendation_list = sorted(student_info.results['recommended_courses'].items(),
                                        key=lambda x: x[1]['score'], reverse=True)

    # Append code of sorted courses to the list
    final_recommendation_list = []

    # structured recommendation dictionary
    structured_recommendation = {
                                'semester_1':
                                    {
                                        'period_1': [],
                                        'period_2': []
                                    },
                                'semester_2':
                                    {
                                        'period_4': [],
                                        'period_5': []
                                    }
                                }

    for i in sorted_recommendation_list:
        period = student_info.course_data[i[0]]['period']

        course_tmp = {'course_code': i[0],
                      'course_name': student_info.course_data[i[0]]['course_name'],
                      'warning': student_info.results['recommended_courses'][i[0]]['warning'],
                      'warning_recommendation': student_info.results['recommended_courses'][i[0]]['warning_recommendation']}

        flag = False
        for j in period:
            if j == 1 and len(structured_recommendation['semester_1']['period_1']) < max:
                structured_recommendation['semester_1']['period_1'].append(course_tmp)
                flag = True
            if j == 2 and len(structured_recommendation['semester_1']['period_2']) < max:
                structured_recommendation['semester_1']['period_2'].append(course_tmp)
                flag = True
            if j == 4 and len(structured_recommendation['semester_2']['period_4']) < max:
                structured_recommendation['semester_2']['period_4'].append(course_tmp)
                flag = True
            if j == 5 and len(structured_recommendation['semester_2']['period_5']) < max:
                structured_recommendation['semester_2']['period_5'].append(course_tmp)
                flag = True

        if not flag:
            continue

        if include_keywords and recSys.course_based:
            if percentage:
                course_tmp['keywords'] = {k: round(v * 100, 2) for k, v in student_info.results['recommended_courses'][i[0]]['keywords'].items()}
            else:
                course_tmp['keywords'] = student_info.results['recommended_courses'][i[0]]['keywords']
        if include_blooms and recSys.bloom_based:
            if percentage:
                course_tmp['blooms'] = {k: round(v * 100, 2) for k, v in student_info.results['recommended_courses'][i[0]]['blooms'].items()}
            else:
                course_tmp['blooms'] = student_info.results['recommended_courses'][i[0]]['blooms']
        if include_score:
            course_tmp['score'] = student_info.results['recommended_courses'][i[0]]['score']

        final_recommendation_list.append(course_tmp)

        if (len(structured_recommendation['semester_1']['period_1']) >= max
                and len(structured_recommendation['semester_1']['period_2']) >= max
                and len(structured_recommendation['semester_2']['period_4']) >= max
                and len(structured_recommendation['semester_2']['period_5']) >= max):
            break

    student_info.results['structured_recommendation'] = structured_recommendation
    student_info.results['sorted_recommended_courses'] = final_recommendation_list

def print_recommendation(results, include_keywords=False, include_blooms=False, include_score=False):
    print("\nTop recommended courses:")
    for i in results['sorted_recommended_courses']:
        print_text(i, include_keywords, include_blooms, include_score)

    print(f"\n\nSemester 1:")
    print(f"\nPeriod 1:")
    for i in results['structured_recommendation']['semester_1']['period_1']:
        print_text(i, include_keywords, include_blooms, include_score)
    print(f"\nPeriod 2:")
    for i in results['structured_recommendation']['semester_1']['period_2']:
        print_text(i, include_keywords, include_blooms, include_score)
    print(f"\nSemester 2:")
    print(f"\nPeriod 4:")
    for i in results['structured_recommendation']['semester_2']['period_4']:
        print_text(i, include_keywords, include_blooms, include_score)
    print(f"\nPeriod 5:")
    for i in results['structured_recommendation']['semester_2']['period_5']:
        print_text(i, include_keywords, include_blooms, include_score)

def print_text(i, include_keywords, include_blooms, include_score):
    text = f"{i['course_code']} - {i['course_name']}"
    if include_keywords:
        text += f" - {i['keywords']}"
    if include_blooms:
        text += f" - {i['blooms']}"
    if include_score:
        text += f" - {i['score']}"
    text += f" - {i['warning']}"
    for j in i['warning_recommendation']:
        text += f" - {j}"
    print(text)

class StudentNode:
    def __init__(self, results, student_intput, course_data, student_data):
        self.results = results
        self.student_input = student_intput
        self.course_data = course_data
        self.student_data = student_data

    def set_id(self, id):
        self.id = id

# %%
