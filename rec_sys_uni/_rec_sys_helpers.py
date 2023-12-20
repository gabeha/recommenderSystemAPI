def make_results_template(course_data):
    results = {
        "recommended_courses": {}
    }
    for course in course_data:
        results["recommended_courses"][course] = {"score": 0,
                                                  "period": course_data[course]["period"],
                                                  }
    return results


def semester_course_cleaning(course_data, semester):  # TODO: DEPRECATED
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


def sort_by_periods(recSys,
                    recommended_courses,
                    course_data,
                    max,
                    include_keywords=False,
                    include_blooms=False,
                    include_score=False,
                    include_warnings=False,
                    include_content=False,
                    percentage=False):
    # Sort recommended courses by score
    sorted_recommendation_list = sorted(recommended_courses.items(),
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
        period = course_data[i[0]]['period']

        course_tmp = {'course_code': i[0],
                      'course_name': course_data[i[0]]['course_name'],
                      }

        flag = False
        # Add courses to periods with certain max number of courses in period
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

        # Add other information to the course, for testing purposes
        if include_warnings and recSys.warning_model:
            course_tmp['warning'] = recommended_courses[i[0]]['warning'],
            course_tmp['warning_recommendation'] = recommended_courses[i[0]]['warning_recommendation']

        course_tmp = course_tmp.copy()

        if include_keywords and recSys.keyword_based:
            if percentage:
                course_tmp['keywords'] = {k: round(v * 100, 2) for k, v in
                                          recommended_courses[i[0]]['keywords'].items()}
            else:
                course_tmp['keywords'] = recommended_courses[i[0]]['keywords']
            course_tmp['keywords_score'] = recommended_courses[i[0]]['keywords_score']
        if include_blooms and recSys.bloom_based:
            if percentage:
                course_tmp['blooms'] = {k: round(v * 100, 2) for k, v in
                                        recommended_courses[i[0]]['blooms'].items()}
            else:
                course_tmp['blooms'] = recommended_courses[i[0]]['blooms']
            course_tmp['blooms_score'] = recommended_courses[i[0]]['blooms_score']
        if include_content and recSys.content_based:
            if percentage:
                course_tmp['content'] = {k: round(v * 100, 2) for k, v in
                                         recommended_courses[i[0]]['content'].items()}
            else:
                course_tmp['content'] = recommended_courses[i[0]]['content']
            course_tmp['content_score'] = recommended_courses[i[0]]['content_score']
        if include_score:
            course_tmp['score'] = recommended_courses[i[0]]['score']

        final_recommendation_list.append(course_tmp)

        if (len(structured_recommendation['semester_1']['period_1']) >= max
                and len(structured_recommendation['semester_1']['period_2']) >= max
                and len(structured_recommendation['semester_2']['period_4']) >= max
                and len(structured_recommendation['semester_2']['period_5']) >= max):
            break

    return final_recommendation_list, structured_recommendation


# Print for test
def print_recommendation(recSys,
                         recommended_courses,
                         course_data,
                         max,
                         include_keywords=False,
                         include_blooms=False,
                         include_score=False,
                         include_warning=False,
                         include_content=False,
                         percentage=False):
    sorted_recommendation_list, structured_recommendation = sort_by_periods(recSys,
                                                                            recommended_courses,
                                                                            course_data,
                                                                            max,
                                                                            include_keywords,
                                                                            include_blooms,
                                                                            include_score,
                                                                            include_warning,
                                                                            include_content,
                                                                            percentage)
    print("\nTop recommended courses:")
    for i in sorted_recommendation_list:
        print_text(i, include_keywords, include_blooms, include_score, include_warning, include_content)

    print(f"\n\nSemester 1:")
    print(f"\nPeriod 1:")
    for i in structured_recommendation['semester_1']['period_1']:
        print_text(i)
    print(f"\nPeriod 2:")
    for i in structured_recommendation['semester_1']['period_2']:
        print_text(i)
    print(f"\nSemester 2:")
    print(f"\nPeriod 4:")
    for i in structured_recommendation['semester_2']['period_4']:
        print_text(i)
    print(f"\nPeriod 5:")
    for i in structured_recommendation['semester_2']['period_5']:
        print_text(i)


def print_text(i,
               include_keywords=False,
               include_blooms=False,
               include_score=False,
               include_warning=False,
               include_content=False):
    text = f"{i['course_code']} - {i['course_name']}"
    if include_keywords:
        text += f"\n - Keywords {i['keywords']}"
        text += f" - {i['keywords_score']}"
    if include_blooms:
        text += f"\n - Blooms {i['blooms']}"
        text += f" - {i['blooms_score']}"
    if include_content:
        text += f"\n - Content {i['content']}"
        text += f" - {i['content_score']}"
    if include_score:
        text += f"\n - Final Score {i['score']}"
    if include_warning:
        text += f"\n - Warning {i['warning']}"
        for j in i['warning_recommendation']:
            text += f"\n - {j}"
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
