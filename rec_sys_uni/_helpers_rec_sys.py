def make_results_template(results, course_data):
    for course in course_data:
        results["recommended_courses"][course] = {"score": 0,
                                                  "period": course_data[course]["period"],
                                                  "warning": False}
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

#%%
