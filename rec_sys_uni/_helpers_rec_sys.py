def make_results_template(results, course_data):
    for course in course_data:
        results["recommended_courses"][course] = { "score": 0,
                                                   "period": course_data[course]["period"],
                                                   "warning": False }
    return results
