import json

"""
TODO: Data should be preprossed before using the following functions
function: get_course_data
return: course_data : dictionary 
"""


def get_course_data():
    raw_json = open('rec_sys_uni/datasets/data/course/course_data.json')
    raw_course_data = json.load(raw_json)
    final_course_data = {}
    for i in raw_course_data:
        final_course_data[i['code']] = {
            'course_name': i['title'],
            'period': i['period'],
            'level': i['level'],
            'prerequisites': [],
            'description': i['desc'],
            'ilos': i['ilo']
        }
    return final_course_data


"""
TODO: Data should be preprossed before using the following functions
function: get_student_data
return: student_data : dictionary
"""
# def get_student_data():
#     return
