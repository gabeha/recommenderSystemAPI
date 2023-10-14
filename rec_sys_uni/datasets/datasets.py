import json
from os import path

"""
TODO: Data should be preprossed before using the following functions
function: get_course_data
return: course_data : dictionary 
"""


def get_course_data():
    file_name = 'rec_sys_uni/datasets/data/course/course_data.json'

    # Check if file exists
    if path.isfile(file_name) is False:
        raise Exception("File not found")

    # Read JSON file
    with open(file_name, encoding="utf8") as fp:
        raw_course_data = json.load(fp)

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

def get_domains_data():
    file_name = 'rec_sys_uni/datasets/data/course/domains_course_data.json'
    # Check if file exists
    if path.isfile(file_name) is False:
        raise Exception("File not found")
    # Read JSON file
    with open(file_name, encoding="utf8") as fp:
        adaptation_data = json.load(fp)

    return adaptation_data




"""
TODO: Data should be preprossed before using the following functions
function: get_student_data
return: student_data : dictionary
"""
# def get_student_data():
#     return
