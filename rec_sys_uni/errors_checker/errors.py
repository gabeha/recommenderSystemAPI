from rec_sys_uni.errors_checker.exceptions.rec_sys_errors import *

def check_student_input(student_input):
    """
    student_input : dictionary {
                                keywords: {keywords(String): weight(float), ...} ,
                                blooms: {blooms(String): weight(float), ...}
                                }
                                    e.g. student_input =    {
                                                                keywords: {
                                                                                'python': 0.5,
                                                                                'data science': 0.2
                                                                          },
                                                                blooms: {
                                                                            'create': 0.5,
                                                                            'understand': 0.75,
                                                                            'apply': 0.25',
                                                                            'analyze': 0.5,
                                                                            'evaluate': 0,
                                                                            'remember': 1
                                                                        }
                                                            }
    """
    if student_input == None:
        raise StudentInputFormatError("student_input is None")
    if not isinstance(student_input, dict):
        raise StudentInputFormatError("student_input is not a dictionary")
    if not 'keywords' in student_input:
        raise StudentInputFormatError("student_input does not have keywords")
    if not 'blooms' in student_input:
        raise StudentInputFormatError("student_input does not have blooms")
    if not StudentInputFormatError(student_input['keywords'], dict):
        raise StudentInputFormatError("student_input['keywords'] is not a dictionary")
    if not isinstance(student_input['blooms'], dict):
        raise StudentInputFormatError("student_input['blooms'] is not a dictionary")
    for key in student_input['keywords']:
        if not isinstance(key, str):
            raise StudentInputFormatError("student_input['keywords'] has a key that is not a string")
        if not isinstance(student_input['keywords'][key], float):
            raise StudentInputFormatError("student_input['keywords'][key] is not a float")
    for key in student_input['blooms']:
        if not isinstance(key, str):
            raise StudentInputFormatError("student_input['blooms'] has a key that is not a string")
        if not isinstance(student_input['blooms'][key], float):
            raise StudentInputFormatError("student_input['blooms'][key] is not a float")




def check_student_data(student_data):
    """
    student_data : dictionary  {
                                courses_taken:
                                    {
                                        course_id(String):
                                                            {
                                                                passed: boolean,
                                                                grade: float,
                                                                period: int || [from, to],
                                                                year: int
                                                            },
                                        ...
                                    }
                                }
    """
    if not isinstance(student_data, dict):
        raise StudentDataFormatError("student_data is not a dictionary")
    if not 'courses_taken' in student_data:
        raise StudentDataFormatError("student_data does not have courses_taken")
    for c in student_data['courses_taken']:
        key = student_data['courses_taken'][c]
        if 'passed' not in key:
            raise StudentDataFormatError(f"{key} does not have key passed")
        if 'grade' not in key:
            raise StudentDataFormatError(f"{key} does not have key grade")
        if 'period' not in key:
            raise StudentDataFormatError(f"{key} does not have key period")
        if 'year' not in key:
            raise StudentDataFormatError(f"{key} does not have key year")
        if not isinstance(key['passed'], bool):
            raise StudentDataFormatError(f"{key}[passed] is not a boolean")
        if not isinstance(key['grade'], float):
            raise StudentDataFormatError(f"{key}[float] is not a floar")
        if not isinstance(key['period'], int) and not isinstance(key['period'], list):
            raise StudentDataFormatError(f"{key}[period] is not a int or list")
        if isinstance(key['period'], list):
            if len(key['period']) != 2:
                raise StudentDataFormatError(f"{key}[period] is length {len(key['period'])}, should be length of 2")
        if not isinstance(key['year'], int):
            raise StudentDataFormatError(f"{key}[year] is not an integer")

def check_course_data(course_data):
    """
    course_data : dictionary    {
                                    course_id(String): dictionary
                                            {
                                                course_name: String,    e.g "Philosophy of Science"
                                                period: [int, ...] or [[int, int], ...],    e.g [1, 4] or [[1, 2, 3], [4, 5, 6]] or [[1,2]]
                                                level: int,        e.g 1, 2, 3
                                                prerequisites: [course_id(String), ...],    e.g ["COR1001", "COR1002"]
                                                description: String,    e.g "This course is about ..."
                                                ilos: [String, ...],    e.g ["Be able to apply the scientific method to a given problem",
                                                                             "Be able to explain the difference between science and pseudoscience"]
                                            },
                                    ...
                                }
    """
    if course_data == None:
        raise CourseDataFormatError("course_data is None")
    if not isinstance(course_data, dict):
        raise CourseDataFormatError("course_data is not a dictionary")
    for key in course_data:
        if not isinstance(key, str):
            raise CourseDataFormatError("course_data has a key that is not a string")
        if not isinstance(course_data[key], dict):
            raise CourseDataFormatError(f"{key} is not a dictionary")
        if not 'course_name' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have course_name")
        if not 'period' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have period")
        if not 'level' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have level")
        if not 'prerequisites' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have prerequisites")
        if not 'description' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have description")
        if not 'ilos' in course_data[key]:
            raise CourseDataFormatError(f"{key} does not have ilos")
        if not isinstance(course_data[key]['course_name'], str):
            raise CourseDataFormatError(f"{key} 'course_name' is not a string")
        if not isinstance(course_data[key]['period'], list):
            raise CourseDataFormatError(f"{key} 'period' is not a list")
        for period in course_data[key]['period']:
            if not isinstance(period, int) and not isinstance(period, list):
                raise CourseDataFormatError(f"{key}: 'period' has an element {period} that is not an integer or list. but {type(period)}")
        if not isinstance(course_data[key]['level'], int):
            raise CourseDataFormatError("course_data[key]['level'] is not an integer")
        if not isinstance(course_data[key]['prerequisites'], list):
            raise CourseDataFormatError(f"{key} 'prerequisites' is not a list")
        for prerequisite in course_data[key]['prerequisites']:
            if not isinstance(prerequisite, str):
                raise CourseDataFormatError(f"{key} 'prerequisites' has an element that is not a string")
        if not isinstance(course_data[key]['description'], str):
            raise CourseDataFormatError(f"{key} 'description' is not a string")
        if not isinstance(course_data[key]['ilos'], list):
            raise CourseDataFormatError(f"{key} 'ilos' is not a list")
        for ilo in course_data[key]['ilos']:
            if not isinstance(ilo, str):
                raise CourseDataFormatError(f"{key} 'ilos' has an element that is not a string")


