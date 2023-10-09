from rec_sys_uni.errors_checker.errors import check_student_input, check_student_data, check_course_data
from rec_sys_uni.datasets.datasets import get_course_data
from rec_sys_uni._helpers_rec_sys import make_results_template
from rec_sys_uni.rec_systems._systems import *


class RecSys():

    def __init__(self):
        self.precomputed_course = False
        self.precomputed_bloom = False
        self.contraints = False
        self.top_n = 20
        self.validate_input = True
        self.system_course_data = True

    """
    function: validate_system_input
    description: validate the format of the input data
    """

    def validate_system_input(self,
                              student_input,
                              course_data=None,
                              student_data=None):

        check_student_input(student_input)

        if self.system_course_data:
            course_data = get_course_data()

        check_course_data(course_data)

        if student_data is not None:
            check_student_data(student_data)

        return student_input, course_data, student_data

    """
    function: get_list_recommended_courses
    description: get list of recommended courses based on student input
    parameters: student_input : dictionary {
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
                                                    
                student_data : dictionary  {
                                            courses_taken: 
                                                {
                                                    course_id(String): 
                                                                        {
                                                                            passed: boolean,
                                                                            grade: float,
                                                                            period: int or [from, to],
                                                                            year: int
                                                                        }, 
                                                    ...
                                                }
                                            }
                                            
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
                validate_input: boolean || validate the format of the input data
                system_course_data: boolean || use the course data from the system
    return: dictionary of recommended courses 
            e.g. { 
                    recommended_courses: dictionary {
                                                        course_id(String):
                                                                    {   
                                                                        score: float,
                                                                        period: [int, ...] or [[int, int], ...],    e.g [1, 4] or [[1, 2, 3], [4, 5, 6]] or [[1,2]], TODO: In the later stage should be an recommended period
                                                                        warning: boolean
                                                                    },
                                                        ...
                                                    },
                    sorted_recommended_courses: top_n list of course_id(String) sorted by score,
                    structured_recommendation: TODO (Later stage)                        
                    explanation: String
                }
                    
    """

    def get_recommendation(self,
                           student_intput,
                           course_data=None,
                           student_data=None,
                           ):

        try:
            if self.validate_input:
                student_data, course_data, student_data = self.validate_system_input(student_intput,
                                                                                     course_data,
                                                                                     student_data)
        except Exception as e:
            raise e

        results = {"recommended_courses": {},
                   "sorted_recommended_courses": [],
                   "explanation": ""}

        results = make_results_template(results, course_data)

        results = compute_recommendation(results, student_intput, course_data, student_data, self.precomputed_course,
                                         self.precomputed_bloom, self.top_n)

        if self.contraints:
            results = compute_constraints(results)  # recommended_courses
            results = compute_warnings(results, student_data)  # structured_recommendation
        else:
            results = compute_warnings(results, student_data)  # sorted_recommended_courses

        self.results = results
        self.student_input = student_intput
        self.course_data = course_data
        self.student_data = student_data

        return results

    def _settings_(self):
        print(f"\nPrecomputed_course: {self.precomputed_course} \n" +
              f"Precomputed_bloom: {self.precomputed_bloom} \n" +
              f"Contraints: {self.contraints} \n" +
              f"Top_n: {self.top_n} \n" +
              f"Validate_input: {self.validate_input} \n" +
              f"System_course_data: {self.system_course_data} \n")
