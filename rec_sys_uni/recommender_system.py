from rec_sys_uni.errors_checker.errors import check_student_input, check_student_data, check_course_data
from rec_sys_uni.datasets.datasets import get_course_data
from rec_sys_uni._helpers_rec_sys import make_results_template, semester_course_cleaning, StudentNode, sort_by_periods
from rec_sys_uni.rec_systems._systems import *
from rec_sys_uni.rec_systems.course_based_sys.course_based import CourseBasedRecSys
from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBasedRecSys
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM


class RecSys:

    def __init__(self,
                 course_based: CourseBasedRecSys = None,
                 bloom_based: BloomBasedRecSys = None,
                 explanation: LLM = None,
                 top_n=7):
        self.constraints = False
        self.validate_input = True
        self.system_course_data = True
        self.course_based = course_based
        self.bloom_based = bloom_based
        self.explanation = explanation
        self.top_n = top_n

    def validate_system_input(self,
                              student_input,
                              course_data=None,
                              student_data=None):
        """
        function: validate_system_input
        description: validate the format of the input data
        """

        check_student_input(student_input)

        if self.system_course_data:
            course_data = get_course_data()

        check_course_data(course_data)

        if student_data is not None:
            check_student_data(student_data)

        return student_input, course_data, student_data

    def get_recommendation(self,
                           student_intput,
                           course_data=None,
                           student_data=None,
                           ):
        """
        function: get_list_recommended_courses
        description: get list of recommended courses based on student input
        parameters: student_input : dictionary {
                                                keywords: {keywords(String): weight(float), ...} ,
                                                blooms: {blooms(String): weight(float), ...}
                                                semester: float
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
                                                                                        'evaluate': 0.0,
                                                                                        'remember': 1.0
                                                                                        }
                                                                                semester: 1.0 TODO: DEPRECATED (Do not use this key)
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
        return: StudentNode object, which contains: (check _helpers_rec_sys.py)
                results: dictionary of recommended courses
                e.g. {
                        recommended_courses: dictionary {
                                                            course_id(String):
                                                                        {
                                   Total score of each model                score: float,
                                   Course periods                           period: [int, ...] or [[int, int], ...],    e.g [1, 4] or [[1, 2, 3], [4, 5, 6]] or [[1,2]]
                                                                            warning: boolean
                                   (after applying CourseBased model)       keywords: {scores to each keyword}
                                   (after applying BloomBased model)        blooms: {scores to each bloom}
                                                                        },
                                                            ...
                                                        },
                        sorted_recommended_courses: list of courses {'course_code': String,
                                                                     'course_name': String
Note: if you want include keyword or blooms, check                   'keywords': {keywords(String): weight(float), ...} ,
rec_sys_uni._helpers_rec_sys.py -> sort_by_periods function          'blooms': {blooms(String): weight(float), ...}

                        structured_recommendation: dictionary {
                                                                semester_1: dictionary {
(if you apply sort_by_periods function,                             period_1: list of course_id(String), condition <= top_n
when you will have structured_recommendation                        period_2: list of course_id(String), condition <= top_n
and sorted_recommended_courses key in the results)              },
                                                                semester_2: dictionary {
                                                                    period_4: list of course_id(String), condition <= top_n
                                                                    period_5: list of course_id(String), condition <= top_n
                                                                }
                                                              },
                        explanation: String TODO: DEPRECATED (Do not use this key)
                    }
                student_input: dictionary
                course_data: dictionary
                student_data: dictionary
        """

        if self.validate_input:
            student_input, course_data, student_data = self.validate_system_input(student_intput,
                                                                                  course_data,
                                                                                  student_data)

        # course_data = semester_course_cleaning(course_data, student_intput['semester']) # TODO: DEPRECATED

        results = {"recommended_courses": {},
                   "sorted_recommended_courses": [],
                   "explanation": ""}

        results = make_results_template(results, course_data)

        student_info = StudentNode(results, student_intput, course_data, student_data)

        compute_recommendation(self, student_info)

        # Sort by periods
        sort_by_periods(self, student_info, self.top_n, include_keywords=True, include_score=True,
                        include_blooms=False)

        return student_info

    def generate_explanation(self, student_info):
        self.explanation.generate_explanation(self, student_info)

    def compute_constraints(self, student_info):
        # Here you can call the integer linear programming model
        pass

    def compute_warnings(self, student_info):
        # Here you can call the warnings prediction model
        pass

    def print_config(self):
        print(f"RecSys settings: \n" +
              f"Contraints: {self.constraints} \n" +
              f"Validate_input: {self.validate_input} \n" +
              f"System_course_data: {self.system_course_data} \n" +
              f"Course_based: {self.course_based} \n" +
              f"Bloom_based: {self.bloom_based} \n" +
              f"Top_n: {self.top_n} \n")
