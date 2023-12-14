from rec_sys_uni.errors_checker.errors import check_student_input, check_student_data, check_course_data
from rec_sys_uni.datasets.datasets import get_course_data, get_student_data
from rec_sys_uni._rec_sys_helpers import make_results_template, StudentNode, sort_by_periods
from rec_sys_uni.rec_systems._systems import *
from rec_sys_uni.rec_systems.keyword_based_sys.keyword_based import KeywordBased
from rec_sys_uni.rec_systems.bloom_based_sys.bloom_based import BloomBased
from rec_sys_uni.rec_systems.llm_explanation.LLM import LLM
from rec_sys_uni.rec_systems.warning_model.warning_model import WarningModel
import json
from datetime import datetime
import pymongo
from bson.objectid import ObjectId
from rec_sys_uni.rec_systems.planners.ucm_planner import UCMPlanner


class RecSys:

    def __init__(self,
                 keyword_based: KeywordBased = None,
                 bloom_based: BloomBased = None,
                 explanation: LLM = None,
                 warning_model: WarningModel = None,
                 planner: UCMPlanner = None,
                 validate_input: bool = True,
                 top_n: int = 7,
                 ):
        self.keyword_based = keyword_based
        self.bloom_based = bloom_based
        self.explanation = explanation
        self.warning_model = warning_model
        self.planner = planner
        self.validate_input = validate_input
        self.top_n = top_n
        self.db = pymongo.MongoClient("mongodb://localhost:27017/")["RecSys"]

    def validate_system_input(self,
                              student_input,
                              course_data,
                              student_data,
                              system_course_data,
                              system_student_data):
        """
        function: validate_system_input
        description: validate the format of the input data
        """

        check_student_input(student_input)

        if system_student_data:
            student_data = get_student_data()

        if system_course_data:
            exception_courses = []
            if student_data:
                exception_courses = list(student_data['courses_taken'].keys())
            course_data = get_course_data(except_courses=exception_courses)

        check_course_data(course_data)

        if student_data is not None:
            check_student_data(student_data)

        return student_input, course_data, student_data

    def get_recommendation(self,
                           student_intput,
                           course_data=None,
                           student_data=None,
                           system_course_data=True,
                           system_student_data=False,
                           ):
        """
        function: get_recommendation
        description: get Student Node object
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
        return: StudentNode object, which contains: (check _rec_sys_helpers.py)
                results: dictionary of recommended courses
                e.g. {
                        recommended_courses: dictionary {
                                                            course_id(String):
                                                                        {
                                   Total score of each model                score: float,
                                   Course periods                           period: [int, ...] or [[int, int], ...],    e.g [1, 4] or [[1, 2, 3], [4, 5, 6]] or [[1,2]]
                                                                            warning: boolean
                                                                            warning_recommendation: list of warning recommendation
                                   (after applying CourseBased model)       keywords: {scores to each keyword}
                                   (after applying BloomBased model)        blooms: {scores to each bloom}
                                                                        },
                                                            ...
                                                        },
                        sorted_recommended_courses: list of courses {'course_code': String,
                                                                     'course_name': String,
                                                                     'warning': boolean,
                                                                     'warning_recommendation': list of warning recommendation,
                                                                     'keywords': {keywords(String): weight(float), ...} ,
                                                                     'blooms': {blooms(String): weight(float), ...}
                                                                     'score': float,

                        structured_recommendation: dictionary {
                                                                semester_1: dictionary {
(if you apply sort_by_periods function,                             period_1: list of course_id(String), condition <= top_n
when you will have structured_recommendation                        period_2: list of course_id(String), condition <= top_n
and sorted_recommended_courses key in the results)              },
                                                                semester_2: dictionary {
                                                                    period_4: list of course_id(String), condition <= top_n
                                                                    period_5: list of course_id(String), condition <= top_n
                                                                }
                                                              }
                    }
                student_input: dictionary
                course_data: dictionary
                student_data: dictionary
        """

        if self.validate_input:
            student_input, course_data, student_data = self.validate_system_input(student_intput,
                                                                                  course_data,
                                                                                  student_data,
                                                                                  system_course_data,
                                                                                  system_student_data)

        # Make results template
        results = make_results_template(course_data)

        # Create StudentNode object
        student_info = StudentNode(results, student_intput, course_data, student_data)

        # Compute recommendation and store in student_info.results['recommended_courses']
        if self.keyword_based or self.bloom_based:
            compute_recommendation(self, student_info)

        # Compute warnings and store in student_info.results['recommended_courses']
        if self.warning_model:
            compute_warnings(self, student_info)

        # Sort by periods
        sorted_recommendation_list, structured_recommendation = sort_by_periods(self,
                                                                                student_info.results['recommended_courses'],
                                                                                student_info.course_data,
                                                                                self.top_n,
                                                                                include_keywords=False,
                                                                                include_score=False,
                                                                                include_blooms=False,
                                                                                include_warnings=True,
                                                                                percentage=False
                                                                                )
        student_info.results['sorted_recommended_courses'] = sorted_recommendation_list
        student_info.results['structured_recommendation'] = structured_recommendation

        # Save student_info to database
        now = datetime.now()

        collection = self.db["student_results"]
        input_dict = {
            "student_id": "123",
            "student_input": student_info.student_input,
            "course_data": student_info.course_data,
            "student_data": student_info.student_data,
            "results": student_info.results,
            "time": now.strftime("%d/%m/%Y %H:%M:%S")
        }
        object_db = collection.insert_one(input_dict)
        student_info.set_id(object_db.inserted_id)

        # Return StudentNode object
        return student_info

    def generate_explanation(self, student_id, course_code):
        return json.loads(compute_explanations(self, student_id, course_code))

    def make_timeline(self, student_id):
        return compute_timeline(self, student_id)

    def print_config(self):
        print(f"RecSys settings: \n" +
              f"Validate_input: {self.validate_input} \n" +
              f"Keyword_based: {self.keyword_based} \n" +
              f"Bloom_based: {self.bloom_based} \n" +
              f"Explanation: {self.explanation} \n" +
              f"Warning_model: {self.warning_model} \n" +
              f"Planner: {self.planner}" +
              f"Top_n: {self.top_n} \n")
