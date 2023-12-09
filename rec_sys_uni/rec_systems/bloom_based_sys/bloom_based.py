import numpy as np
import json


class BloomBased:

    def __init__(self, precomputed_blooms: bool = True, top_n: int | None = 4):
        """
        Initializes the BloomBasedRecSys with the given parameters.

        Args:
            precomputed_blooms (bool, optional): If True, uses precomputed Bloom's values. Defaults to True.
            top_n (int, optional): If assigned, it ensures only the top_n recommendations are also weighted according to Bloom's taxonomy.
        """
        assert precomputed_blooms, 'Currently precomputed_blooms=False is not implemented - Dennis'
        self.precomputed_blooms = precomputed_blooms
        self.top_n = top_n

    def recommend(self, student_info):
        """
        Generates course recommendations based on the student's input and course data.

        Args:
            student_info: The recommendation system object that contains course data and student input.

        Raises:
            AssertionError: If precomputed_blooms is set to False.
        """
        course_data = student_info.course_data
        student_blooms = student_info.student_input['blooms']

        if self.precomputed_blooms:
            with open('rec_sys_uni/datasets/data/course/precomputed_blooms.json', 'r') as file:
                blooms = json.load(file)

        else:
            raise AssertionError('Opa! Did you assign precomputed_blooms=False? - Dennis')

        recommended_courses = student_info.results['recommended_courses']
        top_scores = sorted([recommended_courses[code]['score'] for code in course_data][-self.top_n:])
        for (idx, code) in enumerate(course_data):
            for label in student_blooms:
                if recommended_courses[code]['score'] in top_scores:
                    recommended_courses[code]['score'] += blooms[code][label] * student_blooms[label]
                recommended_courses[code]["blooms"][label] = blooms[code][label]
        student_info.results['recommended_courses'] = recommended_courses
