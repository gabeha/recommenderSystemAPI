import numpy as np
import json


class BloomBased:

    def __init__(self,
                 precomputed_blooms: bool = True,
                 top_n: int | None = 4,
                 score_alg: str = 'sum',
                 scaler: str = 'None', # 'MaxMin' or 'None
                 ):
        """
        Initializes the BloomBasedRecSys with the given parameters.

        Args:
            precomputed_blooms (bool, optional): If True, uses precomputed Bloom's values. Defaults to True.
            top_n (int, optional): If assigned, it ensures only the top_n recommendations are also weighted according to Bloom's taxonomy.
        """
        assert precomputed_blooms, 'Currently precomputed_blooms=False is not implemented - Dennis'
        self.precomputed_blooms = precomputed_blooms
        self.score_alg = score_alg
        self.scaler = scaler
        self.top_n = top_n

    def recommend(self, course_data, student_blooms):
        """
        Generates course recommendations based on the student's input and course data.

        Args:
            student_info: The recommendation system object that contains course data and student input.

        Raises:
            AssertionError: If precomputed_blooms is set to False.
        """
        if self.precomputed_blooms:
            with open('rec_sys_uni/datasets/data/course/precomputed_blooms.json', 'r') as file:
                blooms = json.load(file)

        else:
            raise AssertionError('Opa! Did you assign precomputed_blooms=False? - Dennis')

        blooms_output = {}
        for index, course in enumerate(course_data):
            blooms_output[course] = {}
            blooms_values = blooms[course]
            for bloom in blooms_values:
                blooms_output[course][bloom] = blooms_values[bloom]

        return blooms_output
