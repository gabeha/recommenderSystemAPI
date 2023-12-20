import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBased:
    def __init__(self,
                 model_name: str = "all-MiniLM-L12-v2",
                 distance: str = 'cos',
                 score_alg: str = 'sum',
                 scaler: str = 'None', # 'MaxMin' or 'None
                 ):
        self.model_name = model_name
        self.distance = distance
        self.score_alg = score_alg
        self.scaler = scaler

    def compute_course_similarity(self, course_list, student_data):
        course_taken_tmp = []
        course_taken_embeddings = []
        for i in student_data['courses_taken']:
            if i[:3] != 'COR':
                course_taken_tmp.append(i)
                course_taken_embeddings.append(np.load(
                    f'rec_sys_uni/datasets/data/course/precomputed_courses/{self.model_name}/course_embed_{i}.npy'))
        course_taken_embeddings = np.array(course_taken_embeddings)
        course_taken_embeddings = course_taken_embeddings.reshape(course_taken_embeddings.shape[0], -1)


        course_list_tmp = []
        course_list_embeddings = []
        for i in course_list:
            course_list_tmp.append(i)
            course_list_embeddings.append(np.load(
                f'rec_sys_uni/datasets/data/course/precomputed_courses/{self.model_name}/course_embed_{i}.npy'))

        output_similarity = {}
        for index, course_code in enumerate(course_list_tmp):
            doc_embedding = course_list_embeddings[index].reshape(1, -1)
            if self.distance == 'cos':
                distances = cosine_similarity(doc_embedding, course_taken_embeddings)
            elif self.distance == 'dot':
                distances = np.dot(doc_embedding, course_taken_embeddings.T)
            output_similarity[course_code] = {}
            for j, code_taken in enumerate(course_taken_tmp):
                output_similarity[course_code][code_taken] = round(float(distances[0][j]), 4)
        return output_similarity
