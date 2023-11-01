from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Tuple
import torch
import os

from sklearn.metrics.pairwise import cosine_similarity
from rec_sys_uni.datasets.datasets import get_domains_data_GPT
from rec_sys_uni.errors_checker.exceptions.rec_sys_errors import ModelDoesNotExistError, PrecomputedCoursesError, AdaptationLayerError


class CourseBasedRecSys:

    def __init__(self,
                 model_name: str = 'all-MiniLM-L12-v2',
                 top_n: int = 100,
                 seed_help: bool = False,
                 domain_adapt: bool = False,
                 zero_adapt: bool = False,
                 seed_type: str = 'title',  # 'title' or 'domains'
                 domain_type: str = 'title',  # 'title' or 'domains'
                 zero_type: str = 'title',  # 'title' or 'domains'
                 adaptive_thr: float = 0.0,
                 minimal_similarity_zeroshot: float = 0.8,
                 precomputed_course=False
                 ):
        """
        The constructor for CourseBasedRecSys class.
        :param model_name: model name of the sentence transformer
        :param top_n: number of keywords to be extracted
        :param seed_help: apply seed help
        :param domain_adapt: apply domain adaptation
        :param zero_adapt: apply zero-shot adaptation
        :param seed_type: type of the seed help either 'title' or 'domains'
        :param domain_type: type of the domain adaptation either 'title' or 'domains'
        :param zero_type: type of the zero-shot adaptation either 'title' or 'domains'
        :param adaptive_thr: adaptive threshold for the zero-shot adaptation
        :param minimal_similarity_zeroshot: minimal similarity between a candidate and a domain word for the zero-shot adaptation
        :param precomputed_course: use precomputed course embeddings or not
        """
        try:
            self.course_based_model = SentenceTransformer(model_name)
        except Exception:
            raise ModelDoesNotExistError("Such Model Name does not exist")

        self.model_name = model_name
        self.top_n = top_n
        self.seed_help = seed_help
        self.seed_type = seed_type
        self.domain_type = domain_type
        self.zero_type = zero_type
        self.domain_adapt = domain_adapt
        self.zero_adapt = zero_adapt
        self.adaptive_thr = adaptive_thr
        self.minimal_similarity_zeroshot = minimal_similarity_zeroshot
        self.precomputed_course = precomputed_course

        # Settings check
        if precomputed_course:
            if not os.path.exists(f'rec_sys_uni/datasets/data/course/precomputed_courses/{self.model_name}'):
                raise PrecomputedCoursesError(f"You did not compute embeddings for such model -> {self.model_name}")
        if domain_adapt:
            if not os.path.exists(f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{domain_type}_training/target_embed'):
                raise AdaptationLayerError(f"You did not compute target embed embeddings with type {domain_type} for such model -> {self.model_name}")
            if not os.path.exists(f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{domain_type}_training/attention_layer'):
                raise AdaptationLayerError(f"You did not compute attention layer with type {domain_type} for such model -> {self.model_name}")
        if zero_adapt:
            if not os.path.exists(f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{zero_type}_training/domain_word'):
                raise AdaptationLayerError(f"You did not compute domain word embeddings with type {zero_type} for such model -> {self.model_name}")

    def print_config(self):
        print(f"CourseBasedRecSys config: \n"+
              f"model_name: {self.model_name}\n"+
              f"seed_help: {self.seed_help}\n"+
              f"domain_adapt: {self.domain_adapt}\n"+
              f"zero_adapt: {self.zero_adapt}\n")

    def recommend(self, recSys):

        # Initiate KeyBERT model
        kw_model = KeyBERT(model=self.course_based_model)

        course_data = recSys.course_data  # Get course data
        student_input = recSys.student_input['keywords']  # Get student input
        domains_data = get_domains_data_GPT()  # Get domains data

        # Put keywords in a list
        keywords = []
        for i in student_input:
            keywords.append(i)

        # Get seed keywords, course descriptions, course codes, and doc embeddings
        course_descriptions = []
        course_codes = []
        seed_keywords = []
        doc_embeddings = None
        for i in course_data:
            course_descriptions.append(course_data[i]['description'])
            course_codes.append(i)
            if self.seed_help:
                if self.seed_type == 'title':
                    seed_keywords.append([course_data[i]['course_name']])
                elif self.seed_type == 'domains':
                    seed_keywords.append(domains_data[i])
            if self.precomputed_course:
                doc_embed_tmp = np.load(
                    f'rec_sys_uni/datasets/data/course/precomputed_courses/{self.model_name}/course_embed_{i}.npy')
                if doc_embeddings is None:
                    doc_embeddings = doc_embed_tmp
                else:
                    doc_embeddings = np.append(doc_embeddings, doc_embed_tmp, axis=0)

        if not self.seed_help: seed_keywords = None

        # Extract probabilities of keywords
        keywords_relevance = extract_keywords_relevance(docs=course_descriptions,
                                                        candidates=keywords,
                                                        keyBERT=kw_model,
                                                        model_name=self.model_name,
                                                        course_codes=course_codes,
                                                        domain_type=self.domain_type,
                                                        zero_type=self.zero_type,
                                                        domain_adapt=self.domain_adapt,
                                                        zero_adapt=self.zero_adapt,
                                                        adaptive_thr=self.adaptive_thr,
                                                        minimal_similarity_zeroshot=self.minimal_similarity_zeroshot,
                                                        doc_embeddings=doc_embeddings,
                                                        seed_keywords=seed_keywords,
                                                        top_n=self.top_n)

        # Sum all weights of keywords
        recommended_courses = recSys.results['recommended_courses']
        for index, code in enumerate(course_data):
            keywords_weightes = keywords_relevance[index]
            for i in keywords_weightes:
                recommended_courses[code]['score'] += i[1] * student_input[i[0]]
        recSys.results['recommended_courses'] = recommended_courses


def apply_zero_adaptation(candidate_embeddings, doc_embedding, domain_word_embeddings, adaptive_thr,
                          minimal_similarity_zeroshot):
    computed_embeddings = []
    for candidate_embedding in candidate_embeddings:
        candidate_embedding = candidate_embedding.reshape(1, -1)
        max_similarity = np.max(cosine_similarity(candidate_embedding, domain_word_embeddings))
        if max_similarity < minimal_similarity_zeroshot:
            computed_embeddings.append(candidate_embedding[0])
        else:
            temp_embedding = (1 - adaptive_thr * max_similarity) * candidate_embedding + adaptive_thr * max_similarity * doc_embedding
            computed_embeddings.append(temp_embedding[0])
    computed_embeddings = np.stack(computed_embeddings)
    return computed_embeddings


def extract_keywords_relevance(
        docs: Union[str, List[str]],
        candidates: List[str],
        keyBERT: KeyBERT,
        model_name: str,
        course_codes: List[str] = None,
        domain_type: str = 'title',
        zero_type: str = 'title',
        domain_adapt: bool = False,
        zero_adapt: bool = False,
        adaptive_thr: float = 0.0,
        minimal_similarity_zeroshot: float = 0.8,
        top_n: int = 100,
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Extract keywords and/or keyphrases

    To get the biggest speed-up, make sure to pass multiple documents
    at once instead of iterating over a single document.

    Arguments:
        docs: The document(s) for which to calculate keywords/keyphrases relevance

        candidates: Candidate keywords/keyphrases to use for calculating the relevance

        keyBERT model: The model to use for extraction embeddings

        model_name: The name of the model to use for extraction embeddings

        course_codes: The codes of the courses

        domain_type: The type of the domain adaptation

        zero_type: The type of the zero-shot adaptation

        domain_adapt: Whether to apply domain adaptation

        zero_adapt: Whether to apply zero-shot adaptation

        adaptive_thr: The threshold for the adaptive weighting of the domain words

        minimal_similarity_zeroshot: The minimal similarity between a candidate and a domain word

        top_n: Return the top n keywords/keyphrases with the closest distances to the original document.

        seed_keywords: Seed keywords that may guide the extraction of keywords by
                       steering the similarities towards the seeded keywords.
                       NOTE: when multiple documents are passed,
                       `seed_keywords`funtions in either of the two ways below:
                       - globally: when a flat list of str is passed, keywords are shared by all documents,
                       - locally: when a nested list of str is passed, keywords differs among documents.

        doc_embeddings: The embeddings of each document.

    Returns:
        keywords: The keywords probabilities for a document with their respective distances
                  to the input document.

    """
    # Check for a single, empty document
    if isinstance(docs, str):
        if docs:
            docs = [docs]
        else:
            return []

    # Extract embeddings
    if doc_embeddings is None:
        doc_embeddings = keyBERT.model.embed(docs)

    word_embeddings = keyBERT.model.embed(candidates)

    # Guided KeyBERT either local (keywords shared among documents) or global (keywords per document)
    if not domain_adapt and not zero_adapt:
        if seed_keywords is not None:
            if isinstance(seed_keywords[0], str):
                seed_embeddings = keyBERT.model.embed(seed_keywords).mean(axis=0, keepdims=True)
            elif len(docs) != len(seed_keywords):
                raise ValueError("The length of docs must match the length of seed_keywords")
            else:
                seed_embeddings = np.vstack([
                    keyBERT.model.embed(keywords).mean(axis=0, keepdims=True)
                    for keywords in seed_keywords
                ])
            doc_embeddings = ((doc_embeddings * 3 + seed_embeddings) / 4)

    # Find probabilities of keywords
    all_keywords = []
    for index, _ in enumerate(docs):

        try:
            candidate_embeddings = word_embeddings
            doc_embedding = doc_embeddings[index].reshape(1, -1)

            # Adoptation Layer Extension
            if domain_adapt or zero_adapt:
                code = course_codes[index]
                candidate_embeddings_pt = torch.from_numpy(candidate_embeddings)

                if domain_adapt:
                    attention_layer = torch.load(
                        f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/{domain_type}_training/attention_layer/attention_layer_{code}.pth')
                    target_word_embeddings_pt = torch.load(
                        f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/{domain_type}_training/target_embed/target_embed_{code}.pth')
                    attention_layer.eval()
                    candidate_embeddings_ = attention_layer(candidate_embeddings_pt,
                                                            target_word_embeddings_pt).detach().numpy()
                    candidate_embeddings = np.average([candidate_embeddings, candidate_embeddings_], axis=0,
                                                      weights=[2, 1])
                if zero_adapt:
                    domain_word_embeddings = np.load(
                        f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/{zero_type}_training/domain_word/domain_embed_{code}.npy')
                    candidate_embeddings = apply_zero_adaptation(candidate_embeddings, doc_embedding,
                                                                 domain_word_embeddings, adaptive_thr,
                                                                 minimal_similarity_zeroshot)

                if seed_keywords is not None:
                    seed_embeddings = keyBERT.model.embed([" ".join(seed_keywords[index])])
                    doc_embedding = np.average(
                        [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
                    )

            # Compute distances between keywords and document
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            keywords = [
                           (candidates[index], round(float(distances[0][index]), 4))
                           for index in distances.argsort()[0][-top_n:]
                       ][::-1]

            all_keywords.append(keywords)

        # Capturing empty keywords
        except ValueError:
            all_keywords.append([])

    return all_keywords
