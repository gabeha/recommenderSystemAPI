# from intel_extension_for_transformers.transformers import OptimizedModel # Do not delete
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Tuple
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig, AutoTokenizer

from rec_sys_uni.datasets.datasets import get_domains_data_GPT
from rec_sys_uni.errors_checker.exceptions.rec_sys_errors import ModelDoesNotExistError, PrecomputedCoursesError, \
    AdaptationLayerError


class EmbeddingModel:
    def __init__(self, model_name):
        pass

    # def __init__(self, model_name):
    #     try:
    #         config = AutoConfig.from_pretrained(model_name)
    #         self.model = OptimizedModel.from_pretrained(model_name, config=config)
    #         self.model.eval()
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     except Exception:
    #         raise ModelDoesNotExistError("Such Model Name does not exist, check INTEL models in the huggingface.co")
    #
    # def embed(self, docs):
    #     input = self.tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
    #     output = self.model(**input)
    #     return output[1]


class KeywordBased:

    def __init__(self,
                 model_name: str = "all-MiniLM-L12-v2",
                 top_n: int = 100,
                 seed_help: bool = False,
                 domain_adapt: bool = False,
                 zero_adapt: bool = False,
                 seed_type: str = 'title',  # 'title' or 'domains'
                 domain_type: str = 'title',  # 'title' or 'domains'
                 zero_type: str = 'title',  # 'title' or 'domains'
                 adaptive_thr: float = 0.0,
                 minimal_similarity_zeroshot: float = 0.8,
                 score_alg: str = 'rrf',  # 'sum' or 'rrf'
                 distance: str = 'cos',  # 'cos' or 'dot'
                 backend: str = 'keyBert',  # 'keyBert' or 'Intel'
                 scaler: str = 'MaxMin',  # 'MaxMin' or 'None
                 sent_splitter: bool = False,
                 precomputed_course=False
                 ):
        """
        The constructor for KeywordBasedRecSys class.
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
        :param score_alg: score algorithm either 'sum' or 'rrf'
        :param distance: distance metric either 'cos' or 'dot'
        :param backend: backend either 'keyBert' or 'Intel'
        :param scaler: apply min-max scaler to the keywords weights
        :param sent_splitter: apply sentence splitter
        :param precomputed_course: use precomputed course embeddings or not
        """
        if backend == 'keyBert':
            try:
                transformer_model = SentenceTransformer(model_name)
            except Exception:
                raise ModelDoesNotExistError("Such Model Name does not exist")

        # Initiate KeyBERT model
        if backend == 'Intel':
            self.kw_model = EmbeddingModel(model_name)
        elif backend == 'keyBert':
            self.kw_model = KeyBERT(model=transformer_model)
        self.distance = distance
        self.backend = backend
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
        self.score_alg = score_alg
        self.scaler = scaler
        self.sent_splitter = sent_splitter
        self.precomputed_course = precomputed_course

        # Settings check
        if precomputed_course:
            if not os.path.exists(f'rec_sys_uni/datasets/data/course/precomputed_courses/{self.model_name}'):
                raise PrecomputedCoursesError(f"You did not compute embeddings for such model -> {self.model_name}")
        if domain_adapt:
            if not os.path.exists(
                    f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{domain_type}_training/target_embed'):
                raise AdaptationLayerError(
                    f"You did not compute target embed embeddings with type {domain_type} for such model -> {self.model_name}")
            if not os.path.exists(
                    f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{domain_type}_training/attention_layer'):
                raise AdaptationLayerError(
                    f"You did not compute attention layer with type {domain_type} for such model -> {self.model_name}")
        if zero_adapt:
            if not os.path.exists(
                    f'rec_sys_uni/datasets/data/adaptation_model/{self.model_name}/{zero_type}_training/domain_word'):
                raise AdaptationLayerError(
                    f"You did not compute domain word embeddings with type {zero_type} for such model -> {self.model_name}")

    def print_config(self):
        print(f"CourseBasedRecSys config: \n" +
              f"model_name: {self.model_name}\n" +
              f"seed_help: {self.seed_help}\n" +
              f"domain_adapt: {self.domain_adapt}\n" +
              f"zero_adapt: {self.zero_adapt}\n" +
              f"seed_type: {self.seed_type}\n" +
              f"domain_type: {self.domain_type}\n" +
              f"zero_type: {self.zero_type}\n" +
              f"adaptive_thr: {self.adaptive_thr}\n" +
              f"minimal_similarity_zeroshot: {self.minimal_similarity_zeroshot}\n" +
              f"score_alg: {self.score_alg}\n" +
              f"distance: {self.distance}\n" +
              f"backend: {self.backend}\n" +
              f"scaler: {self.scaler}\n" +
              f"sent_splitter: {self.sent_splitter}\n" +
              f"precomputed_course: {self.precomputed_course}\n")

    def recommend(self, course_data, student_keywords):

        domains_data = get_domains_data_GPT()  # Get domains data

        # Put keywords in a list
        keywords = []
        for i in student_keywords:
            keywords.append(i)

        # Get seed keywords, course descriptions, course codes, and doc embeddings
        course_descriptions = []
        course_codes = []
        seed_keywords = []
        doc_embeddings = None
        for i in course_data:
            desc = course_data[i]['description'].replace('course', course_data[i]['course_name'])
            desc += ", ".join(course_data[i]['ilos']).replace('course', course_data[i]['course_name'])
            course_descriptions.append(desc)
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

        if not self.sent_splitter:
            # Extract probabilities of keywords
            keywords_relevance = extract_keywords_relevance(docs=course_descriptions,
                                                            candidates=keywords,
                                                            keyBERT=self.kw_model,
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
                                                            distance=self.distance)

            keywords_output = {}
            for index, course in enumerate(course_data):
                keywords_output[course] = {}
                keywords = keywords_relevance[index]
                for keyword in keywords:
                    keywords_output[course][keyword[0]] = keyword[1]

            return keywords_output
        else:
            nlp = spacy.load("en_core_web_sm")
            course_sentences = {}
            for index, description in enumerate(course_descriptions):
                doc = nlp(description)
                doc = [sent.text for sent in doc.sents]
                code = [course_codes[index] for _ in range(len(doc))]
                # Extract probabilities of sentences
                sentence_relevance = extract_keywords_relevance(docs=doc,
                                                                candidates=keywords,
                                                                keyBERT=self.kw_model,
                                                                model_name=self.model_name,
                                                                course_codes=code,
                                                                domain_type=self.domain_type,
                                                                zero_type=self.zero_type,
                                                                domain_adapt=self.domain_adapt,
                                                                zero_adapt=self.zero_adapt,
                                                                adaptive_thr=self.adaptive_thr,
                                                                minimal_similarity_zeroshot=self.minimal_similarity_zeroshot,
                                                                seed_keywords=seed_keywords,
                                                                sent_splitter=self.sent_splitter,
                                                                distance=self.distance)
                course_sentences[course_codes[index]] = sentence_relevance
            # print(course_sentences)
            # TODO: Implementation of Linear Programming
            raise NotImplementedError("Sentence Splitter is not implemented yet")

def apply_zero_adaptation(candidate_embeddings, doc_embedding, domain_word_embeddings, adaptive_thr,
                          minimal_similarity_zeroshot):
    computed_embeddings = []
    for candidate_embedding in candidate_embeddings:
        candidate_embedding = candidate_embedding.reshape(1, -1)
        max_similarity = np.max(cosine_similarity(candidate_embedding, domain_word_embeddings))
        if max_similarity < minimal_similarity_zeroshot:
            computed_embeddings.append(candidate_embedding[0])
        else:
            temp_embedding = (
                                     1 - adaptive_thr * max_similarity) * candidate_embedding + adaptive_thr * max_similarity * doc_embedding
            computed_embeddings.append(temp_embedding[0])
    computed_embeddings = np.stack(computed_embeddings)
    return computed_embeddings


def extract_keywords_relevance(
        docs: Union[str, List[str]],
        candidates: List[str],
        keyBERT,
        model_name: str,
        course_codes: List[str] = None,
        domain_type: str = 'title',
        zero_type: str = 'title',
        domain_adapt: bool = False,
        zero_adapt: bool = False,
        adaptive_thr: float = 0.0,
        minimal_similarity_zeroshot: float = 0.8,
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
        sent_splitter: bool = False,
        distance: str = 'cos'
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
        if isinstance(keyBERT, EmbeddingModel):
            doc_embeddings = keyBERT.embed(docs)
        elif isinstance(keyBERT, KeyBERT):
            doc_embeddings = keyBERT.model.embed(docs)

    if isinstance(keyBERT, EmbeddingModel):
        word_embeddings = keyBERT.embed(candidates)
    elif isinstance(keyBERT, KeyBERT):
        word_embeddings = keyBERT.model.embed(candidates)

    # Find cosine similarity between course description and keywords
    all_keywords = []
    for index, _ in enumerate(docs):

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

        # Seed Filtering
        if seed_keywords is not None:
            if isinstance(keyBERT, EmbeddingModel):
                seed_embeddings = keyBERT.embed([" ".join(seed_keywords[index])])
            elif isinstance(keyBERT, KeyBERT):
                seed_embeddings = keyBERT.model.embed([" ".join(seed_keywords[index])])

            doc_embedding = np.average(
                [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
            )

        # Compute distances between keywords and document
        if distance == 'cos':
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
        elif distance == 'dot':
            distances = np.dot(doc_embedding, candidate_embeddings.T)

        # keywords = [
        #                (candidates[index], round(float(distances[0][index]), 4))
        #                for index in distances.argsort()[0][-top_n:]
        #            ][::-1]
        if not sent_splitter:
            keywords = [
                (candidates[index], round(float(distances[0][index]), 4))
                for index in range(len(candidates))
            ]
        else:
            keywords = {docs[index]: [
                (candidates[index], round(float(distances[0][index]), 4))
                for index in range(len(candidates))
            ]}

        all_keywords.append(keywords)

    return all_keywords

# %%
