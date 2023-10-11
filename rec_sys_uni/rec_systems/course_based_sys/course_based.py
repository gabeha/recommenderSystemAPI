from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Tuple

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from keybert._mmr import mmr
from keybert._maxsum import max_sum_distance

from rec_sys_uni.errors_checker.exceptions.rec_sys_errors import CourseBasedSettingsError


class CourseBasedRecSys:

    def __init__(self,
                 course_based_model=SentenceTransformer('all-MiniLM-L6-v2'),
                 keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                 stop_words: Union[str, List[str]] = "english",
                 top_n: int = 10,
                 min_df: int = 1,
                 use_maxsum: bool = False,
                 use_mmr: bool = False,
                 diversity: float = 0.5,
                 nr_candidates: int = 20,
                 highlight: bool = False,
                 seed_keywords: Union[List[str], List[List[str]]] = None,
                 threshold: float = None,
                 force_keywords=False,
                 precomputed_course=False
                 ):
        self.course_based_model = course_based_model
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.stop_words = stop_words
        self.top_n = top_n
        self.min_df = min_df
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.nr_candidates = nr_candidates
        self.highlight = highlight
        self.seed_keywords = seed_keywords
        self.threshold = threshold
        self.force_keywords = force_keywords
        self.precomputed_course = precomputed_course

    def recommend(self, recSys):

        kw_model = KeyBERT(model=self.course_based_model)

        course_data = recSys.course_data
        student_input = recSys.student_input['keywords']

        keywords = []
        for i in student_input:
            keywords.append(i)

        course_descriptions = []
        for i in course_data:
            course_descriptions.append(course_data[i]['description'])

        doc_embeddings = None
        word_embeddings = None

        if self.precomputed_course:
            doc_embeddings = np.load('doc_embeddings.npy')  # TODO: Need to implemented
            word_embeddings = np.load('word_embeddings.npy')  # TODO: Need to implemented

        keywords_relevance = extract_keywords_relevance(course_descriptions, keywords, kw_model,
                                                        doc_embeddings=doc_embeddings,
                                                        word_embeddings=word_embeddings,
                                                        keyphrase_ngram_range=self.keyphrase_ngram_range,
                                                        stop_words=self.stop_words,
                                                        top_n=self.top_n,
                                                        min_df=self.min_df,
                                                        use_maxsum=self.use_maxsum,
                                                        use_mmr=self.use_mmr,
                                                        diversity=self.diversity,
                                                        nr_candidates=self.nr_candidates,
                                                        highlight=self.highlight,
                                                        seed_keywords=self.seed_keywords,
                                                        threshold=self.threshold,
                                                        force_keywords=self.force_keywords)

        # Sum all weights of keywords
        recommended_courses = recSys.results['recommended_courses']
        for index, code in enumerate(course_data):
            keywords_weightes = keywords_relevance[index]
            score = 0
            for i in keywords_weightes:
                score += i[1] * student_input[i[0]]
            recommended_courses[code]['score'] = score
        recSys.results['recommended_courses'] = recommended_courses


def extract_keywords_relevance(
        docs: Union[str, List[str]],
        candidates: List[str],
        keyBERT: KeyBERT,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
        threshold: float = None,
        force_keywords=False
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Extract keywords and/or keyphrases

    To get the biggest speed-up, make sure to pass multiple documents
    at once instead of iterating over a single document.

    Arguments:
        docs: The document(s) for which to extract keywords/keyphrases
        candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)

        keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                               NOTE: This is not used if you passed a `vectorizer`.
        stop_words: Stopwords to remove from the document.
                    NOTE: This is not used if you passed a `vectorizer`.
        top_n: Return the top n keywords/keyphrases
        min_df: Minimum document frequency of a word across all documents
                if keywords for multiple documents need to be extracted.
                NOTE: This is not used if you passed a `vectorizer`.
        use_maxsum: Whether to use Max Sum Distance for the selection
                    of keywords/keyphrases.
        use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                 selection of keywords/keyphrases.
        diversity: The diversity of the results between 0 and 1 if `use_mmr`
                   is set to True.
        nr_candidates: The number of candidates to consider if `use_maxsum` is
                       set to True.
        highlight: Whether to print the document and highlight its keywords/keyphrases.
                   NOTE: This does not work if multiple documents are passed.
        seed_keywords: Seed keywords that may guide the extraction of keywords by
                       steering the similarities towards the seeded keywords.
                       NOTE: when multiple documents are passed,
                       `seed_keywords`funtions in either of the two ways below:
                       - globally: when a flat list of str is passed, keywords are shared by all documents,
                       - locally: when a nested list of str is passed, keywords differs among documents.
        doc_embeddings: The embeddings of each document.
        word_embeddings: The embeddings of each potential keyword/keyphrase across
                         across the vocabulary of the set of input documents.
                         NOTE: The `word_embeddings` should be generated through
                         `.extract_embeddings` as the order of these embeddings depend
                         on the vectorizer that was used to generate its vocabulary.
        force_keywords: Forse to compute distance for all input candidate keywords.

    Returns:
        keywords: The keywords for a document with their respective distances
                  to the input document.

    """
    # Check for a single, empty document
    if isinstance(docs, str):
        if docs:
            docs = [docs]
        else:
            return []

    # Extract potential words using a vectorizer / tokenizer
    try:
        count = CountVectorizer(
            ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            vocabulary=candidates,
        ).fit(docs)
    except ValueError:
        return []

    # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
    # and will be removed in 1.2. Please use get_feature_names_out instead.
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = count.get_feature_names_out()
    else:
        words = count.get_feature_names()
    df = count.transform(docs)

    # Check if the right number of word embeddings are generated compared with the vectorizer
    if word_embeddings is not None:
        if word_embeddings.shape[0] != len(words):
            raise ValueError("Make sure that the `word_embeddings` are generated from the function "
                             "`.extract_embeddings`. \nMoreover, the `candidates`, `keyphrase_ngram_range`,"
                             "`stop_words`, and `min_df` parameters need to have the same values in both "
                             "`.extract_embeddings` and `.extract_keywords`.")

    # Extract embeddings
    if doc_embeddings is None:
        doc_embeddings = keyBERT.model.embed(docs)
    if word_embeddings is None:
        word_embeddings = keyBERT.model.embed(words)

    # Guided KeyBERT either local (keywords shared among documents) or global (keywords per document)
    # if seed_keywords is not None:
    #     if isinstance(seed_keywords[0], str):
    #         seed_embeddings = self.model.embed(seed_keywords).mean(axis=0, keepdims=True)
    #     elif len(docs) != len(seed_keywords):
    #         raise ValueError("The length of docs must match the length of seed_keywords")
    #     else:
    #         seed_embeddings = np.vstack([
    #             self.model.embed(keywords).mean(axis=0, keepdims=True)
    #             for keywords in seed_keywords
    #         ])
    #     doc_embeddings = ((doc_embeddings * 3 + seed_embeddings) / 4)

    # Find keywords
    all_keywords = []
    for index, _ in enumerate(docs):

        try:
            # Select embeddings
            if force_keywords:
                candidates = words
                candidate_embeddings = word_embeddings
            else:
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
            doc_embedding = doc_embeddings[index].reshape(1, -1)

            # Maximal Marginal Relevance (MMR)
            if use_mmr:
                keywords = mmr(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    diversity,
                )

            # Max Sum Distance
            elif use_maxsum:
                keywords = max_sum_distance(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    nr_candidates,
                )

            # Cosine-based keyword extraction
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [
                               (candidates[index], round(float(distances[0][index]), 4))
                               for index in distances.argsort()[0][-top_n:]
                           ][::-1]

            all_keywords.append(keywords)

        # Capturing empty keywords
        except ValueError:
            all_keywords.append([])

    # Highlight keywords in the document
    # if len(all_keywords) == 1:
    #     if highlight:
    #         highlight_document(docs[0], all_keywords[0], count)
    #     all_keywords = all_keywords[0]

    # Fine-tune keywords using an LLM
    # if self.llm is not None:
    #     import torch
    #     doc_embeddings = torch.from_numpy(doc_embeddings).float().to("cuda")
    #     if isinstance(all_keywords[0], tuple):
    #         candidate_keywords = [[keyword[0] for keyword in all_keywords]]
    #     else:
    #         candidate_keywords = [[keyword[0] for keyword in keywords] for keywords in all_keywords]
    #     keywords = self.llm.extract_keywords(
    #         docs,
    #         embeddings=doc_embeddings,
    #         candidate_keywords=candidate_keywords,
    #         threshold=threshold
    #     )
    #     return keywords
    return all_keywords
