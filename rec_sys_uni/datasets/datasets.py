import json
from os import path
from adaptkeybert import KeyBERT as adaptKeyBERT
from keybert import KeyBERT as comKeyBERT
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rec_sys_uni.errors_checker.exceptions.rec_sys_errors import ModelDoesNotExistError
import torch

"""
TODO: Data should be preprossed before using the following functions
function: get_course_data
return: course_data : dictionary 
"""


def get_course_data(only_courses=True, except_courses=[]):
    file_name = 'rec_sys_uni/datasets/data/course/course_data.json'

    # Check if file exists
    if path.isfile(file_name) is False:
        raise Exception("File not found")

    # Read JSON file
    with open(file_name, encoding="utf8") as fp:
        raw_course_data = json.load(fp)

    final_course_data = {}
    for i in raw_course_data:
        course = i['code']
        if ("SKI" not in course and "PRO" not in course and
            "UGR" not in course and "CAP" not in course and
            "LAN" not in course and course not in except_courses) or not only_courses:
            final_course_data[course] = {
                'course_name': i['title'],
                'period': i['period'],
                'level': i['level'],
                'prerequisites': [],
                'description': i['desc'],
                'ilos': i['ilo']
            }
    return final_course_data


def get_domains_data_GPT():
    file_name = 'rec_sys_uni/datasets/data/course/domains_course_data.json'
    # Check if file exists
    if path.isfile(file_name) is False:
        raise Exception("File not found")
    # Read JSON file
    with open(file_name, encoding="utf8") as fp:
        adaptation_data = json.load(fp)

    return adaptation_data


def get_keyword_data_GPT():
    file_name = 'rec_sys_uni/datasets/data/course/keywords_course_data.json'
    # Check if file exists
    if path.isfile(file_name) is False:
        raise Exception("File not found")
    # Read JSON file
    with open(file_name, encoding="utf8") as fp:
        adaptation_data = json.load(fp)

    return adaptation_data


def calculate_zero_shot_adaptation(course_data, model_name, attention,
                                   adaptive_thr: float = 0.15,
                                   minimal_similarity_zeroshot: float = 0.8):
    """
    :param course_data: Course Data
    :param model_name: Model Name
    :param attention: Attention Word/s
    :param adaptive_thr: adaptive threshold for the zero-shot adaptation
    :param minimal_similarity_zeroshot: minimal similarity between a candidate and a domain word for the zero-shot adaptation
    """
    # Model Check
    try:
        SentenceTransformer(model_name)
    except Exception:
        raise ModelDoesNotExistError("Such Model Name does not exist")

    list_courses = [x for x in course_data]
    print(f"Calculate Zero-Shot Adaptation based on course title for the model: {model_name}")
    print(f"System will calculate this list of courses: {list_courses}")
    print(f"Check information above!!!")
    user_input = input("Do you CONFIRM? (Y/N) any other input will be considered as No")
    if user_input == 'Y':
        path = f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/title_training/domain_word'

        # Create a new directory because it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"The new directory {path} for {model_name} is created!")

        if isinstance(attention, str):
            attention = [attention]

        for code in list_courses:
            print("Calculating domain word embedding for " + str(code))
            kw_model = adaptKeyBERT(model=SentenceTransformer(model_name), zero_adapt=True)
            kw_model.zeroshot_pre_train(attention, adaptive_thr=adaptive_thr,
                                        minimal_similarity_zeroshot=minimal_similarity_zeroshot)
            np.save(f'{path}/domain_embed_{code}.npy', kw_model.domain_word_embeddings)


def calculate_few_shot_adaptation(course_data, model_name, attention,
                                  lr=1e-4, epochs=100,  # Training Parameters
                                  start_index=0,
                                  include_description=True,
                                  include_title=False,
                                  include_ilos=False, ):
    """
    :param course_data: Course Data
    :param model_name: Model Name
    :param attention: Attention Word/s
    :param lr: Learning Rate for Attention Layer
    :param epochs: Number of Epochs for Attention Layer
    :param start_index: Start Index for the list of courses
    :param include_description: Include Description in the training
    :param include_title: Include Title in the training
    :param include_ilos: Include ILOs in the training
    """

    # Model Check
    try:
        SentenceTransformer(model_name)
    except Exception:
        raise ModelDoesNotExistError("Such Model Name does not exist")

    list_courses = [x for x in course_data]
    print(f"Calculate Few-Shot Adaptation based on course title for the model: {model_name}")
    print(f"System will calculate this list of courses: {list_courses}")
    print(f"Check information above!!!")
    user_input = input("Do you CONFIRM? (Y/N) any other input will be considered as No")
    if user_input == 'Y':

        path_attention = f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/title_training/attention_layer'
        path_target = f'rec_sys_uni/datasets/data/adaptation_model/{model_name}/title_training/target_embed'

        # Create a new directory because it does not exist
        if not os.path.exists(path_target):
            os.makedirs(path_target)
            print(f"The new directory {path_target} for {model_name} is created!")

        # Create a new directory because it does not exist
        if not os.path.exists(path_attention):
            os.makedirs(path_attention)
            print(f"The new directory {path_attention} for {model_name} is created!")

        if isinstance(attention, str):
            attention = [attention]

        for i, code in enumerate(list_courses):
            if (i >= start_index):
                print("Calculating for " + str(code))
                print("Index " + str(i) + " out of " + str(len(list_courses)))
                desc = ""

                if include_title:
                    desc += course_data[code]['course_name']
                if include_description:
                    desc += course_data[code]['description']
                if include_ilos:
                    desc += "\n".join(course_data[code]['ilos'])

                kw_model = adaptKeyBERT(model=model_name, domain_adapt=True)
                kw_model.pre_train([desc], [attention], lr=lr, epochs=epochs)
                torch.save(kw_model.attention_layer, f'{path_attention}/attention_layer_{code}.pth')
                torch.save(kw_model.target_word_embeddings_pt, f'{path_target}/target_embed_{code}.pth')


def calculate_precomputed_courses(course_data, model_name,
                                  include_description=True,
                                  include_title=False,
                                  include_ilos=False, ):
    try:
        SentenceTransformer(model_name)
    except Exception:
        raise ModelDoesNotExistError("Such Model Name does not exist")

    list_courses = [x for x in course_data]
    print(f"Calculate precomputed courses for the model: {model_name}")
    print(f"System will calculate this list of courses: {list_courses}")
    print(f"Check information above!!!")
    user_input = input("Do you CONFIRM? (Y/N) any other input will be considered as No")
    if user_input == 'Y':

        path = f'rec_sys_uni/datasets/data/course/precomputed_courses/{model_name}'
        # Create a new directory because it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"The new directory {path} for {model_name} is created!")

    kw_model = comKeyBERT(model=model_name)
    for code in course_data:
        print("Calculating for " + str(code))
        desc = ""
        if include_title:
            desc += course_data[code]['course_name']
        if include_description:
            desc += course_data[code]['description']
        if include_ilos:
            desc += "\n".join(course_data[code]['ilos'])

        embed = kw_model.model.embed([desc])
        np.save(f'{path}/course_embed_{code}.npy', embed)


def get_student_data():
    """
    function: get_student_data
    return: student_data : dictionary
    """
    # Read JSON file
    with open('rec_sys_uni/datasets/data/student/student.json', encoding="utf8") as fp:
        student_data = json.load(fp)
    return student_data
