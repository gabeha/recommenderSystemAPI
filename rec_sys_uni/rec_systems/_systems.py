from datetime import datetime
import pymongo
from rec_sys_uni.rec_systems._system_helpers import *
from bson.objectid import ObjectId
import json


def compute_recommendation(recSys, student_info):
    # Compute recommendation
    ranker_dict = {
        "score_alg": {},  # sum, rrf
        "scaler": {},
        "output": {},
    }

    # Keyword-based
    if recSys.keyword_based:
        keywords_output = recSys.keyword_based.recommend(student_info.course_data,
                                                         student_info.student_input['keywords'])
        ranker_dict["score_alg"]["keywords"] = recSys.keyword_based.score_alg
        ranker_dict["scaler"]["keywords"] = recSys.keyword_based.scaler
        ranker_dict["output"]["keywords"] = keywords_output

    # Bloom's taxonomy
    if recSys.bloom_based:
        blooms_output = recSys.bloom_based.recommend(student_info.course_data,
                                                     student_info.student_input['blooms'])
        ranker_dict["score_alg"]["blooms"] = recSys.bloom_based.score_alg
        ranker_dict["scaler"]["blooms"] = recSys.bloom_based.scaler
        ranker_dict["output"]["blooms"] = blooms_output

    # Content-based
    if recSys.content_based and student_info.student_data:
        content_based_output = recSys.content_based.compute_course_similarity(
                                            student_info.results["recommended_courses"].keys(),
                                            student_info.student_data)
        ranker_dict["score_alg"]["content"] = recSys.content_based.score_alg
        ranker_dict["scaler"]["content"] = recSys.content_based.scaler
        ranker_dict["output"]["content"] = content_based_output

    # Add outputs to recommended courses
    student_info.results["recommended_courses"] = add_outputs_to_recommended_courses(
        student_info.results["recommended_courses"],
        student_info.course_data,
        student_info.student_input,
        ranker_dict)

    # Calculate final score
    student_info.results["recommended_courses"] = calculate_final_score_TEMPORY(
        student_info.results["recommended_courses"],
        recSys.top_n)


def compute_warnings(recSys, student_info):
    # Predict the pass or fail for courses and give warning recommendations
    prediction_results = recSys.warning_model.predict(student_info.student_data,
                                                      student_info.course_data,
                                                      student_info.results["recommended_courses"].keys())
    for code in prediction_results:
        student_info.results["recommended_courses"][code]["warning"] = prediction_results[code]["warning"]
        student_info.results["recommended_courses"][code]["warning_recommendation"] = prediction_results[code][
            "warning_recommendation"]


def compute_explanations(recSys, student_id, course_code):
    collection = recSys.db["student_results"]
    student_info = collection.find({"_id": ObjectId(student_id)})[0]
    student_input = student_info["student_input"]
    course_result = student_info["results"]["recommended_courses"][course_code]
    course = student_info["course_data"][course_code]

    response = recSys.explanation.generate_explanation(student_input, course_code, course, course_result)

    collection_LLM = recSys.db["LLM_usage"]
    llm_usage_old = collection_LLM.find_one({"_id": student_info["student_id"]})

    # Keep track of LLM usage for each student
    if llm_usage_old is None:
        collection_LLM.insert_one({
            "_id": student_info["student_id"],
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        })
    else:
        completion_tokens = llm_usage_old["completion_tokens"] + response.usage.completion_tokens
        prompt_tokens = llm_usage_old["prompt_tokens"] + response.usage.prompt_tokens
        total_tokens = llm_usage_old["total_tokens"] + response.usage.total_tokens
        collection_LLM.update_one({"_id": student_info["student_id"]}, {"$set": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }})

    now = datetime.now()

    # Insert explanation into LLM_results
    recSys.db["LLM_results"].insert_one({
        "student_id": student_info["student_id"],
        "course_code": course_code,
        "student_input": student_info["student_input"],
        "course_keywords": course_result['keywords'],
        "explanation": json.loads(response.choices[0].message.content),
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens,
        "time": now.strftime("%d/%m/%Y %H:%M:%S")
    })

    return response.choices[0].message.content


def compute_timeline(RecSys, student_id):
    collection = RecSys.db['student_results']
    student_info = collection.find({'_id': ObjectId(student_id)})[0]
    course_data = student_info['results']['recommended_courses']
    return RecSys.planner.plan(course_data)
