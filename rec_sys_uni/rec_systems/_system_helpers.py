from sklearn.preprocessing import MinMaxScaler


def add_outputs_to_recommended_courses(recommended_courses, course_data, student_input, ranker_dict):
    """
    Add output to recommended courses
    :param recommended_courses: recommended courses
    :param course_data: course data
    :param student_input: student input
    :param ranker_dict: ranker dictionary
    :return: recommended courses with outputs and total output score for each type
    """
    recommended_courses = recommended_courses.copy()

    # Scale outputs
    for type_ in ranker_dict['output']:
        scaler = ranker_dict['scaler'][type_]
        if scaler == 'MinMax':
            ranker_dict['output'][type_] = apply_MinMaxScaler(ranker_dict['output'][type_])

    # Add outputs to recommended courses
    for index, code in enumerate(course_data):
        for type_ in ranker_dict['output']:
            recommended_courses[code][type_] = ranker_dict['output'][type_][code]

            score_alg = ranker_dict['score_alg'][type_]
            # Calculate total score for each type
            if score_alg == 'sum':
                recommended_courses[code][type_ + "_score"] = sum_(ranker_dict['output'][type_][code],
                                                                   student_input,
                                                                   type_)
    # Calculate total score for each type (we need all courses for that calculation)
    for type_ in ranker_dict['output']:
        score_alg = ranker_dict['score_alg'][type_]
        if score_alg == 'rrf':
            results = apply_reciprocal_rank_fusion(ranker_dict['output'][type_], student_input, type_)
            for code in results:
                recommended_courses[code][type_ + "_score"] = results[code]


    return recommended_courses


def apply_MinMaxScaler(outputs):
    """
    Apply MinMaxScaler to outputs
    :param outputs: outputs
    :return: outputs with scaled values
    """
    outputs = outputs.copy()
    matrix = []
    for output in outputs:
        tmp_matrix = []
        for i in outputs[output]:
            tmp_matrix.append(outputs[output][i])
        matrix.append(tmp_matrix)
    matrix = MinMaxScaler().fit_transform(matrix)
    for j, output in enumerate(outputs):
        for i, k in enumerate(outputs[output]):
            outputs[output][k] = matrix[j][i]
    return outputs


def sum_(outputs, stundet_input, type_):
    """
    Calculate sum of outputs
    :param outputs: outputs
    :param stundet_input: student input
    :param type_: type of output
    :return: sum of outputs
    """
    if type_ in ['keywords', 'blooms']:
        score = 0
        for i in outputs:
            score += outputs[i] * stundet_input[type_][i]
        return score
    else:
        return sum(outputs.values())


def apply_reciprocal_rank_fusion(outputs, student_input, type_, k=60):
    """
    Apply Reciprocal Rank Fusion algorithm to outputs
    :param outputs: outputs
    :param student_input: student input
    :param type_: type of output
    :param k: constant
    :return: outputs with RRF applied
    """
    if type_ in ['keywords', 'blooms']:
        key_outputs = student_input[type_]
    else:
        keys = list(outputs.keys())
        key_outputs = outputs[keys[0]].keys()

    results = {}
    for i in outputs:
        results[i] = 0

    for key in key_outputs:
        # Sort by keyword score
        sorted_keyword_list = sorted(outputs.items(),
                                     key=lambda x: x[1][key], reverse=True)

        for rank in range(len(sorted_keyword_list)):
            # Reciprocal Rank Fusion algorithm
            if type_ in ['keywords', 'blooms']:
                results[sorted_keyword_list[rank][0]] += key_outputs[key] / (rank + k)
            else:
                results[sorted_keyword_list[rank][0]] += 1 / (rank + k)

    return results


def calculate_final_score_TEMPORY(recommended_courses, top_n): # TODO: SHOULD BE CHANGED ON SMTH BETTER
    # Calculate final score
    recommended_courses = recommended_courses.copy()

    for code in recommended_courses:
        recommended_courses[code]['score'] = recommended_courses[code]['keywords_score']

    # TEST (GOOD RESULTS)
    # results_score = apply_reciprocal_rank_fusion_final(recommended_courses,
    #                                           ['keywords_score', 'content_score'],
    #                                                    k=60)
    # for code in results_score:
    #     recommended_courses[code]['score'] = results_score[code]


    top_scores = sorted(recommended_courses.items(),
                        key=lambda x: x[1]['score'], reverse=True)[:top_n]
    for i in top_scores:
        recommended_courses[i[0]]['score'] += recommended_courses[i[0]]['blooms_score']

    return recommended_courses

def apply_reciprocal_rank_fusion_final(results, measure, k=60):
    """
    Apply Reciprocal Rank Fusion algorithm to final results
    :param results: results
    :param measure: measure
    :param k: constant
    :return: results with RRF applied
    """
    results_score = {}
    for i in results:
        results_score[i] = 0
    for key in measure:
        # Sort by keyword score
        sorted_keyword_list = sorted(results.items(),
                                     key=lambda x: x[1][key], reverse=True)

        for rank in range(len(sorted_keyword_list)):
            # Reciprocal Rank Fusion algorithm
            results_score[sorted_keyword_list[rank][0]] += 1 / (rank + k)

    return results_score
