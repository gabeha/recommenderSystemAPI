from openai import OpenAI


class LLM:
    def __init__(self,
                 token: str
                 ):
        self.llm = OpenAI(api_key=token)

    def generate_explanations_GPT(self, student_input, course_code, course, course_results):
        """
        Generate explanations with OpenAI for recommended courses.
        """
        # Generate prompt for each course
        prompt = generate_prompt_per_course(student_input, course_code, course, course_results)
        # Add the prompt to the template
        final_prompt_tokenized = add_to_template(prompt)
        # Generate explanation
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=final_prompt_tokenized,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response

    def generate_explanation(self, student_input, course_code, course, course_results):
        """
        Generate explanations for recommended courses.
        """
        return self.generate_explanations_GPT(student_input, course_code, course, course_results)


def generate_prompt_per_course(student_input, course_code, course, course_results):
    # Define the period of prompt
    text_promt = "student request:\n"

    text_promt += (f"1. 'course_code': '{course_code}'\n")

    text_promt += (f"2. 'course_title': '{course['course_name']}'\n")

    text_promt += f"3. course description: '{course['description']}'\n"

    text_promt += f"4. intended learning outcomes:\n"

    for i in course['ilos']:
        text_promt += f"- {i}\n"

    text_promt += f"5. keyword similarities with course description:\n"
    for key in course_results['keywords']:
        text_promt += f"- {key}: {round(course_results['keywords'][key], 2)}\n"

    text_promt += f"6. student interest in keywords:\n"
    for key in student_input['keywords']:
        text_promt += f"- {key}: {student_input['keywords'][key]}\n"

    text_promt += ("\nYou should answer in the JSON format:\n"
                   "{\n"
                   "'course_code': ... ,\n"
                   "'course_title': ... ,\n"
                   "'summary_description': ... ,\n"
                   "'keywords_explanation': {\n"
                   "'keyword1': ... ,\n"
                   "'keyword2': ... ,\n"
                   "...\n"
                   "}\n"
                   "}")
    return text_promt


def add_to_template(text_promt):
    template = get_prompt_template_per_course_GPT()
    # Add recommended courses to ask the LLM to generate explanation
    template.append(
        {
            "role": "user",
            "content": f'{text_promt}'
        }
    )

    # Return template
    return template


def get_prompt_template_per_course_GPT():
    """
    function: get the prompt template for LLM
    """
    return [
        {
            "role": "system",
            "content":
                (
                    "You are tasked with assisting students in choosing academic courses that align closely with their "
                    "interests and educational aspirations. To facilitate personalized course recommendations, "
                    "you must leverage the information provided based on a student's specified keywords.\n\n"

                    "You will be supplied with a student request, which includes the following attributes:\n"
                    "1. 'course_code': ...\n"
                    "2. 'course_title': ...\n"
                    "3. course description: ...\n"
                    "4. intended learning outcomes: ...\n"
                    "5. similarities between keywords and course description: ... \n"
                    "Similarity values are provided for each keyword, which indicate the cosine correlation between "
                    "the keyword and the course description. The values are in a scale of -1 to 1."
                    " Be aware that the provided values might not always accurately represent the actual similarity.\n"
                    "6. student interest in keywords: ...\n"
                    "The student's interest in each keyword is provided as a value from 0 to 1, with 0 indicating no "
                    "interest, 0.25 indicating low interest, 0.5 indicating neutral interest, 0.75 indicating "
                    "high interest, 1 indicating very-high interest.\n\n"


                    "Based on the course overview, generate a recommendation for the specified course. "
                    "Ensure generate the answer in the JSON format:\n"
                    "{\n"
                    "'course_code': ... ,\n"
                    "'course_title': ... ,\n"
                    "'summary_description': ... , (Offer a clear and detailed description of the course, including the topics that will be "
                    "covered and the skills that will be taught.)\n"
                    "'keywords_explanation': (Provide a thoughtful analysis explaining how this course aligns with keywords.)\n"
                    "{\n"
                    "'keyword1': ... ,\n"
                    "'keyword2': ... ,\n"
                    "...\n"
                    "}\n"
                    "}\n\n"

                    "Remember, your goal is to provide explanation to a student request as accurately as "
                    "possible, which will require you to interpret the relevance of course in relation to the interests "
                    "specified by the student's keywords. If you see no relevance, explain why it is not relevant. Do not try to make connections\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "student request:\n"
                    "1. 'course_code': 'SCI1010'\n"
                    "2. 'course_title': 'Basic Mathematical Tools'\n"
                    "3. course description: ...\n"
                    "4. intended learning outcomes: ...\n"
                    "5. keyword similarities with course description:\n"
                    "- artificial intelligence: 0.23\n"
                    "- math: 1.0\n"
                    "- data analyze: 0.22\n"
                    "- statistics: 0.31\n"
                    "6. student interest in keywords:\n"
                    "- artificial intelligence: 0.6\n"
                    "- math: 1.0\n"
                    "- data analyze: 0.4\n"
                    "- statistics: 0.8\n\n"
                    "You should answer in the JSON format:\n"
                    "{\n"
                    "'course_code': ... ,\n"
                    "'course_title': ... ,\n"
                    "'summary_description': ... ,\n"
                    "'keywords_explanation': {\n"
                    "'keyword1': ... ,\n"
                    "'keyword2': ... ,\n"
                    "...\n"
                    "}\n"
                    "}"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "{\n"
                    "'course_code': 'SCI1010',\n"
                    "'course_title': 'Basic Mathematical Tools',\n"
                    "'summary_description': 'Basic Mathematical Tools is tailored for students who need to strengthen their mathematical foundation, particularly those with limited pre-university mathematical background. It begins with a three-week review of secondary school mathematics and progressively covers more advanced topics. The course encompasses a broad range of mathematical disciplines including analysis, algebra, probability theory, techniques for differentiation, function analysis, and geometry. It aims to equip students with essential mathematical skills, analytical reasoning, and a comprehensive understanding of mathematics' applicability across various fields.',\n"
                    "'keywords_explanation': {\n"
                    "'artificial intelligence': 'The relevance of artificial intelligence in this course is moderate (keyword similarity: 0.23). While not a direct focus, the mathematical concepts taught are foundational to understanding and working with AI technologies, aligning with the notable student interest in AI (student interest: 0.6).',\n"
                    "'math': 'Mathematics is the central theme of this course (keyword similarity: 1.0), covering a wide range of mathematical topics and techniques. This aligns perfectly with the high interest of students in math (student interest: 1.0).',\n"
                    "'data analyze': 'Data analysis has some relevance (keyword similarity: 0.22), as the course covers mathematical foundations that are essential in analyzing and interpreting data, corresponding with the moderate student interest in data analysis (student interest: 0.4).',\n"
                    "'statistics': 'Statistics is somewhat relevant (keyword similarity: 0.31) to the course, particularly through topics like probability theory. This aligns with the students' strong interest in statistics (student interest: 0.8).'\n"
                    "}\n"
                    "}"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "student request:\n"
                    "1. 'course_code': 'SCI2033'\n"
                    "2. 'course_title': 'Data Mining'\n"
                    "3. course description: ...\n"
                    "4. intended learning outcomes: ...\n"
                    "5. keyword similarities with course description:\n"
                    "- criminal law: 0.06\n"
                    "- law: 0.04\n"
                    "- analyze: 0.26\n"
                    "- human rights: 0.0\n"
                    "- justice: 0.19\n"
                    "- law enforcement: 0.23\n"
                    "- policy: 0.0\n"
                    "- society: 0.03\n"
                    "6. student interest in keywords:\n"
                    "- criminal law: 0.6\n"
                    "- law: 1.0\n"
                    "- analyze: 0.4\n"
                    "- human rights: 0.8\n"
                    "- justice: 0.8\n"
                    "- law enforcement: 0.8\n"
                    "- policy: 0.5\n"
                    "- society: 0.3\n\n"
                    "You should answer in the JSON format:\n"
                    "{\n"
                    "'course_code': ... ,\n"
                    "'course_title': ... ,\n"
                    "'summary_description': ... ,\n"
                    "'keywords_explanation': {\n"
                    "'keyword1': ... ,\n"
                    "'keyword2': ... ,\n"
                    "...\n"
                    "}\n"
                    "}"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "{\n"
                    "'course_code': 'SCI2033',\n"
                    "'course_title': 'Data Mining',\n"
                    "'summary_description': Data Mining is a comprehensive course focused on the field of extracting insightful information from large datasets. It integrates elements of artificial intelligence, machine learning, and statistics. The course is structured to provide practical experience with a variety of data mining techniques and tools, emphasizing their application in real-world scenarios. Key topics include Data Mining and Knowledge Discovery, Data Preparation, Decision-Tree Induction, Rule Induction, Bayesian Learning, Ensemble Techniques, Clustering, Association Rules, and interpreting data mining results. This course aims to equip students with the skills necessary for applying data mining techniques in research and business contexts.',\n"
                    "'keywords_explanation': {\n"
                    "'criminal law': 'There is no significant relevance of this course to criminal law (keyword similarity: 0.06). The course content does not cover legal principles, case law, or criminal justice systems.',\n"
                    "'law': 'The course does not directly relate to law (keyword similarity: 0.04). It does not involve legal theory, legislation, or legal practices.',\n"
                    "'analyze': 'Analysis is a fundamental part of this course (keyword similarity: 0.26). It focuses on analyzing large datasets to extract meaningful information, which is central to the field of data mining.',\n"
                    "'human rights': 'This course does not cover human rights (keyword similarity: 0.0). It does not delve into human rights law, ethical considerations, or related societal issues.',\n"
                    "'justice': 'There is no direct relevance to justice (keyword similarity: 0.19). The course content does not address the legal or philosophical aspects of justice.',\n"
                    "'law enforcement': 'The course does not specifically relate to law enforcement (keyword similarity: 0.23). It does not focus on policing strategies, criminal investigations, or legal enforcement mechanisms.',\n"
                    "'policy': 'The course does not have a direct connection to policy making (keyword similarity: 0.0). It does not explore policy development, analysis, or implementation strategies.',\n"
                    "'society': 'There is no direct relevance of this course to societal issues (keyword similarity: 0.03). It does not engage with social structures, societal challenges, or cultural studies.'\n"
                    "}\n"
                    "}"

                )
        }
    ]

# %%
