from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from openai import OpenAI
from threading import Thread
from langchain.llms import HuggingFaceTextGenInference
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from rec_sys_uni.rec_systems.llm_explanation.Streaming import StreamingRecSys
import transformers
import time
import asyncio
import os


class LLM:
    def __init__(self, url: str, token: str, model_id: str, model_name: str, chat_gpt: bool = False, thread_mode: bool = False):
        self.model_id = model_id
        self.thread_mode = thread_mode
        self.chat_gpt = chat_gpt

        if chat_gpt:
            self.llm = OpenAI(api_key=token)
        else:
            self.llm = HuggingFaceTextGenInference(
                inference_server_url=url,
                max_new_tokens=1000,
                temperature=0.05,
                # top_k=10,
                # top_p=0.95,
                # typical_p=0.95,
                repetition_penalty=1.1,
                streaming=True,
                stop_sequences=["\n\n\n\n"],
            )
            # Authentication
            self.llm.client.headers = {"Authorization": f"Bearer {token}"}


        # # Load local FAISS database
        # db = load_local_db_FAISS(model_name)
        #
        # # RAG Pipeline
        # self.rag = RetrievalQA.from_chain_type(
        #     llm=self.llm, chain_type='stuff',
        #     retriever=db.as_retriever()
        # )


    def generate_in_thread(self, prompt_tokenized, course_code):
        result = self.llm(prompt_tokenized, callbacks=[StreamingRecSys(course_code)])
        # print("\n\n")
        # print(f"Course: {course_code}")
        # print(result)

    def generate_GPT_in_thread(self, prompt, course_code):
        response = self.llm.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=prompt,
            temperature=0.1,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=1.1,
            presence_penalty=0.95
        )
        # print("\n\n")
        # print(f"Course: {course_code}")
        # print(response.choices[0].message.content)

    def generate_explanations_sync(self, recSys, student_info):
        """
        Synchronously generate explanations for recommended courses.

        :param recSys: RecSys object
        :param student_info: StudentNode object
        """
        for course in student_info.results['sorted_recommended_courses']:
            # Generate prompt for each course
            prompt = generate_prompt_per_course(student_info.student_input, course, student_info.course_data[course['course_code']])
            # Add the prompt to the template
            final_prompt_tokenized = add_to_template(prompt, self.model_id)
            # Create a coroutine for the task
            # self.rag.run(final_prompt_tokenized, callbacks=[StreamingRecSys(course['course_code'])])
            print("\n")
            print(f"Course: {course['course_code']}")
            if self.thread_mode:
                Thread(target=self.generate_in_thread, args=(final_prompt_tokenized, course['course_code'])).start()
            else:
                self.llm(final_prompt_tokenized, callbacks=[StreamingRecSys(course['course_code'])])




    def generate_explanations_GPT(self, recSys, student_info):
        """
        Generate explanations with OpenAI for recommended courses.

        :param recSys: RecSys object
        :param student_info: StudentNode object
        """
        for course in student_info.results['sorted_recommended_courses']:
            # Generate prompt for each course
            prompt = generate_prompt_per_course(student_info.student_input, course, student_info.course_data[course['course_code']])
            # Add the prompt to the template
            final_prompt_tokenized = add_to_template(prompt, self.model_id, True)
            # Create a coroutine for the task
            # self.rag.run(final_prompt_tokenized, callbacks=[StreamingRecSys(course['course_code'])])
            print("\n")
            print(f"Course: {course['course_code']}")
            if self.thread_mode:
                Thread(target=self. generate_GPT_in_thread, args=(final_prompt_tokenized, course['course_code'])).start()
            else:
                response = self.llm.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=final_prompt_tokenized,
                    temperature=0.1,
                    max_tokens=4096,
                    top_p=0.95,
                    frequency_penalty=1.1,
                    presence_penalty=0.95
                )
                print(response.choices[0].message.content)
            time.sleep(2)


    def generate_explanation(self, recSys, student_info):
        """
        Generate explanations for recommended courses.
        """
        print("Hello from LLM")
        # Assuming generate_explanations_async is an async method
        if not self.chat_gpt:
            self.generate_explanations_sync(recSys, student_info)
        else:
            self.generate_explanations_GPT(recSys, student_info)


        print(f"Number of tasks: {len(student_info.results['sorted_recommended_courses'])}")


def generate_prompt_per_course(student_input, course, course_description):
    # Define the period of prompt
    text_promt = "\nCourse:"

    text_promt += (f"\n1. Course Title: {course['course_name']} ({course['course_code']})")

    text_promt += f"\n2. Course Description: {course_description}"

    text_promt += f"\n3. Keywords: "
    for key in course['keywords']:
        text_promt += f"{key}: {course['keywords'][key] / student_input['keywords'][key]}, "
    text_promt += "."  # Finish the line with dot

    # text_promt += f"\nStudent input: 'keywords': "
    # for key in student_input['keywords']:
    #     text_promt += f"{key}: {round(student_input['keywords'][key] * 100, 2)}, "
    # text_promt += "."  # Finish the line with dot

    text_promt += ("You should answer by the following template:\n"
                   "Course Title: \n"
                   "Description: \n"
                   "Explanation: \n")
    return text_promt


def add_to_template(text_promt, model_id, chat_gpt=False):
    if chat_gpt:
        template = get_prompt_template_per_course_GPT()
    else:
        template = get_prompt_template_per_course()
    # Add recommended courses to ask the LLM to generate explanation
    template.append(
        {
            "role": "user",
            "content": f'{text_promt}'
        }
    )

    # Tokenize the template for the LLM
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    # Return the tokenized template
    if chat_gpt:
        return template
    else:
        return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)


def create_knowledge_base(course_data, model_name):
    knowledge_base = []
    index = 0
    # Create knowledge base
    for code in course_data:
        title = course_data[code]['course_name']  # Title of the course
        desc = course_data[code]['description']  # Description of the course
        ilos = course_data[code]['ilos']  # ILOs of the course
        text = "Title of the course: " + title + ". "  # For each text add the title of the course
        for i in desc.split('.'):
            text_length = len(text)
            i_length = len(i)
            if text_length + i_length < 1000:  # Check if the text is less than 1000 characters
                text += i.strip() + '.'
            else:
                knowledge_base.append(text.strip())  # If the text is more than 1000 characters,
                # add it to the knowledge base
                text = "Title of the course: " + title + ". " + i.strip() + '.'  # Start a new text
        knowledge_base.append(text)

        # Add ILOs to the knowledge base with the title of the course
        text = f"\nThe {title} has the following ILOs: " + ", ".join(ilos).strip() + ". "
        knowledge_base.append(text)
        index += 1

    with (open(r'rec_sys_uni/datasets/data/LLM/knowledge_base.txt', 'w')) as fp:
        fp.write('\n'.join(knowledge_base))

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )

    # Load documents and split them
    documents = TextLoader("rec_sys_uni/datasets/data/LLM/knowledge_base.txt").load()
    # Initialize text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Split documents into chunks
    docs = text_splitter.split_documents(documents)

    # Create local vector database
    db = FAISS.from_documents(docs, embedding_model)
    # Save local vector database
    db.save_local("rec_sys_uni/datasets/data/LLM/db_FAISS")


def load_local_db_FAISS(model_name):
    """
    function: load local FAISS database for Retrieval Augmented Generation (RAG)
    """
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )
    # Load local vector database for RAG
    db = FAISS.load_local("rec_sys_uni/datasets/data/LLM/db_FAISS", embedding_model)
    return db


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
                    "you must leverage the information provided based on a student's specified interests.\n"

                    "You will be supplied with a course, which includes the following attributes:\n"
                    "1. Course Title.\n"
                    "2. Keywords: A dictionary consisting of relevant keywords associated with the course, along with "
                    "their values  "
                    "(e.g., values from 10 to 20 suggest a small correlation, from 30 to 60 a medium "
                    "correlation, and from 70 to 100 a large correlation). "
                    "Be aware that the provided values might not "
                    "always accurately represent the course content.\n"

                    # "In addition to the course information, you will receive inputs from students, which include:\n" 
                    # "keywords: A dictionary of keywords and their associated values that represent the student's areas of "
                    # "interest "
                    # "(e.g., values from 10 to 20 suggest a small interest, from 30 to 60 a neutral interest, "
                    # "and from 70 to 100 a high interest)."
                    "\n"

                    "Based on the course and the student input, generate a recommendation for the specified course. "
                    "Ensure to include the following details:\n"

                    "Course Title: State the full title of the course.\n"
                    "Description: Offer a clear and detailed description of the course, including the topics that will be "
                    "covered and the skills that will be taught.\n"
                    "Explanation: Provide a thoughtful analysis explaining how this course aligns with the student's "
                    "expressed interests.\n"

                    "Remember, your goal is to match courses to a student's interests, keywords as accurately as "
                    "possible, which will require you to interpret the relevance of course in relation to the interests "
                    "specified by the student.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Basic Mathematical Tools (SCI1010)\n"
                    "2. Keywords: math: 54.23, artificial intelligence: 29.53, data analyze: 30.21, statistics: 30.64.\n\n"
                    # "Student input:\n" 
                    # "'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template:\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Basic Mathematical Tools (SCI1010)\n"
                    "Description: This course provides foundational knowledge in mathematics, covering analysis, algebra, "
                    "and probability. It starts with a refresher of high school topics and progresses to advanced subjects "
                    "preparing students for quantitative courses. Skills in solving equations, differentiation, and function "
                    "analysis are developed.\n"
                    "Explanation:\n"
                    "- Math: The student's interest in math correlates with the course's intensive focus on refreshing and "
                    "building upon mathematical principles and techniques.\n"
                    "- Artificial Intelligence: While the course does not directly deal with artificial intelligence, the "
                    "mathematical foundations taught are essential for understanding AI algorithms and models. Mathematical "
                    "tools, especially in probability and statistics, are the backbone of machine learning, a subset of AI.\n"
                    "- Data Analysis: Data analysis relies heavily on statistics and mathematical models to interpret data. "
                    "This course provides fundamental skills that will aid in the comprehension and application of data analysis "
                    "techniques.\n"
                    "- Statistics: Given the student's interest in statistics, the course's inclusion of probability theory, "
                    "which is a bedrock of statistical analysis, makes it a good match.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Foundations of Cognitive Psychology (SSC2062)\n"
                    "2. Keywords: math: 'math': 26.06, artificial intelligence: 31.64, data analyze: 20.14, statistics: 18.72\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Foundations of Cognitive Psychology (SSC2062)\n"
                    "Description: The Foundations of Cognitive Psychology course examines the human mind from an "
                    "information processing perspective, tracing its history in psychology and its application in cognitive "
                    "neuroscience. It covers how thought is structured and studied, including topics like perception, "
                    "attention, language, and emotion, using a problem-based learning approach to apply theory to "
                    "practice in psychology.\n"
                    "Explanation:\n"
                    "- Math: The Foundations of Cognitive Psychology does involve some mathematical concepts, "
                    "likely in statistical methods and data analysis\n"
                    "- Artificial Intelligence: Cognitive psychology informs many AI concepts, especially those related to "
                    "how machines can mimic human thought processes. The correlation is present because cognitive psychology's "
                    "principles underpin many AI models, even if it's not the course's main focus.\n"
                    "- Data Analysis: Cognitive psychology often involves the analysis of experimental data to understand "
                    "cognitive processes.\n"
                    "- Statistics: The Foundations of Cognitive Psychology does involve some statistical concepts\n"
                )
        },
        # {
        #     "role": "user",
        #     "content":
        #         (
        #             "Course:\n"
        #             "1. Course Title: Optimization (SCI3003)\n"
        #             "2. Keywords: math: 34.0, artificial intelligence: 44.34, data analyze: 25.01, statistics: 20.34.\n\n"
        #             # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
        #             "You should answer by the following template::\n"
        #             "Course Title: \n"
        #             "Description: \n"
        #             "Explanation: \n"
        #         )
        # },
        # {
        #     "role": "assistant",
        #     "content":
        #         (
        #             "Course Title: Optimization (SCI3003)\n"
        #             "Description: Optimization focuses on improving performance by using mathematical models. It covers "
        #             "techniques like gradient methods and Newton's method, exploring both unconstrained and constrained "
        #             "optimization, including linear programming and the Lagrange multiplier method. Practical applications "
        #             "are demonstrated through various examples.\n"
        #             "Explanation:\n"
        #             "- Math: Given the strong foundation in mathematics that this course provides, it matches the student's "
        #             "interest. Understanding optimization requires a deep grasp of mathematical concepts which the student "
        #             "has shown a significant interest in.\n"
        #             "- Artificial Intelligence: Optimization is a key component in artificial intelligence for improving algorithms "
        #             "and decision-making processes. The student's interest is reflected in the course's strong connection to AI, "
        #             "particularly in techniques like gradient methods and iterative solutions which are used in machine learning.\n"
        #             "- Data Analysis: The course's focus on fitting models to data and solving least squares problems is highly "
        #             "relevant to data analysis, a field reliant on optimization for predictive modeling and interpretation.\n"
        #             "- Statistics: Statistical models often include optimization problems, such as least squares for regression "
        #             "analysis, that are covered in this course.\n"
        #         )
        # },
        # {
        #     "role": "user",
        #     "content":
        #         (
        #             "Course:\n"
        #             "1. Course Title: Statistics I (SSC2061)\n"
        #             "2. Keywords: math: 33.93, artificial intelligence: 21.72, data analyze: 35.55, statistics: 71.36.\n\n"
        #             # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
        #             "You should answer by the following template::\n"
        #             "Course Title: \n"
        #             "Description: \n"
        #             "Explanation: \n"
        #         )
        # },
        # {
        #     "role": "assistant",
        #     "content":
        #         (
        #             "Course Title: Statistics I (SSC2061)\n"
        #             "Description: An introductory course to statistics tailored for social and life sciences, covering descriptive "
        #             "and inferential statistics. The course aims to make statistics approachable through practical assignments, "
        #             "enabling students to visualize data, test hypotheses, and understand associations between different data types.\n"
        #             "Explanation:\n"
        #             "- Math: The mathematical underpinnings of statistical theory are integral to this course. Given the student's "
        #             "interest in math, there's a clear relevance and applicability for the student wishing to apply mathematical "
        #             "concepts within the context of statistics.\n"
        #             "- Artificial Intelligence: While AI is not the primary focus of this course, the foundational knowledge in "
        #             "statistics is invaluable for understanding machine learning, a cornerstone of AI. The course includes relevant "
        #             "content that would benefit a student interested in AI, particularly in the application of statistical methods "
        #             "to inform and evaluate AI models.\n"
        #             "- Data Analysis: With a focus on descriptive and inferential statistics, the course covers the essentials of "
        #             "data analysis, making sense of data which is a significant interest for the student.\n"
        #             "- Statistics: The student's keen interest in statistics is directly met by this course. This indicates a major "
        #             "focus of the course on statistical principles and methodologies, perfectly aligning with what the student is "
        #             "looking to learn.\n"
        #         )
        # },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Contemporary World History (SCI1010)\n"
                    "2. Keywords: math: 23.83, artificial intelligence: 20.72, data analyze: 29.62, statistics: 32.56.\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Contemporary World History (SCI1010)\n"
                    "Description: This course explores significant historical events and their impacts on the modern world. It employs "
                    "a philosophical lens to study history, with a focus on state and nation concepts, economic trends, and global power "
                    "dynamics. Case studies from various periods are used to link the past with current global situations.\n"
                    "Explanation:\n"
                    "- Math: This history course likely does not delve into these areas, focusing instead on the narrative, analysis, "
                    "and interpretation of historical events.\n"
                    "- Artificial Intelligence: AI typically involves the use of computer science and mathematics to create systems "
                    "capable of performing tasks that usually require human intelligence. Contemporary World History does not address "
                    "these computational or technical aspects and instead focuses on the socio-political development of human societies.\n"
                    "- Data Analysis: Although historical studies can involve the analysis of data from past events, this course likely "
                    "focuses more on qualitative analysis and theoretical perspectives rather than the quantitative data analysis the "
                    "student is interested in.\n"
                    "- Statistics: It won't likely cover statistical theories or tools such as hypothesis testing, probability distributions, "
                    "or data visualization techniques that a student interested in statistics would be looking to study.\n"
                )
        }
    ]


def get_prompt_template_per_course():
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
                    "you must leverage the information provided based on a student's specified interests.\n"

                    "You will be supplied with a course, which includes the following attributes:\n"
                    "1. Course Title.\n"
                    "2. Keywords: A dictionary consisting of relevant keywords associated with the course, along with "
                    "their values  "
                    "(e.g., values from 10 to 20 suggest a small correlation, from 30 to 60 a medium "
                    "correlation, and from 70 to 100 a large correlation). "
                    "Be aware that the provided values might not "
                    "always accurately represent the course content.\n"

                    # "In addition to the course information, you will receive inputs from students, which include:\n" 
                    # "keywords: A dictionary of keywords and their associated values that represent the student's areas of "
                    # "interest "
                    # "(e.g., values from 10 to 20 suggest a small interest, from 30 to 60 a neutral interest, "
                    # "and from 70 to 100 a high interest)."
                    "\n"

                    "Based on the course and the student input, generate a recommendation for the specified course. "
                    "Ensure to include the following details:\n"

                    "Course Title: State the full title of the course.\n"
                    "Description: Offer a clear and detailed description of the course, including the topics that will be "
                    "covered and the skills that will be taught.\n"
                    "Explanation: Provide a thoughtful analysis explaining how this course aligns with the student's "
                    "expressed interests.\n"

                    "Remember, your goal is to match courses to a student's interests, keywords as accurately as "
                    "possible, which will require you to interpret the relevance of course in relation to the interests "
                    "specified by the student.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Basic Mathematical Tools (SCI1010)\n"
                    "2. Keywords: math: 54.23, artificial intelligence: 29.53, data analyze: 30.21, statistics: 30.64.\n\n"
                    # "Student input:\n" 
                    # "'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template:\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Basic Mathematical Tools (SCI1010)\n"
                    "Description: This course provides foundational knowledge in mathematics, covering analysis, algebra, "
                    "and probability. It starts with a refresher of high school topics and progresses to advanced subjects "
                    "preparing students for quantitative courses. Skills in solving equations, differentiation, and function "
                    "analysis are developed.\n"
                    "Explanation:\n"
                    "- Math: The student's interest in math correlates with the course's intensive focus on refreshing and "
                    "building upon mathematical principles and techniques.\n"
                    "- Artificial Intelligence: While the course does not directly deal with artificial intelligence, the "
                    "mathematical foundations taught are essential for understanding AI algorithms and models. Mathematical "
                    "tools, especially in probability and statistics, are the backbone of machine learning, a subset of AI.\n"
                    "- Data Analysis: Data analysis relies heavily on statistics and mathematical models to interpret data. "
                    "This course provides fundamental skills that will aid in the comprehension and application of data analysis "
                    "techniques.\n"
                    "- Statistics: Given the student's interest in statistics, the course's inclusion of probability theory, "
                    "which is a bedrock of statistical analysis, makes it a good match.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Foundations of Cognitive Psychology (SSC2062)\n"
                    "2. Keywords: math: 'math': 26.06, artificial intelligence: 31.64, data analyze: 20.14, statistics: 18.72\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Foundations of Cognitive Psychology (SSC2062)\n"
                    "Description: The Foundations of Cognitive Psychology course examines the human mind from an "
                    "information processing perspective, tracing its history in psychology and its application in cognitive "
                    "neuroscience. It covers how thought is structured and studied, including topics like perception, "
                    "attention, language, and emotion, using a problem-based learning approach to apply theory to "
                    "practice in psychology.\n"
                    "Explanation:\n"
                    "- Math: The Foundations of Cognitive Psychology does involve some mathematical concepts, "
                    "likely in statistical methods and data analysis\n"
                    "- Artificial Intelligence: Cognitive psychology informs many AI concepts, especially those related to "
                    "how machines can mimic human thought processes. The correlation is present because cognitive psychology's "
                    "principles underpin many AI models, even if it's not the course's main focus.\n"
                    "- Data Analysis: Cognitive psychology often involves the analysis of experimental data to understand "
                    "cognitive processes.\n"
                    "- Statistics: The Foundations of Cognitive Psychology does involve some statistical concepts\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Optimization (SCI3003)\n"
                    "2. Keywords: math: 34.0, artificial intelligence: 44.34, data analyze: 25.01, statistics: 20.34.\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Optimization (SCI3003)\n"
                    "Description: Optimization focuses on improving performance by using mathematical models. It covers "
                    "techniques like gradient methods and Newton's method, exploring both unconstrained and constrained "
                    "optimization, including linear programming and the Lagrange multiplier method. Practical applications "
                    "are demonstrated through various examples.\n"
                    "Explanation:\n"
                    "- Math: Given the strong foundation in mathematics that this course provides, it matches the student's "
                    "interest. Understanding optimization requires a deep grasp of mathematical concepts which the student "
                    "has shown a significant interest in.\n"
                    "- Artificial Intelligence: Optimization is a key component in artificial intelligence for improving algorithms "
                    "and decision-making processes. The student's interest is reflected in the course's strong connection to AI, "
                    "particularly in techniques like gradient methods and iterative solutions which are used in machine learning.\n"
                    "- Data Analysis: The course's focus on fitting models to data and solving least squares problems is highly "
                    "relevant to data analysis, a field reliant on optimization for predictive modeling and interpretation.\n"
                    "- Statistics: Statistical models often include optimization problems, such as least squares for regression "
                    "analysis, that are covered in this course.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Statistics I (SSC2061)\n"
                    "2. Keywords: math: 33.93, artificial intelligence: 21.72, data analyze: 35.55, statistics: 71.36.\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Statistics I (SSC2061)\n"
                    "Description: An introductory course to statistics tailored for social and life sciences, covering descriptive "
                    "and inferential statistics. The course aims to make statistics approachable through practical assignments, "
                    "enabling students to visualize data, test hypotheses, and understand associations between different data types.\n"
                    "Explanation:\n"
                    "- Math: The mathematical underpinnings of statistical theory are integral to this course. Given the student's "
                    "interest in math, there's a clear relevance and applicability for the student wishing to apply mathematical "
                    "concepts within the context of statistics.\n"
                    "- Artificial Intelligence: While AI is not the primary focus of this course, the foundational knowledge in "
                    "statistics is invaluable for understanding machine learning, a cornerstone of AI. The course includes relevant "
                    "content that would benefit a student interested in AI, particularly in the application of statistical methods "
                    "to inform and evaluate AI models.\n"
                    "- Data Analysis: With a focus on descriptive and inferential statistics, the course covers the essentials of "
                    "data analysis, making sense of data which is a significant interest for the student.\n"
                    "- Statistics: The student's keen interest in statistics is directly met by this course. This indicates a major "
                    "focus of the course on statistical principles and methodologies, perfectly aligning with what the student is "
                    "looking to learn.\n"
                )
        },
        {
            "role": "user",
            "content":
                (
                    "Course:\n"
                    "1. Course Title: Contemporary World History (SCI1010)\n"
                    "2. Keywords: math: 23.83, artificial intelligence: 20.72, data analyze: 29.62, statistics: 32.56.\n\n"
                    # "Student input: 'keywords': math: 50, artificial intelligence: 50, data analyze: 50, statistics: 50.\n"
                    "You should answer by the following template::\n"
                    "Course Title: \n"
                    "Description: \n"
                    "Explanation: \n"
                )
        },
        {
            "role": "assistant",
            "content":
                (
                    "Course Title: Contemporary World History (SCI1010)\n"
                    "Description: This course explores significant historical events and their impacts on the modern world. It employs "
                    "a philosophical lens to study history, with a focus on state and nation concepts, economic trends, and global power "
                    "dynamics. Case studies from various periods are used to link the past with current global situations.\n"
                    "Explanation:\n"
                    "- Math: This history course likely does not delve into these areas, focusing instead on the narrative, analysis, "
                    "and interpretation of historical events.\n"
                    "- Artificial Intelligence: AI typically involves the use of computer science and mathematics to create systems "
                    "capable of performing tasks that usually require human intelligence. Contemporary World History does not address "
                    "these computational or technical aspects and instead focuses on the socio-political development of human societies.\n"
                    "- Data Analysis: Although historical studies can involve the analysis of data from past events, this course likely "
                    "focuses more on qualitative analysis and theoretical perspectives rather than the quantitative data analysis the "
                    "student is interested in.\n"
                    "- Statistics: It won't likely cover statistical theories or tools such as hypothesis testing, probability distributions, "
                    "or data visualization techniques that a student interested in statistics would be looking to study.\n"
                )
        }
    ]

#%%
