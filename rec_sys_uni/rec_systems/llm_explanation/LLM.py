from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import RetrievalQA
from rec_sys_uni.rec_systems.llm_explanation.Streaming import StreamingRecSys
from rec_sys_uni._helpers_rec_sys import sort_by_periods
import transformers


class LLM:
    def __init__(self, url: str, token: str, model_id: str, model_name: str, per_periods: bool = False):
        self.model_id = model_id
        self.per_periods = per_periods
        llm = HuggingFaceTextGenInference(
            inference_server_url=url,
            max_new_tokens=14000,
            temperature=0.1,
            # top_k=10,
            # top_p=0.95,
            # typical_p=0.95,
            repetition_penalty=1.1,
            streaming=True,
        )
        # Authentication
        llm.client.headers = {"Authorization": f"Bearer {token}"}
        # Load local FAISS database
        db = load_local_db_FAISS(model_name)

        # RAG Pipeline
        self.rag = RetrievalQA.from_chain_type(
            llm=llm, chain_type='stuff',
            retriever=db.as_retriever()
        )

    def generate_explanation(self, recSys, student_info):
        """
        parameters: recSys : RecSys object
                    student_info : StudentNode object
                    top_n : int
        function: generate explanation using LLM for the recommended courses
        """
        # Sort by periods
        sort_by_periods(recSys, student_info, max=recSys.top_n, include_keywords=True, include_blooms=False)
        courses = student_info.results['structured_recommendation']

        if self.per_periods: # Full explanation per period
            # Generate prompt
            text_promt_1 = generate_prompt_per_period(courses['period_1'], "1", self.model_id)
            text_promt_2 = generate_prompt_per_period(courses['period_2'], "2", self.model_id)

            # Generate explanation
            self.rag(text_promt_1, callbacks=[StreamingRecSys()])
            self.rag(text_promt_2, callbacks=[StreamingRecSys()])
        else: # Full explanation per course
            for i in courses['period_1']:
                text_promt = generate_prompt_per_course(i, "1", self.model_id)
                self.rag(text_promt, callbacks=[StreamingRecSys()])

            for i in courses['period_2']:
                text_promt = generate_prompt_per_course(i, "2", self.model_id)
                self.rag(text_promt, callbacks=[StreamingRecSys()])


def generate_prompt_per_period(period, num, model_id):
    # Define the period of prompt
    text_promt = f"Period {num}:"
    count = 1
    for i in period:
        text = f"\n{count}. "  # Place in the recommended list
        for j in i:
            if j != 'keywords':  # If the key is not keywords, then it is course_code or course_name
                text += f"{j}: {i[j]}. "
            else:  # If the key is keywords, then it is a dictionary of keywords
                text += f"{j}: "
                for k in i[j]:
                    text += f"{k}: {i[j][k]}, "  # Add each keyword and its value to the text
        text_promt += text + "."  # Finish the line with dot
        count += 1
    # Get the prompt template
    template = get_prompty_template_per_period()
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)

def generate_prompt_per_course(course, num, model_id):
    # Define the period of prompt
    text_promt = "Course:"
    for i in course:
        if i == 'course_code':
            text_promt += (f"\n1. Course code: {course[i]}")
        elif i == 'course_name':
            text_promt += (f"\n2. Course name: {course[i]}")
        elif i == 'keywords':
            text_promt += f"\n3. Keywords: "
            for k in course[i]:
                text_promt += f"{k}: {course[i][k]}, "  # Add each keyword and its value to the text
        text_promt += "."  # Finish the line with dot
    # Get the prompt template
    template = get_prompty_template_per_course()
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


def get_prompty_template_per_period():
    """
    function: get the prompt template for LLM
    """
    return [
        {
            "role": "system",
            "content":
                """
                You are tasked with assisting students in choosing academic courses that align closely with their interests and educational aspirations. To facilitate personalized course recommendations, you must leverage the information provided based on a student's specified interests.
                
                You will be supplied with a recommended list of courses for a specific period (e.g., Period 1, Period 2), which includes the following attributes:
                
                "course_code": The unique identifier for the course.
                "course_name": The official title of the course.
                "keywords": A dictionary consisting of relevant keywords associated with the course, along with their values. Be aware that the provided values might not always accurately represent the course content.
                In addition to the course information, you will receive inputs from students, which include:
                
                "keywords": A dictionary of keywords and their associated values that represent the student's areas of interest.
                Based on the recommended course list and the student input, generate a set of course recommendations for the specified period. For each course you recommend, ensure to include the following details:
                
                Course Name: State the full title of the course.
                Description: Offer a clear and detailed description of the course, including the topics that will be covered and the skills that will be taught.
                Explanation: Provide a thoughtful analysis explaining how this course aligns with the student's expressed interests.
                
                Remember, your goal is to match courses to a student's interests, keywords as accurately as possible, which will require you to interpret the relevance of course in relation to the interests specified by the student.
                """
        },
        {
            "role": "user",
            "content":
                """
                Period 1:
                1. course_code: SCI1010. course_name: Basic Mathematical Tools. keywords: math: 0.5414, statistics: 0.3019, data analyze: 0.2977, artificial intelligence: 0.2915, .
                2. course_code: SCI3003. course_name: Optimization. keywords: artificial intelligence: 0.4353, math: 0.3381, data analyze: 0.2453, statistics: 0.1996, .
                3. course_code: SSC2061. course_name: Statistics I. keywords: statistics: 0.7102, data analyze: 0.347, math: 0.3343, artificial intelligence: 0.2078, .
                4. course_code: COR1003. course_name: Contemporary World History. keywords: math: 0.229, artificial intelligence: 0.1998, data analyze: 0.2887, statistics: 0.3171, .
                
                Student input: 'keywords': math: 0.5, artificial intelligence: 0.5, data analyze: 0.5, statistics: 0.5, .
                """
        },
        {
            "role": "assistant",
            "content":
                """
                Based on the student's input, we recommend the following courses for Period 1:
                
                1. Course Name: Basic Mathematical Tools (SCI1010)
                Description: This course provides foundational knowledge in mathematics, covering analysis, algebra, and probability. It starts with a refresher of high school topics and progresses to advanced subjects preparing students for quantitative courses. Skills in solving equations, differentiation, and function analysis are developed.
                Explanation: 
                - Math : The student's interest in math correlates with the course's intensive focus on refreshing and building upon mathematical principles and techniques.
                - Artificial Intelligence : While the course does not directly deal with artificial intelligence, the mathematical foundations taught are essential for understanding AI algorithms and models. Mathematical tools, especially in probability and statistics, are the backbone of machine learning, a subset of AI.
                - Data Analysis : Data analysis relies heavily on statistics and mathematical models to interpret data. This course provides fundamental skills that will aid in the comprehension and application of data analysis techniques.
                - Statistics : Given the student's interest in statistics, the course's inclusion of probability theory, which is a bedrock of statistical analysis, makes it a good match.
                
                2. Course Name: Optimization (SCI3003
                Description: Optimization focuses on improving performance by using mathematical models. It covers techniques like gradient methods and Newton's method, exploring both unconstrained and constrained optimization, including linear programming and the Lagrange multiplier method. Practical applications are demonstrated through various examples.
                Explanation: 
                - Math : Given the strong foundation in mathematics that this course provides, it matches the student's interest. Understanding optimization requires a deep grasp of mathematical concepts which the student has shown a significant interest in.
                - Artificial Intelligence : Optimization is a key component in artificial intelligence for improving algorithms and decision-making processes. The student's interest is reflected in the course's strong connection to AI, particularly in techniques like gradient methods and iterative solutions which are used in machine learning.
                - Data Analysis : The course's focus on fitting models to data and solving least squares problems is highly relevant to data analysis, a field reliant on optimization for predictive modeling and interpretation.
                - Statistics : Statistical models often include optimization problems, such as least squares for regression analysis, that are covered in this course.
                
                3. Course Name: Statistics I (SSC2061)
                Description: An introductory course to statistics tailored for social and life sciences, covering descriptive and inferential statistics. The course aims to make statistics approachable through practical assignments, enabling students to visualize data, test hypotheses, and understand associations between different data types.
                Explanation: 
                - Math : The mathematical underpinnings of statistical theory are integral to this course. Given the student's interest in math, there's a clear relevance and applicability for the student wishing to apply mathematical concepts within the context of statistics.
                - Artificial Intelligence : While AI is not the primary focus of this course, the foundational knowledge in statistics is invaluable for understanding machine learning, a cornerstone of AI. The course includes relevant content that would benefit a student interested in AI, particularly in the application of statistical methods to inform and evaluate AI models.
                - Data Analysis : With a focus on descriptive and inferential statistics, the course covers the essentials of data analysis, making sense of data which is a significant interest for the student.
                - Statistics : The student's keen interest in statistics is directly met by this course. This indicates a major focus of the course on statistical principles and methodologies, perfectly aligning with what the student is looking to learn.
                
                4. Course Name: Contemporary World History (COR1003)
                Description: This course explores significant historical events and their impacts on the modern world. It employs a philosophical lens to study history, with a focus on state and nation concepts, economic trends, and global power dynamics. Case studies from various periods are used to link the past with current global situations.
                Explanation: 
                Math: This history course likely does not delve into these areas, focusing instead on the narrative, analysis, and interpretation of historical events.
                Statistics: It won't likely cover statistical theories or tools such as hypothesis testing, probability distributions, or data visualization techniques that a student interested in statistics would be looking to study.
                Artificial Intelligence: AI typically involves the use of computer science and mathematics to create systems capable of performing tasks that usually require human intelligence. Contemporary World History does not address these computational or technical aspects and instead focuses on the socio-political development of human societies.
                Data Analysis: Although historical studies can involve the analysis of data from past events, this course likely focuses more on qualitative analysis and theoretical perspectives rather than the quantitative data analysis the student is interested in.
                """
        }
    ]

def get_prompty_template_per_course():
    """
    function: get the prompt template for LLM
    """
    return [
        {
            "role": "system",
            "content":
                """
                You are tasked with assisting students in choosing academic courses that align closely with their interests and educational aspirations. To facilitate personalized course recommendations, you must leverage the information provided based on a student's specified interests.
                
                You will be supplied with a course, which includes the following attributes:
                
                1. Course Code.
                2. Course Title.
                3. Keywords: A dictionary consisting of relevant keywords associated with the course, along with their values. Be aware that the provided values might not always accurately represent the course content.
                
                In addition to the course information, you will receive inputs from students, which include:
                
                "keywords": A dictionary of keywords and their associated values that represent the student's areas of interest.
                
                Based on the course list and the student input, generate a recommendation for the specified course. For the course you recommend, ensure to include the following details:
                
                Course Name: State the full title of the course.
                Description: Offer a clear and detailed description of the course, including the topics that will be covered and the skills that will be taught.
                Explanation: Provide a thoughtful analysis explaining how this course aligns with the student's expressed interests.
                
                Remember, your goal is to match courses to a student's interests, keywords as accurately as possible, which will require you to interpret the relevance of course in relation to the interests specified by the student.
                """
        },
        {
            "role": "user",
            "content":
                """
                Course:
                1. Course Code: SCI1010. 
                2. Course Name: Basic Mathematical Tools. 
                3. Keywords: math: 0.5414, artificial intelligence: 0.2915, data analyze: 0.2977, statistics: 0.3019, .
                
                Student input: 'keywords': math: 0.5, artificial intelligence: 0.5, data analyze: 0.5, statistics: 0.5, .
                """
        },
        {
            "role": "assistant",
            "content":
                """
                Course Name: Basic Mathematical Tools (SCI1010)
                Description: This course provides foundational knowledge in mathematics, covering analysis, algebra, and probability. It starts with a refresher of high school topics and progresses to advanced subjects preparing students for quantitative courses. Skills in solving equations, differentiation, and function analysis are developed.
                Explanation: 
                - Math : The student's interest in math correlates with the course's intensive focus on refreshing and building upon mathematical principles and techniques.
                - Artificial Intelligence : While the course does not directly deal with artificial intelligence, the mathematical foundations taught are essential for understanding AI algorithms and models. Mathematical tools, especially in probability and statistics, are the backbone of machine learning, a subset of AI.
                - Data Analysis : Data analysis relies heavily on statistics and mathematical models to interpret data. This course provides fundamental skills that will aid in the comprehension and application of data analysis techniques.
                - Statistics : Given the student's interest in statistics, the course's inclusion of probability theory, which is a bedrock of statistical analysis, makes it a good match.
                """
        },
        {
            "role": "user",
            "content":
                """
                Course:
                1. Course Code: SCI3003. 
                2. Course Name: Optimization. 
                3. Keywords: math: 0.3381, artificial intelligence: 0.4353,  data analyze: 0.2453, statistics: 0.1996, .
                
                Student input: 'keywords': math: 0.5, artificial intelligence: 0.5, data analyze: 0.5, statistics: 0.5, .
                """
        },
        {
            "role": "assistant",
            "content":
                """
                Course Name: Optimization (SCI3003)
                Description: Optimization focuses on improving performance by using mathematical models. It covers techniques like gradient methods and Newton's method, exploring both unconstrained and constrained optimization, including linear programming and the Lagrange multiplier method. Practical applications are demonstrated through various examples.\
                Explanation: 
                - Math : Given the strong foundation in mathematics that this course provides, it matches the student's interest. Understanding optimization requires a deep grasp of mathematical concepts which the student has shown a significant interest in.
                - Artificial Intelligence : Optimization is a key component in artificial intelligence for improving algorithms and decision-making processes. The student's interest is reflected in the course's strong connection to AI, particularly in techniques like gradient methods and iterative solutions which are used in machine learning.
                - Data Analysis : The course's focus on fitting models to data and solving least squares problems is highly relevant to data analysis, a field reliant on optimization for predictive modeling and interpretation.
                - Statistics : Statistical models often include optimization problems, such as least squares for regression analysis, that are covered in this course.
                """
        },
        {
            "role": "user",
            "content":
                """
                Course:
                1. Course Code: SSC2061. 
                2. Course Name: Statistics I. 
                3. Keywords: math: 0.3343, artificial intelligence: 0.2078, data analyze: 0.347, statistics: 0.7102, .
                
                Student input: 'keywords': math: 0.5, artificial intelligence: 0.5, data analyze: 0.5, statistics: 0.5, .
                """
        },
        {
            "role": "assistant",
            "content":
                """
                Course Name: Statistics I (SSC2061)
                Description: An introductory course to statistics tailored for social and life sciences, covering descriptive and inferential statistics. The course aims to make statistics approachable through practical assignments, enabling students to visualize data, test hypotheses, and understand associations between different data types.
                Explanation: 
                - Math : The mathematical underpinnings of statistical theory are integral to this course. Given the student's interest in math, there's a clear relevance and applicability for the student wishing to apply mathematical concepts within the context of statistics.
                - Artificial Intelligence : While AI is not the primary focus of this course, the foundational knowledge in statistics is invaluable for understanding machine learning, a cornerstone of AI. The course includes relevant content that would benefit a student interested in AI, particularly in the application of statistical methods to inform and evaluate AI models.
                - Data Analysis : With a focus on descriptive and inferential statistics, the course covers the essentials of data analysis, making sense of data which is a significant interest for the student.
                - Statistics : The student's keen interest in statistics is directly met by this course. This indicates a major focus of the course on statistical principles and methodologies, perfectly aligning with what the student is looking to learn.
                """
        },
        {
            "role": "user",
            "content":
                """
                Course:
                1. Course Code: COR1003. 
                2. Course Name: Contemporary World History. 
                3. Keywords: math: 0.229, artificial intelligence: 0.1998, data analyze: 0.2887, statistics: 0.3171, .
                
                Student input: 'keywords': math: 0.5, artificial intelligence: 0.5, data analyze: 0.5, statistics: 0.5, .
                """
        },
        {
            "role": "assistant",
            "content":
                """
                Course Name: Basic Mathematical Tools (SCI1010)
                Description:  This course explores significant historical events and their impacts on the modern world. It employs a philosophical lens to study history, with a focus on state and nation concepts, economic trends, and global power dynamics. Case studies from various periods are used to link the past with current global situations.
                Explanation: 
                - Math: This history course likely does not delve into these areas, focusing instead on the narrative, analysis, and interpretation of historical events.
                - Artificial Intelligence: AI typically involves the use of computer science and mathematics to create systems capable of performing tasks that usually require human intelligence. Contemporary World History does not address these computational or technical aspects and instead focuses on the socio-political development of human societies.
                - Data Analysis: Although historical studies can involve the analysis of data from past events, this course likely focuses more on qualitative analysis and theoretical perspectives rather than the quantitative data analysis the student is interested in.
                - Statistics: It won't likely cover statistical theories or tools such as hypothesis testing, probability distributions, or data visualization techniques that a student interested in statistics would be looking to study.
                """
        }
    ]
