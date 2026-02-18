import os
from zai import ZaiClient  # Changed from groq import Groq
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# CONFIGURATION
# =========================
# Use the latest GLM model from Z.ai
CURRENT_MODEL = "glm-4.7" 
# Initialize the Z.ai Client
ZAI_CLIENT = ZaiClient(api_key="c5988a99a3524a6c9a669ba11ce94d8b.DajtzU3HlGPJFHuA") 

class RAGPipeline:
    def __init__(self, pdf_folder="pdfs"):
        print("Initializing Knowledge Base...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists("faiss_index"):
            self.db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        else:
            docs = []
            if not os.path.exists(pdf_folder):
                os.makedirs(pdf_folder)
            
            for file in os.listdir(pdf_folder):
                if file.lower().endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(os.path.join(pdf_folder, file))
                        docs.extend(loader.load())
                    except: continue
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            if chunks:
                self.db = FAISS.from_documents(chunks, self.embeddings)
                self.db.save_local("faiss_index")
            else:
                raise Exception("No PDF content found in 'pdfs' folder.")

    def query(self, topic, mcq_count=3, essay_count=2, difficulty="Beginner", mode="General"):
        docs = self.db.similarity_search(topic, k=5)
        context_text = "\n\n".join([d.page_content for d in docs])
        sources = {os.path.basename(d.metadata.get('source', 'Unknown')) for d in docs}
        
        if mode == "Instructor Mode":
            system_msg = "You are a professional university professor specializing in technical curriculum design."
            user_msg = f"""Create a {difficulty} level technical exam about '{topic}' based on the provided context.
            
            CONTEXT:
            {context_text[:5000]}

            STRICT REQUIREMENTS:
            1. Generate EXACTLY {mcq_count} Multiple Choice Questions (MCQs). Each with 4 options, Correct Answer, and Explanation.
            2. Generate EXACTLY {essay_count} Essay Questions. Include a 'Model Answer Key'.

            FORMATTING: Use clear headings.
            """
        else:
            system_msg = "You are a helpful academic assistant."
            user_msg = f"Use the context to answer: {topic}\n\nContext: {context_text[:3000]}"

        try:
            # Updated to ZaiClient syntax
            chat_completion = ZAI_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg}, 
                    {"role": "user", "content": user_msg}
                ],
                model=CURRENT_MODEL, 
                temperature=0.4,
            )
            return chat_completion.choices[0].message.content, sources
        except Exception as e:
            return f"Error calling Z.ai: {str(e)}", sources

    def grade_submission(self, exam_content, student_answer):
        system_msg = "You are a strict university grader. You must output a numerical score and qualitative feedback."
        user_msg = f"""
        Compare the Student Submission against the Model Answer Key provided in the Exam Content.
        
        EXAM CONTENT & KEY:
        {exam_content}

        STUDENT SUBMISSION:
        {student_answer}

        STRICT OUTPUT FORMAT:
        Line 1: Score: [Numerical value out of 100]/100
        Following Lines: Provide detailed feedback explaining the grade.
        """

        try:
            # Updated to ZaiClient syntax
            chat_completion = ZAI_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg}, 
                    {"role": "user", "content": user_msg}
                ],
                model=CURRENT_MODEL,
                temperature=0.2, 
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Grading Error: {str(e)}"