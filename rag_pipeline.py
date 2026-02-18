import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# CONFIGURATION
# =========================
load_dotenv()
CURRENT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment.")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)


class RAGPipeline:
    def __init__(self, pdf_folder="pdfs"):
        print("Initializing Knowledge Base...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists("faiss_index"):
            self.db = FAISS.load_local(
                "faiss_index", self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            docs = []
            if not os.path.exists(pdf_folder):
                os.makedirs(pdf_folder)

            for file in os.listdir(pdf_folder):
                if file.lower().endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(os.path.join(pdf_folder, file))
                        docs.extend(loader.load())
                    except Exception:
                        continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            if chunks:
                self.db = FAISS.from_documents(chunks, self.embeddings)
                self.db.save_local("faiss_index")
            else:
                raise Exception("No PDF content found in 'pdfs' folder.")

    @staticmethod
    def _extract_json_payload(text):
        candidate = (text or "").strip()
        if not candidate:
            raise ValueError("Empty response")

        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\\s*", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\\s*```$", "", candidate)

        try:
            return json.loads(candidate)
        except Exception:
            pass

        decoder = json.JSONDecoder()
        for i, ch in enumerate(candidate):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(candidate[i:])
                    return obj
                except Exception:
                    continue

        raise ValueError("Could not parse JSON payload")

    @staticmethod
    def _normalize_exam_payload(payload, topic, difficulty):
        mcqs_in = payload.get("mcqs", []) if isinstance(payload, dict) else []
        essays_in = payload.get("essays", []) if isinstance(payload, dict) else []

        mcqs = []
        for i, q in enumerate(mcqs_in, start=1):
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "")).strip()
            options = [str(opt).strip() for opt in q.get("options", []) if str(opt).strip()]
            if not question or len(options) < 2:
                continue

            idx = q.get("correct_option_index", 0)
            try:
                idx = int(idx)
            except Exception:
                idx = 0
            idx = max(0, min(idx, len(options) - 1))

            mcqs.append(
                {
                    "id": str(q.get("id", f"MCQ-{i}")),
                    "question": question,
                    "options": options,
                    "correct_option_index": idx,
                    "explanation": str(q.get("explanation", "")).strip(),
                }
            )

        essays = []
        for i, q in enumerate(essays_in, start=1):
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "")).strip()
            if not question:
                continue
            essays.append(
                {
                    "id": str(q.get("id", f"ESSAY-{i}")),
                    "question": question,
                    "model_answer": str(q.get("model_answer", "")).strip(),
                }
            )

        return {
            "topic": str(payload.get("topic", topic)).strip() if isinstance(payload, dict) else topic,
            "difficulty": str(payload.get("difficulty", difficulty)).strip()
            if isinstance(payload, dict)
            else difficulty,
            "mcqs": mcqs,
            "essays": essays,
        }

    def query(
        self,
        topic,
        mcq_count=3,
        essay_count=2,
        difficulty="Beginner",
        mode="General",
    ):
        docs = self.db.similarity_search(topic, k=5)
        context_text = "\\n\\n".join([d.page_content for d in docs])
        sources = {os.path.basename(d.metadata.get("source", "Unknown")) for d in docs}

        if mode == "Instructor Mode":
            system_msg = "You generate university exams in strict JSON only."
            user_msg = f"""
Create a {difficulty} level exam on topic: {topic}
Use this context:\n{context_text[:5000]}

Return ONLY JSON with this schema:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "mcqs": [
    {{
      "id": "MCQ-1",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_option_index": 0,
      "explanation": "..."
    }}
  ],
  "essays": [
    {{
      "id": "ESSAY-1",
      "question": "...",
      "model_answer": "..."
    }}
  ]
}}

Requirements:
- Generate exactly {mcq_count} MCQs.
- Generate exactly {essay_count} essay questions.
- Each MCQ must have exactly 4 options.
- Do not include markdown, prose, or code fences.
"""
        else:
            system_msg = "You are a helpful academic assistant."
            user_msg = f"Use the context to answer: {topic}\\n\\nContext: {context_text[:3000]}"

        try:
            chat_completion = GROQ_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                model=CURRENT_MODEL,
                temperature=0.2 if mode == "Instructor Mode" else 0.4,
            )
            raw = chat_completion.choices[0].message.content
            if mode == "Instructor Mode":
                payload = self._extract_json_payload(raw)
                exam = self._normalize_exam_payload(payload, topic, difficulty)
                return json.dumps(exam, ensure_ascii=True), sources
            return raw, sources
        except Exception as e:
            return f"Error calling AI: {str(e)}", sources

    def grade_submission(self, exam_content, student_answer):
        system_msg = "You are a strict university grader."
        user_msg = f"""
Evaluate the student submission against the exam key.

EXAM CONTENT:
{exam_content}

STUDENT SUBMISSION:
{student_answer}

Rules:
- If JSON is provided, use correct_option_index and model_answer fields.
- Score out of 100.

STRICT OUTPUT FORMAT:
Line 1: Score: [Numerical value]/100
Following lines: concise feedback with strengths, mistakes, and improvements.
"""

        try:
            chat_completion = GROQ_CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                model=CURRENT_MODEL,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Grading Error: {str(e)}"
