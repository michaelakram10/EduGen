import hashlib
import json
import os
import re

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()


def build_db_params():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return {"dsn": database_url, "sslmode": "require"}

    project_ref = os.getenv("SUPABASE_PROJECT_REF")
    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    db_user = os.getenv("SUPABASE_DB_USER", "postgres")
    db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    db_host = os.getenv("SUPABASE_DB_HOST") or (
        f"db.{project_ref}.supabase.co" if project_ref else None
    )
    db_port = os.getenv("SUPABASE_DB_PORT", "5432")

    if db_host and db_password:
        return {
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "sslmode": "require",
        }

    raise RuntimeError(
        "Database is not configured. Set DATABASE_URL or SUPABASE_PROJECT_REF + SUPABASE_DB_PASSWORD."
    )


def get_db_connection():
    params = build_db_params()
    if "dsn" in params:
        return psycopg2.connect(params["dsn"], sslmode=params.get("sslmode", "require"))
    return psycopg2.connect(**params)


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def parse_json_blob(text):
    if not text:
        return None

    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\\s*```$", "", candidate)

    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def format_exam_for_instructor(exam_data):
    lines = [f"Topic: {exam_data.get('topic', '')}", f"Difficulty: {exam_data.get('difficulty', '')}", ""]

    for i, mcq in enumerate(exam_data.get("mcqs", []), start=1):
        lines.append(f"MCQ {i}: {mcq.get('question', '')}")
        options = mcq.get("options", [])
        for j, opt in enumerate(options):
            lines.append(f"  {chr(65 + j)}. {opt}")
        idx = mcq.get("correct_option_index", 0)
        answer = options[idx] if options and isinstance(idx, int) and 0 <= idx < len(options) else ""
        lines.append(f"  Correct: {answer}")
        explanation = mcq.get("explanation", "")
        if explanation:
            lines.append(f"  Explanation: {explanation}")
        lines.append("")

    for i, essay in enumerate(exam_data.get("essays", []), start=1):
        lines.append(f"Essay {i}: {essay.get('question', '')}")
        model_answer = essay.get("model_answer", "")
        if model_answer:
            lines.append(f"  Model Answer: {model_answer}")
        lines.append("")

    return "\n".join(lines)


def format_student_submission(student_data, exam_data=None):
    lines = []

    mcq_answers = student_data.get("mcq_answers", []) if isinstance(student_data, dict) else []
    essay_answers = student_data.get("essay_answers", []) if isinstance(student_data, dict) else []

    if mcq_answers:
        lines.append("MCQ Answers:")
        for i, ans in enumerate(mcq_answers, start=1):
            question = ans.get("question", f"MCQ {i}")
            chosen = ans.get("selected_option", "")
            lines.append(f"  {i}. {question}")
            lines.append(f"     Selected: {chosen}")

    if essay_answers:
        lines.append("\nEssay Answers:")
        for i, ans in enumerate(essay_answers, start=1):
            question = ans.get("question", f"Essay {i}")
            answer = ans.get("answer", "")
            lines.append(f"  {i}. {question}")
            lines.append(f"     {answer}")

    if not lines:
        return "No structured answers found."

    return "\n".join(lines)


# --- AUTH LOGIC ---
def login_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, role FROM users WHERE username=%s AND password=%s",
        (username, hash_password(password)),
    )
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline("pdfs")

# --- UI LOGIC ---
if not st.session_state.logged_in:
    st.title("AI University Portal")
    choice = st.selectbox("Action", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        role = st.selectbox("Role", ["student", "instructor"])
        if st.button("Register"):
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                    (username, hash_password(password), role),
                )
                conn.commit()
                st.success("Created! Please login.")
            except Exception as e:
                st.error(f"Registration Error: {e}")
            finally:
                if "cur" in locals():
                    cur.close()
                if "conn" in locals():
                    conn.close()
    else:
        if st.button("Login"):
            u = login_user(username, password)
            if u:
                st.session_state.logged_in = True
                st.session_state.user = {"id": u[0], "username": u[1], "role": u[2]}
                st.rerun()
            else:
                st.error("Invalid username or password")

else:
    user = st.session_state.user
    st.sidebar.title(f"Welcome, {user['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    if user["role"] == "instructor":
        tab1, tab2, tab3 = st.tabs(["Generate Exam", "Grade Submissions", "Analytics Dashboard"])

        with tab1:
            topic = st.text_input("Topic")
            m, e = st.columns(2)
            mcq_n = m.slider("MCQs", 1, 10, 5)
            ess_n = e.slider("Essays", 1, 5, 2)
            diff = st.select_slider("Level", ["Beginner", "Intermediate", "Expert"])
            if st.button("Save Exam"):
                text, _ = st.session_state.rag.query(topic, mcq_n, ess_n, diff, "Instructor Mode")
                exam_data = parse_json_blob(text)

                if exam_data and (exam_data.get("mcqs") or exam_data.get("essays")):
                    stored_content = json.dumps(exam_data, ensure_ascii=True)
                    st.text_area(
                        "Instructor Preview (includes answer key)",
                        format_exam_for_instructor(exam_data),
                        height=300,
                    )
                else:
                    stored_content = text
                    st.warning("Generated content is not structured JSON; student form mode may not work for this exam.")
                    st.text_area("Preview", text, height=300)

                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO exams (topic, content, difficulty, created_by) VALUES (%s,%s,%s,%s)",
                    (topic, stored_content, diff, user["id"]),
                )
                conn.commit()
                cur.close()
                conn.close()
                st.success("Exam Saved")

        with tab2:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT s.id, u.username, e.topic, s.student_answers, e.content, s.ai_feedback FROM submissions s JOIN users u ON s.student_id = u.id JOIN exams e ON s.exam_id = e.id WHERE e.created_by = %s",
                (user["id"],),
            )
            subs = cur.fetchall()
            cur.close()
            conn.close()

            for s_id, u_name, topic, s_ans, e_cont, feedback in subs:
                exam_data = parse_json_blob(e_cont)
                student_data = parse_json_blob(s_ans)

                with st.expander(f"{u_name} - {topic}"):
                    c1, c2 = st.columns(2)

                    if exam_data:
                        c1.text_area(
                            "Exam + Key",
                            format_exam_for_instructor(exam_data),
                            height=260,
                            key=f"k{s_id}",
                        )
                    else:
                        c1.text_area("Exam + Key", e_cont, height=260, key=f"k{s_id}")

                    if student_data:
                        c2.text_area(
                            "Student Submission",
                            format_student_submission(student_data, exam_data),
                            height=260,
                            key=f"s{s_id}",
                        )
                    else:
                        c2.text_area("Student Submission", s_ans, height=260, key=f"s{s_id}")

                    if feedback:
                        st.info(feedback)
                    elif st.button("Auto-Grade", key=f"b{s_id}"):
                        res = st.session_state.rag.grade_submission(e_cont, s_ans)
                        try:
                            score_match = re.search(r"(\\d+)/100", res) or re.search(
                                r"Score:\\s*(\\d+)", res, re.I
                            )
                            val = int(score_match.group(1)) if score_match else 0
                        except Exception:
                            val = 0

                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            """
                            UPDATE submissions
                            SET ai_feedback = %s, numerical_score = %s
                            WHERE id = %s
                            """,
                            (res, val, s_id),
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.rerun()

        with tab3:
            st.header("Class Performance Analytics")

            conn = get_db_connection()
            query = """
                SELECT e.topic, s.numerical_score, u.username
                FROM submissions s
                JOIN exams e ON s.exam_id = e.id
                JOIN users u ON s.student_id = u.id
                WHERE e.created_by = %s AND s.numerical_score IS NOT NULL
            """
            df = pd.read_sql(query, conn, params=(user["id"],))
            conn.close()

            if not df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Average Score per Topic")
                    avg_scores = df.groupby("topic")["numerical_score"].mean()
                    st.bar_chart(avg_scores)

                with col2:
                    st.subheader("Score Distribution")
                    st.line_chart(df["numerical_score"])

                st.subheader("Student Leaderboard")
                leaderboard = (
                    df.groupby("username")["numerical_score"].mean().sort_values(ascending=False)
                )
                st.table(leaderboard)
            else:
                st.info("No graded data available for analytics yet.")

    else:
        st.title("Student Portal")
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, topic, difficulty, content FROM exams")
        exams = cur.fetchall()
        cur.close()
        conn.close()

        if exams:
            ex = st.selectbox("Choose Exam", exams, format_func=lambda x: f"{x[1]} ({x[2]})")
            exam_id, exam_topic, exam_difficulty, exam_content = ex

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT student_answers, ai_feedback FROM submissions WHERE exam_id=%s AND student_id=%s",
                (exam_id, user["id"]),
            )
            done = cur.fetchone()
            cur.close()
            conn.close()

            exam_data = parse_json_blob(exam_content)

            if done:
                st.warning("Already submitted")
                submitted_answers = done[0] if done else ""
                feedback = done[1] if done else None

                submitted_structured = parse_json_blob(submitted_answers)
                if submitted_structured:
                    st.text_area(
                        "Your Submission",
                        format_student_submission(submitted_structured, exam_data),
                        height=260,
                    )
                else:
                    st.text_area("Your Submission", submitted_answers or "", height=260)

                if feedback:
                    st.success(f"Feedback: {feedback}")
            else:
                if exam_data and (exam_data.get("mcqs") or exam_data.get("essays")):
                    st.subheader(f"{exam_topic} ({exam_difficulty})")

                    with st.form(f"exam_form_{exam_id}"):
                        mcq_answers = []
                        for i, mcq in enumerate(exam_data.get("mcqs", []), start=1):
                            st.markdown(f"**MCQ {i}. {mcq.get('question', '')}**")
                            options = mcq.get("options", [])
                            selected = st.radio(
                                label=f"Choose one option for MCQ {i}",
                                options=list(range(len(options))),
                                format_func=lambda idx, opts=options: opts[idx],
                                key=f"mcq_{exam_id}_{i}",
                            )
                            mcq_answers.append(
                                {
                                    "id": mcq.get("id", f"MCQ-{i}"),
                                    "question": mcq.get("question", ""),
                                    "selected_option_index": int(selected),
                                    "selected_option": options[int(selected)] if options else "",
                                }
                            )

                        essay_answers = []
                        for i, essay in enumerate(exam_data.get("essays", []), start=1):
                            st.markdown(f"**Essay {i}. {essay.get('question', '')}**")
                            answer_text = st.text_area(
                                label=f"Your answer for Essay {i}",
                                key=f"essay_{exam_id}_{i}",
                            )
                            essay_answers.append(
                                {
                                    "id": essay.get("id", f"ESSAY-{i}"),
                                    "question": essay.get("question", ""),
                                    "answer": answer_text,
                                }
                            )

                        submit_exam = st.form_submit_button("Submit Exam")

                    if submit_exam:
                        payload = {
                            "mcq_answers": mcq_answers,
                            "essay_answers": essay_answers,
                        }
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO submissions (exam_id, student_id, student_answers) VALUES (%s,%s,%s)",
                            (exam_id, user["id"], json.dumps(payload, ensure_ascii=True)),
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success("Exam submitted")
                        st.rerun()
                else:
                    st.warning("This exam is in legacy text format. Ask instructor to regenerate it for form mode.")
                    st.text_area("Exam", exam_content, height=280)
                    ans = st.text_area("Answers")
                    if st.button("Submit"):
                        conn = get_db_connection()
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO submissions (exam_id, student_id, student_answers) VALUES (%s,%s,%s)",
                            (exam_id, user["id"], ans),
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.rerun()
