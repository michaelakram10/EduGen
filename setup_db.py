import os

import psycopg2
from dotenv import load_dotenv

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


def create_tables():
    try:
        params = build_db_params()
        if "dsn" in params:
            conn = psycopg2.connect(params["dsn"], sslmode=params.get("sslmode", "require"))
        else:
            conn = psycopg2.connect(**params)
        cur = conn.cursor()

        commands = [
            "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, role TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS exams (id SERIAL PRIMARY KEY, topic TEXT NOT NULL, content TEXT NOT NULL, difficulty TEXT, created_by INTEGER REFERENCES users(id))",
            "CREATE TABLE IF NOT EXISTS submissions (id SERIAL PRIMARY KEY, exam_id INTEGER REFERENCES exams(id), student_id INTEGER REFERENCES users(id), student_answers TEXT, ai_feedback TEXT, numerical_score INTEGER, submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_provider TEXT",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS oauth_subject TEXT",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS email TEXT",
            "CREATE UNIQUE INDEX IF NOT EXISTS users_oauth_identity_uniq ON users(oauth_provider, oauth_subject)",
        ]

        for command in commands:
            cur.execute(command)

        conn.commit()
        print("Tables created/updated successfully.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    create_tables()
