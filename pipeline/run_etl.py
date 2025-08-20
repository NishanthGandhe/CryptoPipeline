import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
user = os.getenv("DB_USER")
pwd = os.getenv("DB_PASS")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

def run_sql_file(conn, path):
    """
    Reads a SQL file, splits it into individual statements,
    ignores comments, and executes them one by one.
    """
    print(f"  Reading SQL from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        sql_statements = f.read().split(';')

    with conn.cursor() as cur:
        for i, statement in enumerate(sql_statements):
            if statement.strip() and not statement.strip().startswith('--'):
                try:
                    print(f"    Executing statement {i+1}...")
                    cur.execute(statement)
                except Exception as e:
                    print(f"      ERROR executing statement: {e}")
                    raise e

def main():    
    print("Connecting…")
    conn_info = f"user={user} password={pwd} host={host} port={port} dbname={db}"
    with psycopg.connect(conn_info, autocommit=True) as conn:
        print("Running transform.sql…")
        run_sql_file(conn, "pipeline/transform.sql")
        
        print("Running metrics.sql…")
        run_sql_file(conn, "pipeline/metrics.sql")
        
    print("Done.")

if __name__ == "__main__":
    main()