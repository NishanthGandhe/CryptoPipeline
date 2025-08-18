import os, psycopg
from dotenv import load_dotenv

load_dotenv()
user = os.getenv("DB_USER")
pwd = os.getenv("DB_PASS")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

def run_sql_file(conn, path):
    with open(path, "r", encoding="utf-8") as f, conn.cursor() as cur:
        cur.execute(f.read())

def main():
    print("Connecting…")
    with psycopg.connect(user=user, password=pwd, host=host, port=port, dbname=db) as conn:
        print("Running transform.sql …")
        run_sql_file(conn, "pipeline/transform.sql")
        print("Running metrics.sql …")
        run_sql_file(conn, "pipeline/metrics.sql")
    print("Done.")

if __name__ == "__main__":
    main()
