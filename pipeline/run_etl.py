import os, psycopg
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

def run_sql_file(conn, path):
    with open(path, "r", encoding="utf-8") as f, conn.cursor() as cur:
        cur.execute(f.read())

def main():
    print("Connecting…")
    with psycopg.connect(user=USER, password=PASSWORD, host=HOST, port=PORT, dbname=DBNAME) as conn:
        print("Running transform.sql …")
        run_sql_file(conn, "pipeline/transform.sql")
        print("Running metrics.sql …")
        run_sql_file(conn, "pipeline/metrics.sql")
    print("Done.")

if __name__ == "__main__":
    main()
