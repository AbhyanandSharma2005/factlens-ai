import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute(
    "UPDATE documents SET status = 'failed' WHERE status = 'processing' AND total_pages = 0"
)
conn.commit()
print(f"{cur.rowcount} stuck documents marked failed")
cur.close()
conn.close()