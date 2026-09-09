import hashlib
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_text_hash(text: str) -> str:
    """Generates a unique SHA-256 hash for a block of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def check_fact_cache(text_hash: str):
    """Checks if facts for this exact text hash already exist in the database."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        cur.execute("SELECT cached_facts FROM fact_cache WHERE text_hash = %s;", (text_hash,))
        row = cur.fetchone()
        if row:
            return row["cached_facts"]
        return None
    except Exception:
        # If table doesn't exist yet, fail gracefully
        return None
    finally:
        cur.close()
        conn.close()

def save_to_fact_cache(text_hash: str, facts: list):
    """Saves extracted facts to the cache table."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fact_cache (
                text_hash TEXT PRIMARY KEY,
                cached_facts JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        cur.execute("""
            INSERT INTO fact_cache (text_hash, cached_facts)
            VALUES (%s, %s)
            ON CONFLICT (text_hash) DO NOTHING;
        """, (text_hash, json.dumps(facts)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Cache save error: {e}")
    finally:
        cur.close()
        conn.close()