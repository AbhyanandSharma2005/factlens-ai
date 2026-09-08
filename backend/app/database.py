import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    # Connect to Neon Postgres
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    # 1. Enable the AI Vector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Create the Documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename TEXT NOT NULL,
            total_pages INT NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'processing',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    
    # 3. Create the Facts table (with a 384-dimension vector column for AI search)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
            subject TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            metric_value JSONB NOT NULL,
            scope_context JSONB NOT NULL,
            qualifiers TEXT[] DEFAULT '{}',
            evidence_page INT NOT NULL,
            evidence_text TEXT NOT NULL,
            evidence_bbox JSONB,
            confidence FLOAT NOT NULL,
            embedding VECTOR(384),
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    
    # 4. Create the Relationships table (The "Knowledge Graph")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_relationships (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            fact_a_id UUID REFERENCES facts(id) ON DELETE CASCADE,
            fact_b_id UUID REFERENCES facts(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            explanation TEXT NOT NULL,
            confidence FLOAT NOT NULL DEFAULT 1.0,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT unique_fact_pair UNIQUE (fact_a_id, fact_b_id)
        );
    """)
    
    # 5. Create a high-speed search index for the AI embeddings
    cur.execute("""
        CREATE INDEX IF NOT EXISTS facts_embedding_idx 
        ON facts USING hnsw (embedding vector_cosine_ops);
    """)
    
    # Save changes and close
    conn.commit()
    cur.close()
    conn.close()
    print("Database and vector tables initialized successfully!")

if __name__ == "__main__":
    init_db()