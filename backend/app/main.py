from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from app.extraction import process_pdf_page
from app.reconciler import adjudicate_pair

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="FactLens AI - Knowledge Layer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def run_ingestion_pipeline(doc_id: str, file_path: str, filename: str):
    import fitz  # PyMuPDF
    conn = get_db()
    cur = conn.cursor()
    doc = None
    
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        for page_idx in range(1, total_pages + 1):
            print(f"Processing {filename} - Page {page_idx}/{total_pages}...")
            
            page_facts = process_pdf_page(file_path, page_idx)
            
            for f in page_facts:
                cur.execute(
                    """INSERT INTO facts (document_id, subject, fact_type, metric_value, scope_context, 
                       qualifiers, evidence_page, evidence_text, evidence_bbox, confidence, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (doc_id, f["subject"], f["fact_type"], json.dumps(f["metric_value"]), 
                     json.dumps(f["scope_context"]), f.get("qualifiers", []), f["evidence_page"], 
                     f["evidence_text"], json.dumps(f["evidence_bbox"]), f["confidence"], f["embedding"])
                )
                fact_id = cur.fetchone()["id"]
                
                cur.execute(
                    """SELECT f.id, f.subject, f.fact_type, f.metric_value, f.scope_context, 
                              f.evidence_page, f.evidence_text, d.filename as doc_name
                       FROM facts f 
                       JOIN documents d ON f.document_id = d.id 
                       WHERE f.document_id != %s AND (f.embedding <=> %s::vector) < 0.28 
                       LIMIT 3""",
                    (doc_id, f["embedding"])
                )
                candidates = cur.fetchall()
                
                for cand in candidates:
                    print(f"Checking relationship between '{f['subject']}' and '{cand['subject']}'...")
                    adj = adjudicate_pair(
                        fact1={"evidence_page": f["evidence_page"], "subject": f["subject"], "fact_type": f["fact_type"], "metric_value": f["metric_value"], "scope_context": f["scope_context"], "evidence_text": f["evidence_text"]},
                        fact2={"evidence_page": cand["evidence_page"], "subject": cand["subject"], "fact_type": cand["fact_type"], "metric_value": cand["metric_value"], "scope_context": cand["scope_context"], "evidence_text": cand["evidence_text"]},
                        doc1_name=filename,
                        doc2_name=cand["doc_name"]
                    )
                    
                    if adj["relation_type"] != "insufficient_evidence":
                        id_a, id_b = sorted([str(fact_id), str(cand["id"])])
                        cur.execute(
                            """INSERT INTO fact_relationships (fact_a_id, fact_b_id, relation_type, explanation, confidence)
                               VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                            (id_a, id_b, adj["relation_type"], adj["explanation"], adj.get("confidence", 1.0))
                        )

        cur.execute("UPDATE documents SET status = 'ready', total_pages = %s WHERE id = %s", (total_pages, doc_id))
        conn.commit()
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        cur.execute("UPDATE documents SET status = 'failed' WHERE id = %s", (doc_id,))
        conn.commit()
    finally:
        if doc:
            doc.close()  # <-- FIX: Release the file lock so Windows can delete it
        cur.close()
        conn.close()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@app.post("/api/documents")
async def upload_document(file: UploadFile, bg_tasks: BackgroundTasks):
    doc_id = str(uuid.uuid4())
    os.makedirs("/tmp/factlens", exist_ok=True)
    temp_path = f"/tmp/factlens/{doc_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO documents (id, filename, status) VALUES (%s, %s, 'processing')", (doc_id, file.filename))
    conn.commit()
    cur.close()
    conn.close()
    
    bg_tasks.add_task(run_ingestion_pipeline, doc_id, temp_path, file.filename)
    return {"document_id": doc_id, "status": "processing", "message": "Document is being analyzed."}

@app.get("/api/facts")
async def get_all_facts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.id, f.subject, f.fact_type, f.metric_value, f.scope_context, f.evidence_text, f.evidence_page, d.filename
        FROM facts f JOIN documents d ON f.document_id = d.id
        ORDER BY f.created_at DESC
    """)
    facts = cur.fetchall()
    
    for fact in facts:
        cur.execute("""
            SELECT r.relation_type, r.explanation, f_other.evidence_text as other_fact_text
            FROM fact_relationships r
            JOIN facts f_other ON (r.fact_a_id = f_other.id OR r.fact_b_id = f_other.id)
            WHERE (r.fact_a_id = %s OR r.fact_b_id = %s) AND f_other.id != %s
        """, (fact["id"], fact["id"], fact["id"]))
        fact["relationships"] = cur.fetchall()
        
    cur.close()
    conn.close()
    return facts