import os
import json
import re
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from fastembed import TextEmbedding

# 1. Initialize the local embedding model (Runs free on your CPU)
print("Loading Embedding Model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Initialize Gemini Client (Pulls from your .env automatically)
client = genai.Client()

# 3. The strict JSON schema we force Gemini to return
EXTRACTION_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "subject": {"type": "STRING", "description": "What or who the fact is about."},
            "fact_type": {"type": "STRING", "description": "e.g., Financial, Operational, Macroeconomic"},
            "metric_value": {
                "type": "OBJECT",
                "properties": {
                    "raw_value": {"type": "STRING", "description": "The exact number or metric"}
                },
            },
            "scope_context": {
                "type": "OBJECT",
                "properties": {
                    "period": {"type": "STRING", "description": "e.g., FY24, Q1 2023, etc."}
                }
            },
            "evidence_text": {"type": "STRING", "description": "MUST be an exact verbatim quote from the text."},
            "confidence": {"type": "NUMBER"}
        },
        "required": ["subject", "fact_type", "metric_value", "scope_context", "evidence_text", "confidence"]
    }
}

def process_pdf_page(pdf_path: str, page_num: int):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text")
    
    # Skip empty or boilerplate pages (like covers)
    if len(text.strip()) < 100:
        return []

    prompt = f"""Extract 3 to 5 of the most important numerical or factual claims from this page.
    The evidence_text MUST be an exact, word-for-word substring from the text below. Do not paraphrase.
    
    PAGE TEXT:
    {text}
    """
    
    # Call Gemini 1.5 Flash
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EXTRACTION_SCHEMA,
            temperature=0.0  # Zero temp ensures factual, non-creative extraction
        )
    )
    
    try:
        raw_facts = json.loads(response.text)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return []

    verified_facts = []
    
    # 4. Anti-Hallucination & Bounding Box Logic
    for item in raw_facts:
        ev_text = item.get("evidence_text", "").strip()
        
        # Clean up invisible newlines so we can match it perfectly
        clean_ev = re.sub(r'\s+', ' ', ev_text)
        clean_doc = re.sub(r'\s+', ' ', text)
        
        if clean_ev and clean_ev in clean_doc:
            # Find the coordinates of the text to draw a highlight box in the React UI later
            rects = page.search_for(ev_text[:50]) 
            bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1] if rects else None
            
            # Create the AI embedding vector for semantic search
            embed_repr = f"{item['subject']} | {item['fact_type']} | {item['metric_value'].get('raw_value', '')}"
            vector = list(embed_model.embed([embed_repr]))[0].tolist()
            
            item["evidence_bbox"] = bbox
            item["evidence_page"] = page_num
            item["embedding"] = vector
            verified_facts.append(item)
        else:
            # If Gemini hallucinates or paraphrases the quote, we reject it!
            print(f"❌ REJECTED (Hallucination detected): {ev_text[:50]}...")

    return verified_facts