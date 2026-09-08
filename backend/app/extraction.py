import os
import json
import re
import time
from dotenv import load_dotenv

# Load .env before initializing any client
load_dotenv()

import pymupdf as fitz  # Cleans up the deprecation warning
from google import genai
from google.genai import types
from fastembed import TextEmbedding

# 1. Initialize the local embedding model (Runs free on CPU)
print("Loading Embedding Model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 2. Explicitly pass the API key to the client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from backend/.env! Please add it.")

client = genai.Client(api_key=api_key)

# 3. Schema for strict fact extraction
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
                "required": ["raw_value"]
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


def looks_like_toc_or_cover(text: str) -> bool:
    """
    Cheap structural heuristic to skip table-of-contents, cover, and divider pages
    before spending an LLM call on them. These pages are mostly short lines
    (titles/headings) with very few lines that contain real data.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True

    short_line_ratio = sum(1 for l in lines if len(l) < 60) / len(lines)
    digit_lines = sum(1 for l in lines if any(c.isdigit() for c in l))
    digit_line_ratio = digit_lines / len(lines)

    # TOC/cover pages: dominated by short heading-like lines, very few lines with numbers
    return short_line_ratio > 0.8 and digit_line_ratio < 0.3


def has_quantifiable_content(item: dict) -> bool:
    """
    Guards against 'facts' that are really just headings or descriptive sentences
    with no actual number, date, or statistic in them.
    """
    raw_val = item.get("metric_value", {}).get("raw_value", "") or ""
    evidence = item.get("evidence_text", "") or ""
    return bool(re.search(r'\d', raw_val)) or bool(re.search(r'\d', evidence))


def process_pdf_page(pdf_path: str, page_num: int):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text")

    if len(text.strip()) < 100:
        return []

    if looks_like_toc_or_cover(text):
        print(f"⏭️  Skipping page {page_num} (looks like TOC/cover, no LLM call made)")
        return []

    prompt = f"""Extract 3 to 5 verifiable factual claims from this page. A valid fact MUST
contain a specific number, statistic, percentage, date, or a named quantitative comparison.

Do NOT extract:
- Chapter titles, section headings, or table of contents entries
- General descriptive or motivational sentences with no data (e.g. slide taglines)
- Page numbers, headers, footers, or boilerplate/disclaimer text

GOOD fact example: "India's CPI inflation eased to 4.9% in FY25 from 5.4% in FY24"
GOOD fact example: "The current account deficit narrowed to 0.6% of GDP in Q3 FY25"
BAD (reject) example: "Chapter 3: External Sector" — this is a heading, not a claim
BAD (reject) example: "Domestic Economy Remains Steady Amidst Global Uncertainties" — this is
  a section title/summary line with no quantifiable content

If this page is a table of contents, cover page, divider, or otherwise contains no facts
meeting the above bar, return an empty JSON array: []

The evidence_text MUST be an exact, word-for-word substring from the text below. Do not paraphrase.

PAGE TEXT:
{text}
"""

    # --- RETRY LOGIC ---
    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EXTRACTION_SCHEMA,
                    temperature=0.0
                )
            )
            break  # Success! Break out of the retry loop.
        except Exception as e:
            print(f"⚠️ API Overloaded (Attempt {attempt + 1}/{max_retries}). Retrying in 5 seconds...")
            time.sleep(5)
            if attempt == max_retries - 1:
                print("Skipping page due to persistent API errors.")
                return []
    # -------------------

    if not response or not response.text:
        return []

    try:
        raw_facts = json.loads(response.text)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return []

    verified_facts = []

    for item in raw_facts:
        ev_text = item.get("evidence_text", "").strip()
        clean_ev = re.sub(r'\s+', ' ', ev_text)
        clean_doc = re.sub(r'\s+', ' ', text)

        if not clean_ev or clean_ev not in clean_doc:
            print(f"❌ REJECTED (hallucination — evidence not found verbatim): {ev_text[:50]}...")
            continue

        if not has_quantifiable_content(item):
            print(f"❌ REJECTED (no quantifiable content — likely a heading/title): {ev_text[:50]}...")
            continue

        rects = page.search_for(ev_text[:50])
        bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1] if rects else None

        embed_repr = f"{item['subject']} | {item['fact_type']} | {item['metric_value'].get('raw_value', '')}"
        vector = list(embed_model.embed([embed_repr]))[0].tolist()

        item["evidence_bbox"] = bbox
        item["evidence_page"] = page_num
        item["embedding"] = vector
        verified_facts.append(item)

    return verified_facts