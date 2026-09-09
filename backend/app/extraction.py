import os
import json
import re
import time
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

import pymupdf as fitz
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from fastembed import TextEmbedding

from app.rate_limiter import groq_rate_limiter, strict_pacer, is_rate_limit_error, is_retryable_error, backoff_delay
from app.cache import get_text_hash, check_fact_cache, save_to_fact_cache

print("Loading Embedding Model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from backend/.env! Please add it.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=api_key)


# --- Pydantic validation layer -------------------------------------------------

class MetricValue(BaseModel):
    raw_value: str


class ScopeContext(BaseModel):
    period: Optional[str] = None


class ExtractedFact(BaseModel):
    subject: str
    fact_type: str
    metric_value: MetricValue
    scope_context: ScopeContext = Field(default_factory=ScopeContext)
    evidence_text: str
    confidence: float = 0.5


class ExtractionResponse(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list)


def extract_json_from_response(raw_text: str) -> dict:
    """
    Defensive parser: Extracts JSON from raw LLM output, handling markdown 
    code blocks, conversational text, and minor formatting flaws.
    """
    if not raw_text:
        return {"facts": []}

    # 1. Try direct parsing if the model was clean
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. Extract content wrapped in markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Fallback: Find the first opening brace '{' and last closing brace '}'
    brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    # If all parsing strategies fail, return an empty structure safely instead of crashing
    print(f"⚠️ Failed to parse LLM output as JSON. Raw text was: {raw_text[:100]}...")
    return {"facts": []}


def _parse_facts_resilient(raw_content: str, page_num: int) -> List[ExtractedFact]:
    # Pass through our defensive parser instead of trusting raw json.loads
    parsed_data = extract_json_from_response(raw_content)
    
    raw_facts_list = parsed_data.get("facts", []) if isinstance(parsed_data, dict) else []
    if not isinstance(raw_facts_list, list):
        return []

    valid_facts = []

    for item in raw_facts_list:
        try:
            # Validate individual fact schema via Pydantic
            fact_obj = ExtractedFact.model_validate(item)
            valid_facts.append(fact_obj)
        except ValidationError as e:
            print(f"⚠️ Dropped malformed fact on page {page_num}: {e} | Item: {item}")
            continue

    return valid_facts


def looks_like_toc_or_cover(text: str) -> bool:
    """Cheap structural heuristic to skip TOC/cover/divider pages before an LLM call."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True

    short_line_ratio = sum(1 for l in lines if len(l) < 60) / len(lines)
    digit_lines = sum(1 for l in lines if any(c.isdigit() for c in l))
    digit_line_ratio = digit_lines / len(lines)

    return short_line_ratio > 0.8 and digit_line_ratio < 0.3


def has_quantifiable_content(item: ExtractedFact) -> bool:
    """Guards against 'facts' that are really headings with no number/date/statistic."""
    return bool(re.search(r'\d', item.metric_value.raw_value)) or bool(re.search(r'\d', item.evidence_text))


def _normalize_for_match(s: str) -> str:
    """
    Normalize text before the verbatim-substring hallucination check.
    """
    if not s:
        return ""
    s = s.replace('\u2018', "'").replace('\u2019', "'")   # curly single quotes
    s = s.replace('\u201c', '"').replace('\u201d', '"')   # curly double quotes
    s = s.replace('\u2013', '-').replace('\u2014', '-')   # en dash / em dash
    s = s.replace('\u00a0', ' ')                          # non-breaking space
    s = s.replace('\ufb01', 'fi').replace('\ufb02', 'fl')  # common ligatures
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _call_groq_with_backoff(prompt: str, max_retries: int = 5):
    """
    Shared call path: acquires the process-wide rate-limit slot before every
    request, and applies exponential backoff for transient errors.
    """
    for attempt in range(max_retries):
        try:
            with groq_rate_limiter:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise data extraction engine. You always output valid JSON representing the requested schema."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
            return response
        except Exception as e:
            if not is_retryable_error(e):
                print(f"❌ Non-retryable error, failing fast: {type(e).__name__}: {e}")
                return None

            rate_limited = is_rate_limit_error(e)
            delay = backoff_delay(attempt, rate_limited)
            reason = "rate limit" if rate_limited else "API error"
            print(f"⚠️ Groq {reason} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            print(f"   Backing off {delay:.1f}s...")
            time.sleep(delay)
            if attempt == max_retries - 1:
                print(f"❌ Giving up after {max_retries} attempts: {e}")
                return None
    return None


def process_pdf_page(pdf_path: str, page_num: int):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    
    # 1. Block-level parsing to isolate text paragraphs
    blocks = page.get_text("blocks")
    filtered_chunks = []
    
    for b in blocks:
        block_text = b[4].strip()
        # Heuristic: If a paragraph has no numbers or is too short, discard it locally (0 tokens)
        if re.search(r'\d', block_text) and len(block_text) > 20:
            filtered_chunks.append(block_text)
            
    filtered_text = "\n\n".join(filtered_chunks)

    # Skip empty or cover-like pages locally
    if len(filtered_text) < 100 or looks_like_toc_or_cover(filtered_text):
        print(f"⏭️ Skipping page {page_num} locally (No numeric data/TOC)")
        return []

    # 2. Check Content-Addressable Cache (0 API calls if text was processed before)
    text_hash = get_text_hash(filtered_text)
    cached_result = check_fact_cache(text_hash)
    if cached_result is not None:
        print(f"⚡ Cache hit for page {page_num}! Skipping LLM call.")
        return cached_result

    # 3. Proactive Rate Limiting (Pace requests mathematically)
    strict_pacer.acquire()

    prompt = f"""Extract 1 to 4 verifiable quantitative facts from this text. 
A valid fact MUST contain a specific number, statistic, percentage, or date.
Respond ONLY with a JSON object:
{{
  "facts": [
    {{
      "subject": "string",
      "fact_type": "string",
      "metric_value": {{"raw_value": "string"}},
      "scope_context": {{"period": "string or null"}},
      "evidence_text": "exact verbatim quote",
      "confidence": 0.9
    }}
  ]
}}

TEXT:{filtered_text}
"""

    response = _call_groq_with_backoff(prompt)

    if not response or not response.choices:
        print(f"Skipping page {page_num} due to persistent API errors.")
        return []

    raw_content = response.choices[0].message.content
    facts = _parse_facts_resilient(raw_content, page_num)

    verified_facts = []
    raw_page_text = page.get_text("text")

    for item in facts:
        ev_text = item.evidence_text.strip()
        clean_ev = _normalize_for_match(ev_text)
        clean_doc = _normalize_for_match(raw_page_text)

        if not clean_ev or clean_ev not in clean_doc:
            print(f"❌ REJECTED (hallucination): {ev_text[:50]}...")
            continue

        if not has_quantifiable_content(item):
            print(f"❌ REJECTED (no quantifiable content): {ev_text[:50]}...")
            continue

        rects = page.search_for(ev_text[:50])
        bbox = [rects[0].x0, rects[0].y0, rects[0].x1, rects[0].y1] if rects else None

        embed_repr = f"{item.subject} | {item.fact_type} | {item.metric_value.raw_value}"
        vector = list(embed_model.embed([embed_repr]))[0].tolist()

        verified_facts.append({
            "subject": item.subject,
            "fact_type": item.fact_type,
            "metric_value": item.metric_value.model_dump(),
            "scope_context": item.scope_context.model_dump(),
            "evidence_text": item.evidence_text,
            "confidence": item.confidence,
            "evidence_bbox": bbox,
            "evidence_page": page_num,
            "embedding": vector,
        })

    # 4. Save successful extraction to cache for future runs
    if verified_facts:
        save_to_fact_cache(text_hash, verified_facts)

    return verified_facts