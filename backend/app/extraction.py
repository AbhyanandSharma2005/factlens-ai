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

from app.rate_limiter import groq_rate_limiter, is_rate_limit_error, is_retryable_error, backoff_delay

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


def _parse_facts_resilient(raw_content: str, page_num: int) -> List[ExtractedFact]:
    """
    Validate each fact in the response individually instead of as one batch.
    """
    try:
        raw = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"❌ Could not parse Groq response on page {page_num} as JSON: {e}")
        return []

    raw_facts = raw.get("facts", []) if isinstance(raw, dict) else []
    if not isinstance(raw_facts, list):
        return []

    good_facts = []
    for i, item in enumerate(raw_facts):
        try:
            good_facts.append(ExtractedFact.model_validate(item))
        except ValidationError as e:
            print(f"⚠️ Dropping malformed fact #{i} on page {page_num} (kept the rest): {e}")

    return good_facts


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
    request, and applies exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            with groq_rate_limiter:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
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
    
    # Extract by 'blocks' (paragraphs) to filter individually
    blocks = page.get_text("blocks")
    
    filtered_text_chunks = []
    for b in blocks:
        block_text = b[4].strip()
        
        # Zero-Token Guardrail: If a paragraph lacks digits, drop it to save tokens
        if re.search(r'\d', block_text) and len(block_text) > 20:
            filtered_text_chunks.append(block_text)
            
    filtered_text = "\n\n".join(filtered_text_chunks)

    # Skip LLM call if negligible numeric content remains or it looks like TOC/cover
    if len(filtered_text) < 100 or looks_like_toc_or_cover(filtered_text):
        print(f"⏭️  Skipping page {page_num} (No numeric data or looks like TOC/cover)")
        return []

    # Optimized Minimized Prompt
    prompt = f"""Extract 1 to 4 verifiable quantitative facts from this text. 
A valid fact MUST contain a specific number, statistic, percentage, or date.
Do NOT extract general text, chapter titles, or boilerplate.
The evidence_text MUST be an exact, word-for-word substring from the text below.

Respond ONLY with a JSON object of this exact shape:
{{
  "facts": [
    {{
      "subject": "string",
      "fact_type": "string (e.g. Financial, Macroeconomic)",
      "metric_value": {{"raw_value": "exact number string"}},
      "scope_context": {{"period": "e.g. FY24 or null"}},
      "evidence_text": "exact verbatim quote",
      "confidence": 0.9
    }}
  ]
}}

PAGE TEXT:{filtered_text}
"""

    response = _call_groq_with_backoff(prompt)

    if not response or not response.choices:
        print(f"Skipping page {page_num} due to persistent API errors.")
        return []

    raw_content = response.choices[0].message.content
    facts = _parse_facts_resilient(raw_content, page_num)

    verified_facts = []
    
    # Use full original page text for verbatim hallucination matching
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

    return verified_facts