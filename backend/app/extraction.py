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

from app.rate_limiter import groq_rate_limiter, is_rate_limit_error, backoff_delay

print("Loading Embedding Model...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from backend/.env! Please add it.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=api_key)


# --- Pydantic validation layer -------------------------------------------------
# Groq's JSON mode guarantees valid JSON, not your exact structure. This is the
# schema-enforcement Gemini's response_schema gave you for free — anything that
# doesn't match gets rejected here, before it ever reaches the evidence/quantifiable
# content checks below.

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


def _call_groq_with_backoff(prompt: str, max_retries: int = 5):
    """
    Shared call path: acquires the process-wide rate-limit slot before every
    request, and applies exponential backoff (longer specifically for 429s).
    """
    for attempt in range(max_retries):
        try:
            with groq_rate_limiter:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            return response
        except Exception as e:
            rate_limited = is_rate_limit_error(e)
            delay = backoff_delay(attempt, rate_limited)
            reason = "rate limit" if rate_limited else "API error"
            print(f"⚠️ Groq {reason} (attempt {attempt + 1}/{max_retries}). Backing off {delay:.1f}s...")
            time.sleep(delay)
            if attempt == max_retries - 1:
                print(f"❌ Giving up after {max_retries} attempts: {e}")
                return None
    return None


def process_pdf_page(pdf_path: str, page_num: int):
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text")

    if len(text.strip()) < 100:
        return []

    if looks_like_toc_or_cover(text):
        print(f"⏭️  Skipping page {page_num} (looks like TOC/cover, no LLM call made)")
        return []

    # Groq's json_object mode requires the word "JSON" to appear in the prompt,
    # and expects a top-level object (not a bare array) — hence the "facts" wrapper.
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
meeting the above bar, return an empty facts array.

The evidence_text MUST be an exact, word-for-word substring from the text below. Do not paraphrase.

Respond ONLY with a JSON object of this exact shape:
{{
  "facts": [
    {{
      "subject": "string",
      "fact_type": "string, e.g. Financial, Macroeconomic, Operational",
      "metric_value": {{"raw_value": "the exact number or metric as a string"}},
      "scope_context": {{"period": "e.g. FY24, Q1 2023, or null"}},
      "evidence_text": "exact verbatim quote from the page text",
      "confidence": 0.0
    }}
  ]
}}

PAGE TEXT:
{text}
"""

    response = _call_groq_with_backoff(prompt)

    if not response or not response.choices:
        print(f"Skipping page {page_num} due to persistent API errors.")
        return []

    raw_content = response.choices[0].message.content

    try:
        parsed = ExtractionResponse.model_validate_json(raw_content)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"⚠️ Schema validation failed on page {page_num}, attempting loose parse: {e}")
        try:
            loose = json.loads(raw_content)
            parsed = ExtractionResponse.model_validate(loose)
        except Exception as e2:
            print(f"❌ Could not parse Groq response on page {page_num}: {e2}")
            return []

    verified_facts = []

    for item in parsed.facts:
        ev_text = item.evidence_text.strip()
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