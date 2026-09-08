import os
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from pydantic import BaseModel, ValidationError

from app.rate_limiter import groq_rate_limiter, is_rate_limit_error, backoff_delay

api_key = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=api_key)


class AdjudicationResult(BaseModel):
    relation_type: str  # corroborates | contradicts | reconciled_by_context | insufficient_evidence
    explanation: str
    confidence: float = 0.5


VALID_RELATION_TYPES = {"corroborates", "contradicts", "reconciled_by_context", "insufficient_evidence"}


def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (s or "").lower())


def is_near_identical(text_a: str, text_b: str) -> bool:
    """
    Detects structural duplicates (e.g. a TOC entry and its matching chapter-start
    heading) that aren't genuine cross-document facts worth adjudicating.
    """
    norm_a, norm_b = _normalize(text_a), _normalize(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    shorter, longer = (norm_a, norm_b) if len(norm_a) <= len(norm_b) else (norm_b, norm_a)
    if len(shorter) > 8 and shorter in longer:
        return True
    return False


def _fallback_result(reason: str) -> dict:
    return {"relation_type": "insufficient_evidence", "explanation": reason, "confidence": 0.0}


def adjudicate_pair(fact1: dict, fact2: dict, doc1_name: str, doc2_name: str) -> dict:
    if is_near_identical(fact1.get("evidence_text", ""), fact2.get("evidence_text", "")):
        return _fallback_result(
            "Skipped: evidence text is structurally near-identical (likely a duplicate "
            "heading/section reference, not a comparable fact)."
        )

    prompt = f"""You are an expert financial and economic adjudicator.
    Compare these two extracted facts from different documents to see if they relate to each other.

    FACT A (Source: {doc1_name}, Page: {fact1['evidence_page']}):
    Subject: {fact1['subject']}
    Fact Type: {fact1['fact_type']}
    Metric: {fact1.get('metric_value', {})}
    Context: {fact1.get('scope_context', {})}
    Verbatim Quote: "{fact1['evidence_text']}"

    FACT B (Source: {doc2_name}, Page: {fact2['evidence_page']}):
    Subject: {fact2['subject']}
    Fact Type: {fact2['fact_type']}
    Metric: {fact2.get('metric_value', {})}
    Context: {fact2.get('scope_context', {})}
    Verbatim Quote: "{fact2['evidence_text']}"

    Classify the relationship into exactly one of these four categories:
    1. "corroborates": The facts state the same or highly consistent claims (allowing for minor rounding).
    2. "contradicts": The facts present a genuine numerical or factual conflict for the exact same entity and timeframe.
    3. "reconciled_by_context": The numbers differ, but it is explained by different time periods (e.g., Q3 vs Full Year), methodologies (e.g., Reported vs Adjusted), or units.
    4. "insufficient_evidence": The facts are completely unrelated or cannot be logically compared.

    Analyze carefully and provide an explanation.

    Respond ONLY with a JSON object of this exact shape:
    {{
      "relation_type": "one of: corroborates, contradicts, reconciled_by_context, insufficient_evidence",
      "explanation": "concise explanation; if reconciled, state exactly what contextual difference explains the gap",
      "confidence": 0.0
    }}
    """

    max_retries = 5
    response = None
    for attempt in range(max_retries):
        try:
            with groq_rate_limiter:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            break
        except Exception as e:
            rate_limited = is_rate_limit_error(e)
            delay = backoff_delay(attempt, rate_limited)
            reason = "rate limit" if rate_limited else "API error"
            print(f"⚠️ Reconciler {reason} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            print(f"   Backing off {delay:.1f}s...")
            time.sleep(delay)
            if attempt == max_retries - 1:
                return _fallback_result("Failed due to persistent API limits.")

    if not response or not response.choices:
        return _fallback_result("Empty response from Groq.")

    raw_content = response.choices[0].message.content

    try:
        result = AdjudicationResult.model_validate_json(raw_content)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Adjudication schema validation failed: {e}")
        return _fallback_result("Failed to validate LLM response against expected schema.")

    if result.relation_type not in VALID_RELATION_TYPES:
        print(f"⚠️ Unexpected relation_type from model: {result.relation_type}")
        return _fallback_result(f"Model returned unrecognized relation_type: {result.relation_type}")

    return result.model_dump()