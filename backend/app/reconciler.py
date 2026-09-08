import os
import json
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

ADJUDICATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "relation_type": {
            "type": "STRING",
            "description": "Must be exactly one of: 'corroborates', 'contradicts', 'reconciled_by_context', 'insufficient_evidence'"
        },
        "explanation": {
            "type": "STRING",
            "description": "Concise explanation of why this relationship holds. If reconciled, state exactly what contextual difference explains the gap."
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence score between 0.0 and 1.0"
        }
    },
    "required": ["relation_type", "explanation", "confidence"]
}

def adjudicate_pair(fact1: dict, fact2: dict, doc1_name: str, doc2_name: str) -> dict:
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
    """
    
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ADJUDICATION_SCHEMA,
            temperature=0.0
        )
    )
    
    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"Adjudication Parsing Error: {e}")
        return {
            "relation_type": "insufficient_evidence", 
            "explanation": "Failed to parse LLM response.", 
            "confidence": 0.0
        }