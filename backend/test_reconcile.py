import json
from app.reconciler import adjudicate_pair

# Fact A: India Economic Survey 2024-25 baseline projection
fact_survey = {
    "evidence_page": 14,
    "subject": "Real GDP Growth FY25",
    "fact_type": "Macroeconomic",
    "metric_value": {"raw_value": "6.5% to 7.0%"},
    "scope_context": {"period": "FY25"},
    "evidence_text": "The Survey conservatively projects real GDP growth of 6.5–7.0 per cent in FY25.",
}

# Fact B: A direct conflicting macroeconomic assessment
fact_imf = {
    "evidence_page": 28,
    "subject": "Real GDP Growth FY25",
    "fact_type": "Macroeconomic",
    "metric_value": {"raw_value": "5.8%"},
    "scope_context": {"period": "FY25"},
    "evidence_text": "Real GDP growth for the Indian economy in FY25 is projected to decelerate sharply to 5.8 per cent.",
}

result = adjudicate_pair(
    fact1=fact_survey,
    fact2=fact_imf,
    doc1_name="01-india-economic-survey-2024-25-excerpt.pdf",
    doc2_name="03-imf-article-iv-consultation.pdf",
)

print(json.dumps(result, indent=2))