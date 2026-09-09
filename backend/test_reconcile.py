import json
from app.reconciler import adjudicate_pair

fact_survey = {
    "evidence_page": 15,
    "subject": "services sector growth",
    "fact_type": "Macroeconomic",
    "metric_value": {"raw_value": "7.2 per cent"},
    "scope_context": {"period": "FY25"},
    "evidence_text": "Growth in the services sector is expected to remain robust at 7.2 per cent",
}

fact_rbi = {
    "evidence_page": 9,
    "subject": "Services sector growth",
    "fact_type": "Macroeconomic",
    "metric_value": {"raw_value": "7.5 per cent"},
    "scope_context": {"period": "2024-25"},
    "evidence_text": "services sector, with a share of 64.1 per cent in GVA, remained the mainstay of aggregate supply with a growth of 7.5 per cent in 2024-25.",
}

result = adjudicate_pair(
    fact1=fact_survey,
    fact2=fact_rbi,
    doc1_name="01-india-economic-survey-2024-25-excerpt.pdf",
    doc2_name="02-rbi-annual-report-2024-25-excerpt.pdf",
)

print(json.dumps(result, indent=2))
