from app.extraction import extract_json_from_response

# Simulate messy LLM outputs containing markdown and conversational text
messy_outputs = [
    'Here is the JSON:\n```json\n{"facts": [{"subject": "GDP", "fact_type": "Macro", "metric_value": {"raw_value": "7%"}, "scope_context": {"period": "FY25"}, "evidence_text": "growth of 7%", "confidence": 0.9}]}\n```',
    'Sure, I can help!\n{"facts": [{"subject": "Inflation", "fact_type": "Macro", "metric_value": {"raw_value": "4.5%"}, "scope_context": {"period": "FY25"}, "evidence_text": "inflation at 4.5%", "confidence": 0.8}]}',
    'Invalid junk text from model'
]

for i, output in enumerate(messy_outputs):
    result = extract_json_from_response(output)
    print(f"Test case {i+1} parsed facts count:", len(result.get("facts", [])))