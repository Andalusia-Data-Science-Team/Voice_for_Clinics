import asyncio
import json
import time
import re
from config import client, DEPLOYMENTS, budget_manager

NOTES = [
    {"id": "diabetes", "text": "Patient has a 10-year history of Type 2 Diabetes. Current HbA1c is 9.5%. Complains of numbness in both feet. Blood sugar logs show frequent highs."},
    {"id": "chest_pain", "text": "55yo male presents with crushing central chest pain radiating to the left arm, started 30 mins ago. Diaphoretic. ECG shows ST elevation in V1-V4."},
    {"id": "otitis_media", "text": "3yo child with fever 39C and tugging at the right ear for 2 days. Tympanic membrane is bulging and erythematous on the right."},
    {"id": "knee_injury", "text": "22yo playing football, twisted right knee. Heard a pop. Swelling developed within 1 hour. Positive Lachman test."},
    {"id": "pneumonia", "text": "68yo with productive cough, rusty sputum, fever, and chills for 4 days. Crackles heard in the right lower lobe. SatO2 91% on room air."}
]

PROMPT = """Analyze the clinical note. Return ONLY a JSON object (no markdown, no backticks).
State the exact ICD-10-AM codes.
Schema:
{
  "icd10": [{"code": "string", "description": "string", "evidence_sentences": ["string"]}],
  "final_diagnosis": "string",
  "rationale": "string",
  "clinical_gap_questions": ["string"]
}
"""

ICD_REGEX = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")

async def test_reasoning(deployment, note):
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": note["text"]}
            ],
            response_format={"type": "json_object"}
        )
        latency = time.perf_counter() - start
        
        usage = response.usage
        budget_manager.add_cost(deployment, usage.prompt_tokens, usage.completion_tokens)
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Validation
        valid_format = True
        codes = []
        for item in parsed.get("icd10", []):
            code = item.get("code", "")
            codes.append(code)
            if not ICD_REGEX.match(code):
                valid_format = False
            if not item.get("evidence_sentences"):
                valid_format = False
                
        return {
            "success": True,
            "latency": latency,
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
            "valid_format": valid_format,
            "codes": codes,
            "parsed": parsed
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.perf_counter() - start,
            "error": str(e)
        }

async def run():
    print("--- Diagnostic Reasoning ---")
    results = {}
    
    # We compare Terra, Sol, Luna (skip gpt-4o-mini maybe, or run on all available)
    reasoning_deployments = [d for d in DEPLOYMENTS if 'gpt-4o-mini' not in d.lower()]
    
    for deployment in reasoning_deployments:
        print(f"\nTesting {deployment}...")
        results[deployment] = {}
        for note in NOTES:
            res = await test_reasoning(deployment, note)
            results[deployment][note["id"]] = res
            
            if res.get("success"):
                print(f"  {note['id']}: {res['latency']:.2f}s | Tokens: {res['tokens']['completion']} | Valid: {res['valid_format']} | Codes: {res['codes']}")
            else:
                print(f"  {note['id']}: FAILED ({res.get('error')})")

    with open("results/reasoning.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
