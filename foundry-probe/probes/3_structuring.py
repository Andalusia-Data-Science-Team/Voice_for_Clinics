import asyncio
import json
import re
from config import client, DEPLOYMENTS, budget_manager

STRUCTURING_PROMPT = """You are a medical transcription structurer.
Rules:
1. Translate Arabic or mixed Arabic/English to English.
2. Organise into sections: Complaint and History, Examination, Assessment, Plan.
3. NEVER add, infer, or sharpen clinical detail.
4. Keep negation, laterality, doses, durations EXACTLY as stated.
5. Mark unclear speech as [unclear], never guess.
"""

def evaluate_output(output, case):
    output_lower = output.lower()
    passed = True
    errors = []
    
    if "must_contain_all" in case:
        for condition in case["must_contain_all"]:
            options = [opt.strip() for opt in condition.lower().split("|")]
            if not any(opt in output_lower for opt in options):
                passed = False
                errors.append(f"Missing one of: {options}")
                
    if "must_contain_any" in case:
        options = [opt.strip().lower() for opt in case["must_contain_any"]]
        if not any(opt in output_lower for opt in options):
            passed = False
            errors.append(f"Missing at least one of: {options}")
            
    if "must_not_contain" in case:
        for condition in case["must_not_contain"]:
            options = [opt.strip() for opt in condition.lower().split("|")]
            if any(opt in output_lower for opt in options):
                passed = False
                errors.append(f"Forbidden phrase found: {options}")
                
    return passed, errors

async def test_case(deployment, case):
    results = []
    for i in range(3):
        try:
            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": STRUCTURING_PROMPT},
                    {"role": "user", "content": case["text"]}
                ],
                temperature=0.0  # Zero temperature for fidelity
            )
            
            budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            output = response.choices[0].message.content
            passed, errors = evaluate_output(output, case)
            results.append({
                "iteration": i + 1,
                "output": output,
                "passed": passed,
                "errors": errors
            })
        except Exception as e:
            results.append({
                "iteration": i + 1,
                "error": str(e),
                "passed": False,
                "errors": [str(e)]
            })
    return results

async def run():
    print("--- Structuring & Extraction Fidelity ---")
    with open("cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    final_results = {}
    
    for deployment in DEPLOYMENTS:
        print(f"Testing {deployment}...")
        final_results[deployment] = {}
        for case in cases:
            res = await test_case(deployment, case)
            final_results[deployment][case["id"]] = res
            
            passes = sum(1 for r in res if r.get("passed"))
            print(f"  Case {case['id']}: {passes}/3 passed")
            if passes < 3:
                for r in res:
                    if not r.get("passed"):
                        print(f"    Error: {r.get('errors')}")
                        
    with open("results/structuring_fidelity.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
