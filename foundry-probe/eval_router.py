import asyncio
import json
import os
import sys

# Ensure demo components can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo.pipeline import run_structuring, run_reasoning
from demo.router import evaluate_triggers, determine_initial_tier
from config import TIER_SIMPLE, TIER_DEFAULT, TIER_ESCALATE

# Generate 12 synthetic cases
SYNTHETIC_CASES = [
    # 4 Simple
    {"id": "syn_simple_1", "text": "Patient with mild headache, paracetamol prescribed."},
    {"id": "syn_simple_2", "text": "Routine checkup for dental cleaning. No issues."},
    {"id": "syn_simple_3", "text": "Patient has a small scratch on left arm. Cleaned and bandaged."},
    {"id": "syn_simple_4", "text": "Complains of mild ear ache for 1 day. Prescribed drops."},
    # 4 Moderate (e.g., 3+ problems, or elderly, or 5 meds)
    {"id": "syn_mod_1", "text": "65yo elderly patient with hypertension, diabetes, and hyperlipidemia. Taking 5 medications. Complains of knee pain."},
    {"id": "syn_mod_2", "text": "Pregnant female presents with severe nausea and vomiting for 2 days."},
    {"id": "syn_mod_3", "text": "Patient has cough, fever, and shortness of breath. Prescribed 3 new meds on top of 2 existing ones."},
    {"id": "syn_mod_4", "text": "ER patient with abnormal primary survey, tachycardia, but stable now after fluids."},
    # 4 Complex (Red flag + complexity)
    {"id": "syn_comp_1", "text": "Patient has severe chest pain, radiating to left arm. Also has diabetes and hypertension."},
    {"id": "syn_comp_2", "text": "Child with suspected sepsis, fever 40C, lethargic. ER abnormal primary survey."},
    {"id": "syn_comp_3", "text": "Loss of consciousness after car accident. Elderly patient with multiple comorbidities."},
    {"id": "syn_comp_4", "text": "Pregnant female with severe bleeding and severe abdominal pain."}
]

async def eval_case(case, mode="difficulty"):
    print(f"Evaluating {case['id']}...")
    try:
        struct_text, signals, _ = await run_structuring(case['text'])
        triggers = evaluate_triggers(signals, "")
        tier, _ = determine_initial_tier(mode, triggers, False)
        
        return {
            "id": case["id"],
            "tier": tier,
            "triggers": triggers,
            "success": True
        }
    except Exception as e:
        print(f"Failed {case['id']}: {e}")
        return {
            "id": case["id"],
            "tier": "Error",
            "triggers": [],
            "success": False,
            "error": str(e)
        }

async def run():
    print("--- Router Evaluation ---")
    with open("cases.json", "r") as f:
        cases = json.load(f)
        
    all_cases = cases + SYNTHETIC_CASES
    
    results = []
    distribution = {TIER_SIMPLE: 0, TIER_DEFAULT: 0, TIER_ESCALATE: 0}
    
    for case in all_cases:
        res = await eval_case(case)
        results.append(res)
        distribution[res["tier"]] = distribution.get(res["tier"], 0) + 1
        
    print("\nTier Distribution (Difficulty Mode):")
    for k, v in distribution.items():
        print(f"  {k}: {v}/{len(all_cases)}")
        
    # Example of compare mode on a single case
    print("\nCompare Mode (syn_comp_1):")
    case = next(c for c in SYNTHETIC_CASES if c["id"] == "syn_comp_1")
    try:
        struct_text, _, _ = await run_structuring(case["text"])
        
        res_simple = await run_reasoning(struct_text, "", TIER_SIMPLE)
        res_default = await run_reasoning(struct_text, "", TIER_DEFAULT)
        res_escalate = await run_reasoning(struct_text, "", TIER_ESCALATE)
        
        print("  Luna Diagnosis: ", res_simple[0].get("final_diagnosis"))
        print("  Terra Diagnosis: ", res_default[0].get("final_diagnosis"))
        print("  Sol Diagnosis: ", res_escalate[0].get("final_diagnosis"))
    except Exception as e:
        print(f"  Compare Mode failed due to API error: {e}")

if __name__ == "__main__":
    asyncio.run(run())
