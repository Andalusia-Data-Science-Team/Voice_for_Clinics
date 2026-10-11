import asyncio
import json
from config import client, DEPLOYMENTS, budget_manager

SCHEMAS = {
    "ER": {
        "name": "er_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "triage_category": {"type": "string", "enum": ["Red", "Yellow", "Green", "Unknown"]},
                "mechanism_of_injury": {"type": "string"},
                "consent": {"type": "string", "enum": ["Yes", "No", "Verbal", "Not Applicable"]}
            },
            "required": ["triage_category", "mechanism_of_injury", "consent"],
            "additionalProperties": False
        }
    },
    "OPD": {
        "name": "opd_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "follow_up_weeks": {"type": "integer"},
                "referral_needed": {"type": "boolean"},
                "vitals_checked": {"type": "boolean"}
            },
            "required": ["follow_up_weeks", "referral_needed", "vitals_checked"],
            "additionalProperties": False
        }
    },
    "Dental": {
        "name": "dental_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tooth_number": {"type": "string"},
                "procedure": {"type": "string", "enum": ["Extraction", "Filling", "Cleaning", "Root Canal", "Other"]},
                "anesthesia_used": {"type": "boolean"}
            },
            "required": ["tooth_number", "procedure", "anesthesia_used"],
            "additionalProperties": False
        }
    }
}

# The transcript doesn't mention triage, follow up, or dental procedures.
# Models should leave strings empty, ints as 0 (or nullable if allowed by strict, but strict requires types), 
# or use default/Unknown enums if they don't hallucinate.
# Wait, strict JSON schemas don't easily allow nulls unless defined as ["string", "null"]. 
# Let's update schemas to allow nulls to properly test if the model leaves unmentioned fields null rather than hallucinating.

for s in SCHEMAS.values():
    for prop, details in s["schema"]["properties"].items():
        if "type" in details and not isinstance(details["type"], list):
            details["type"] = [details["type"], "null"]

TRANSCRIPT = "Patient arrived with left knee pain after a fall. No past medical history. Given paracetamol."

async def test_schema(deployment, schema_name, schema_def):
    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Extract the information into the JSON schema. If a value is not mentioned in the transcript, you MUST output null."},
                {"role": "user", "content": TRANSCRIPT}
            ],
            response_format={"type": "json_schema", "json_schema": schema_def},
            temperature=0.0
        )
        budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)
        
        content = response.choices[0].message.content
        parsed = json.loads(content)
        
        # Check hallucination: basically everything except mechanism_of_injury should be null
        hallucinated = []
        for k, v in parsed.items():
            if k != "mechanism_of_injury" and v is not None:
                hallucinated.append(f"{k}={v}")
                
        return {
            "valid_json": True,
            "parsed": parsed,
            "hallucinated": hallucinated
        }
        
    except Exception as e:
        return {
            "valid_json": False,
            "error": str(e)
        }

async def run():
    print("--- Strict JSON Schema Extraction ---")
    results = {}
    for deployment in DEPLOYMENTS:
        print(f"\nTesting {deployment}...")
        results[deployment] = {}
        for s_name, s_def in SCHEMAS.items():
            res = await test_schema(deployment, s_name, s_def)
            results[deployment][s_name] = res
            print(f"  {s_name}: Valid={res.get('valid_json')}, Hallucinations={res.get('hallucinated', [])}")
            if not res.get("valid_json"):
                print(f"    Error: {res.get('error')}")

    with open("results/strict_json.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
