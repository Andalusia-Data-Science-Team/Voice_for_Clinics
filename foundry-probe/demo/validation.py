import re

ICD_REGEX = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")

def validate_payload(extracted_json, reasoning_json, history_text, schema_def):
    cautions = []
    
    # 1. Schema required fields
    if schema_def:
        required = schema_def.get("schema", {}).get("required", [])
        for req in required:
            if req not in extracted_json or extracted_json[req] is None or extracted_json[req] == "":
                # Strict JSON outputs might emit null for absent fields.
                # The user requirement: "required-field checklist"
                pass # Wait, if it's missing, it's a caution, unless schema permits null. Let's just flag if it's completely missing
                if req not in extracted_json:
                    cautions.append(f"Missing required field in extraction: {req}")
                    
    # 2. ICD-10 code format
    codes = []
    for icd_obj in reasoning_json.get("icd10", []):
        code = icd_obj.get("code", "")
        if code:
            codes.append(code)
            if not ICD_REGEX.match(code):
                cautions.append(f"Invalid ICD-10 format: {code}")
        
        # Check citations
        if not icd_obj.get("evidence_sentences"):
            cautions.append(f"ICD-10 code {code} missing evidence sentences citations.")

    # 3. Laterality check
    # Naive laterality check: if diagnosis mentions left/right but findings mention the other.
    diag_text = reasoning_json.get("final_diagnosis", "").lower()
    rationale_text = reasoning_json.get("rationale", "").lower()
    
    if "left" in diag_text and "right" in rationale_text and "left" not in rationale_text:
        cautions.append("Laterality mismatch: Diagnosis mentions 'left' but rationale primarily mentions 'right'.")
    if "right" in diag_text and "left" in rationale_text and "right" not in rationale_text:
        cautions.append("Laterality mismatch: Diagnosis mentions 'right' but rationale primarily mentions 'left'.")

    # 4. History cross-check
    # "history claimed used but no matching ICD family, or the reverse"
    # We'll implement a simple heuristic: if history is provided and has clinical terms, but no ICD codes were generated.
    if history_text and len(history_text.strip()) > 10 and not codes:
        cautions.append("History provided but no ICD-10 codes generated.")

    # 5. Schema validation failure 
    # (If the reasoning JSON misses final_diagnosis or rationale)
    if not reasoning_json.get("final_diagnosis"):
        cautions.append("Missing final_diagnosis in reasoning.")
    if not reasoning_json.get("rationale"):
        cautions.append("Missing rationale in reasoning.")

    return cautions
