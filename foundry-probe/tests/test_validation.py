from demo.validation import validate_payload

def test_missing_required():
    schema = {"schema": {"required": ["mechanism"]}}
    extracted = {}
    cautions = validate_payload(extracted, {}, "", schema)
    assert any("Missing required" in c for c in cautions)

def test_icd10_format():
    reasoning = {"icd10": [{"code": "123", "evidence_sentences": ["test"]}]}
    cautions = validate_payload({}, reasoning, "", None)
    assert any("Invalid ICD-10 format" in c for c in cautions)
    
    reasoning2 = {"icd10": [{"code": "E11.9", "evidence_sentences": ["test"]}]}
    cautions2 = validate_payload({}, reasoning2, "", None)
    assert not any("Invalid ICD-10 format" in c for c in cautions2)

def test_laterality_mismatch():
    reasoning = {
        "final_diagnosis": "Left arm pain",
        "rationale": "Patient injured right arm"
    }
    cautions = validate_payload({}, reasoning, "", None)
    assert any("Laterality mismatch" in c for c in cautions)

def test_history_cross_check():
    reasoning = {"icd10": []}
    cautions = validate_payload({}, reasoning, "Patient has history of diabetes", None)
    assert any("no ICD-10 codes generated" in c for c in cautions)
