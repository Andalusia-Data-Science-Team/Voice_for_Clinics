# Production Gold Set Extraction

This folder contains clinical cases extracted from production logs (filtering out internal `localhost`, `127.0.0.1`, and Docker private networks) from [usage.jsonl](file:///home/ai/Workspace/Doha/MedVoice/src/logs/usage.jsonl).

## Specialty Excel Workbooks (Annotation-Ready Template)

Each Excel sheet has been prepared for manual ground-truth labeling:
1. **Column A**: `Clinical Note` (the raw dictation / dialogue to be evaluated)
2. **Column B**: `Justification / Clinician Review (Accept/Reject Reason)` (blank for reviewer justification)
3. **Columns C-E**: `Case ID`, `Language`, `Encounter Mode`
4. **All Subsequent Columns**: Department-specific **Expected Output Schema** headers (left **blank** for the medical annotation team to fill as ground truth).

| Workbook | Cases | Schema Headers Included (Ready to Fill) |
| :--- | :--- | :--- |
| [**`production_gold_set_by_specialty_master.xlsx`**](file:///home/ai/Workspace/Doha/MedVoice/data/production_gold_set/production_gold_set_by_specialty_master.xlsx) | 171 | Master workbook with tabs for OPD, ER, and Dental |
| [**`production_gold_set_OPD.xlsx`**](file:///home/ai/Workspace/Doha/MedVoice/data/production_gold_set/production_gold_set_OPD.xlsx) | 131 | `Final Diagnosis`, `ICD-10 Codes`, and all 17 OPD extraction headers |
| [**`production_gold_set_ER.xlsx`**](file:///home/ai/Workspace/Doha/MedVoice/data/production_gold_set/production_gold_set_ER.xlsx) | 37 | `Final Diagnosis`, `ICD-10 Codes`, `Triage Priority`, and all 22 ER extraction headers |
| [**`production_gold_set_DENTAL.xlsx`**](file:///home/ai/Workspace/Doha/MedVoice/data/production_gold_set/production_gold_set_DENTAL.xlsx) | 3 | `Final Diagnosis`, `ICD-10 Codes`, and all 21 Dental extraction headers |

## Reference & Raw Data

- **`gold_set_candidates.json`**: Machine-extracted baseline candidate extractions.
- **`gold_set_candidates.jsonl`**: JSON Lines baseline format.
- **`production_logs_raw.jsonl`**: Preserved raw logs of all 185 production requests.
- **`production_excluded.json`**: 14 non-ok / error records from production.

## Summary Metrics

| Metric | Count |
| :--- | :--- |
| **Total Production Logs Extracted** | 185 |
| **Valid Gold Set Cases** | 171 |
| **Excluded (Errors/Empty)** | 14 |
| **Languages** | English (89), Arabic (82) |
| **Clinical Modes** | Doctor dictation (102), Doctor-Patient conversation (69) |
| **Departments** | OPD (131), ER (37), Dental (3) |
