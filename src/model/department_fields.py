"""
Per-department clinical-note field checklist.

This is the single source of truth QuestionGenerator uses to decide which
fields are "missing" from an extraction — deterministically, from the
extracted JSON itself, not from a second independent LLM guess over the raw
transcript. That's what keeps the "Questions for the doctor" list from ever
contradicting what's already shown filled-in on the Feature extraction card.

icd10_codes and final_diagnosis are intentionally NOT listed here — they are
always AI-recommended (see the extraction prompts) and are never asked about.
The ER department's "triage_priority" field follows the same rule and is
also intentionally excluded, for the same reason (always AI-recommended,
never left blank, so it never needs a "missing field" question).

NOTE: index.html has its own DEPARTMENT_FIELDS (for UI layout/order/icons).
Keep the `key`/`label` values in that JS object and this module in sync if
you ever add/rename/reorder a field.
"""

DEPARTMENT_FIELDS = {
    "er": [
        {"key": "chief_complaint",        "label": "Chief Complaint (C/O)",        "question": "What is the patient's chief complaint (C/O)?"},
        {"key": "onset",                  "label": "Onset",                        "question": "When did the complaint start (onset)?"},
        {"key": "course",                 "label": "Course",                       "question": "How has the condition progressed (course)?"},
        {"key": "duration",               "label": "Duration",                     "question": "How long has the complaint lasted (duration)?"},
        {"key": "past_medical_history",   "label": "Past Medical History",         "question": "What is the patient's past medical history?"},
        {"key": "past_surgical_history",  "label": "Past Surgical History",        "question": "What is the patient's past surgical history?"},
        {"key": "medication_history",     "label": "Medication History",           "question": "What medications is the patient currently taking?"},
        {"key": "allergy_status",         "label": "Allergy Status",               "question": "What is the patient's allergy status (or NKDA)?"},
        {"key": "family_history",         "label": "Family History",               "question": "Is there any relevant family history?"},
        {"key": "social_history",         "label": "Social History",               "question": "Is there any relevant social history (smoking, alcohol, occupation, living situation)?"},
        {"key": "airway",                 "label": "A – Airway",                   "question": "What are the airway (A) findings on primary survey?"},
        {"key": "breathing",              "label": "B – Breathing",                "question": "What are the breathing (B) findings on primary survey?"},
        {"key": "circulation",            "label": "C – Circulation",              "question": "What are the circulation (C) findings on primary survey?"},
        {"key": "disability",             "label": "D – Disability",               "question": "What are the disability (D) / neuro findings on primary survey?"},
        {"key": "exposure",               "label": "E – Exposure",                 "question": "What are the exposure (E) findings on primary survey?"},
        {"key": "examination",            "label": "Examination",                  "question": "What were the examination findings?"},
        {"key": "investigations",         "label": "Investigations",               "question": "Were any labs, ECG, or imaging results obtained?"},
        {"key": "plan_of_care",           "label": "Plan of Care",                 "question": "What is the plan of care?"},
        {"key": "requested_consultation", "label": "Requested Consultation",       "question": "Were any specialty consultations or referrals requested?"},
        {"key": "discharge_instructions", "label": "Discharge Instructions",       "question": "What discharge instructions were given to the patient?"},
        {"key": "patient_disposition",    "label": "Patient Disposition",          "question": "What is the patient's disposition (e.g. admitted, discharged, transferred)?"},
        {"key": "sick_leave",             "label": "Sick Leave",                   "question": "Was sick leave given, and for how long?"},
    ],
    "opd": [
        {"key": "allergy_status",          "label": "Allergy Status",                       "question": "What is the patient's allergy status (or NKDA)?"},
        {"key": "chief_complaint",         "label": "Chief Complaint (C/O)",                "question": "What is the patient's chief complaint (C/O)?"},
        {"key": "onset",                   "label": "Onset",                                "question": "How did the complaint begin (sudden or gradual), and/or when did it start?"},
        {"key": "course",                  "label": "Course",                               "question": "How has the condition progressed (course)?"},
        {"key": "duration",                "label": "Duration",                             "question": "How long has the complaint lasted, and has it recently gotten worse?"},
        {"key": "comorbidities",           "label": "Comorbidities",                        "question": "Does the patient have any comorbidities?"},
        {"key": "medication_history",      "label": "Medication History",                   "question": "What medications is the patient currently taking?"},
        {"key": "past_medical_history",    "label": "Past Medical History",                 "question": "What is the patient's past medical history?"},
        {"key": "past_surgical_history",   "label": "Past Surgical History",                "question": "What is the patient's past surgical history?"},
        {"key": "social_family_history",   "label": "Social / Marital / Family History",    "question": "What is the patient's social, marital, or family history?"},
        {"key": "examination_finding",     "label": "Examination Finding",                  "question": "What were the examination findings?"},
        {"key": "plan_of_care",            "label": "Plan of Care (short/long term)",       "question": "What is the short/long term plan of care?"},
        {"key": "procedures_planned",      "label": "Procedures Planned",                   "question": "Are any procedures planned?"},
        {"key": "referrals",               "label": "Referrals",                            "question": "Were any referrals made?"},
        {"key": "follow_up_timing",        "label": "Follow-up Timing",                     "question": "When should the patient follow up?"},
        {"key": "patient_instructions",    "label": "Patient Instructions",                 "question": "What instructions were given to the patient?"},
        {"key": "outcome_progress_status", "label": "Outcome / Progress Status",            "question": "If this is a follow-up visit, what is the outcome/progress status?"},
    ],
    "dental": [
        {"key": "chief_complaint",        "label": "Chief Complaint (C/O)",       "question": "What is the patient's chief complaint (C/O)?"},
        {"key": "onset",                  "label": "Onset",                       "question": "When did the presenting complaint/condition start?"},
        {"key": "course",                 "label": "Course",                      "question": "How has the condition progressed since onset?"},
        {"key": "duration",               "label": "Duration",                    "question": "How long has the complaint lasted?"},
        {"key": "past_medical_history",   "label": "Past Medical History",        "question": "What is the patient's past medical history (e.g. systemic conditions, current medications)?"},
        {"key": "consent_obtained",       "label": "Consent Obtained",            "question": "Was consent obtained from the patient (Yes/No/Verbal/Not Applicable)?"},
        {"key": "treatment_plan_status",  "label": "Treatment Plan Status",       "question": "What is the treatment plan status (Started/In-Progress/Completed)?"},
        {"key": "charge_category",        "label": "Charge Category",             "question": "What is the charge category (Standard/Insurance/Discounted)?"},
        {"key": "patient_education_given","label": "Patient Education Given",     "question": "Was patient education given after the procedure (Yes/No)?"},
        {"key": "radiographs_taken",      "label": "Radiographs Taken",           "question": "Which radiographs were taken, and at which stage (pre/intra/post-treatment)?"},
        {"key": "tooth_number",           "label": "Tooth Number",                "question": "Which tooth number(s) were involved?"},
        {"key": "subspecialty_focus",     "label": "SubSpecialty Focus",          "question": "Which dental subspecialty does this procedure fall under?"},
        {"key": "procedure_executed",     "label": "Procedure Executed",          "question": "What procedure was executed?"},
        {"key": "materials_inventory",    "label": "Materials Inventory",         "question": "What materials were used?"},
        {"key": "re_make",                "label": "Re-Make",                     "question": "Did any restoration/appliance/impression need to be re-made (Yes/No)?"},
        {"key": "complication_if_present","label": "Complication if Present",     "question": "Were there any complications during or after the procedure?"},
        {"key": "internal_referral",      "label": "Internal Referral",           "question": "Was the patient referred internally to another specialist?"},
        {"key": "service_leakage_alert",  "label": "Service Leakage Alert",       "question": "Was there any external referral reason or patient refusal to note?"},
        {"key": "next_visit_tooth_number","label": "Next Visit Tooth Number",     "question": "Which tooth is the target for the next visit?"},
        {"key": "next_visit_treatment",   "label": "Next Visit Treatment",        "question": "What treatment is planned for the next visit?"},
        {"key": "retention_priority",     "label": "Retention Priority",          "question": "What is the retention priority for this patient?"},
        {"key": "followup_timeline",      "label": "Follow-Up Timeline",          "question": "What is the timeline for follow-up?"},
    ],
}


def get_department_fields(department: str):
    return DEPARTMENT_FIELDS.get((department or "").strip().lower(), DEPARTMENT_FIELDS["opd"])