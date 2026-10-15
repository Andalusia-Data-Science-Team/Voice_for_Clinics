"""
Prompts for the two parallel analysis steps.

  1. get_extraction_prompt  — clinical-note FIELD extraction only. No diagnoses,
                              no ICD-10, no patient history.
  2. get_diagnosis_prompt   — ICD-10 recommendation + reasoning (ER also:
                              differential diagnoses + triage priority). This is
                              the only step that sees patient history.

Both take the same translated text, so they can run concurrently.

Field descriptions are the same wording as the old per-department prompts, just
factored into shared constants. The old "conversation" variants differed only in
a few "from the patient's answers / from the doctor's statements" hints, which
are now one hint paragraph (_CONVERSATION_HINT) appended in conversation mode.

Delete from utils/prompt.py: everything from the "ER — extraction" banner down,
plus _history_block, _HISTORY_FIELDS_DOC and _FINAL_DIAGNOSIS_RATIONALE_DOC.
"""

import json

# ── Shared field descriptions ──────────────────────────────────────────────
_CC = ("Primary reason for the visit. All patient complaints, signs, and symptoms should be "
       "captured here, even if they are not the main reason for the visit.")
_ONSET = ("when and/or how the symptom or complaint started. Include the reported starting point and "
          "whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned. "
          "Do not infer the onset if it is not stated.")
_COURSE = ("Extract how the symptom or complaint has changed or behaved over time since it started. "
           "Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, "
           "persistent, fluctuating, or unchanged. Only report the course if it is explicitly stated or clearly "
           "described in the text. Do not infer a course from the severity of the symptom alone.")
_DURATION = ("Extract how long the symptom or complaint has been present. Include the exact duration when "
             "available, such as hours, days, weeks, months, or years, including any recent change in severity "
             "if mentioned (e.g. \"2 months, worse over the last 3 weeks\"). Do not calculate or infer a "
             "duration unless it is explicitly supported by the patient's statement.")

_CONVERSATION_HINT = (
    "Source hints for a doctor-patient conversation: take the chief complaint from the patient's opening "
    "statements; take history fields (medical, surgical, medication, allergy, family, social) from the "
    "patient's answers; take examination findings, plan of care, referrals and instructions from the "
    "doctor's statements."
)

# ── Per-department extraction schemas (keys must match department_fields.py) ─
_ER_FIELDS = [
    ("chief_complaint", _CC),
    ("onset", _ONSET), ("course", _COURSE), ("duration", _DURATION),
    ("past_medical_history", 'Relevant past medical history. "No Past Medical History" if explicitly stated as such.'),
    ("past_surgical_history", 'Relevant past surgical history. "No Past Surgical History" if explicitly stated as such.'),
    ("medication_history", 'Current home medications the patient is taking. "No Medication History" if explicitly stated as such.'),
    ("allergy_status", 'Documented drug/food/other allergies. "No Known Allergies" if explicitly stated as such.'),
    ("family_history", "Relevant family history (e.g. hereditary/familial conditions), if mentioned."),
    ("social_history", "Relevant social history — smoking, alcohol, occupation, living situation — if mentioned."),
    ("airway", "Primary survey — Airway (A) findings."),
    ("breathing", "Primary survey — Breathing (B) findings."),
    ("circulation", "Primary survey — Circulation (C) findings."),
    ("disability", "Primary survey — Disability (D), neuro status."),
    ("exposure", "Primary survey — Exposure (E) findings."),
    ("examination", 'General/focused physical examination findings. Do not include laboratory, ECG, or imaging results here — those belong in "investigations".'),
    ("investigations", 'Relevant laboratory, ECG, or imaging results mentioned (e.g. troponin, ECG findings, CBC, renal function tests).'),
    ("plan_of_care", 'The intended management plan (e.g. admission, monitoring, treatment, further workup). Do not include specific specialty consultations/referrals here — those belong in "requested_consultation".'),
    ("requested_consultation", 'Any specialty consultations or referrals requested (e.g. "Cardiology", "Infectious Disease", "ICU consultant review"), if mentioned.'),
    ("discharge_instructions", "Instructions given for discharge. If the patient is being admitted rather than discharged, state that discharge instructions are not applicable due to admission."),
    ("patient_disposition", 'The patient\'s disposition/destination from the ER (e.g. "Admitted to CCU", "Admitted to ICU", "Discharged home", "Transferred").'),
    ("sick_leave", 'Sick leave duration/details, if mentioned. State "Not applicable" if the patient is being admitted rather than discharged.'),
]

_OPD_FIELDS = [
    ("allergy_status", "Documented drug/food/other allergies."),
    ("chief_complaint", _CC),
    ("onset", _ONSET), ("course", _COURSE), ("duration", _DURATION),
    ("comorbidities", "Existing comorbid conditions."),
    ("medication_history", 'Current medications (part of medical history). "No Medication History" if explicitly stated as such.'),
    ("past_medical_history", 'Relevant past medical history — INCLUDE any ongoing chronic conditions the patient reports (e.g. diabetes, hypertension, asthma) even though the same condition is also captured under "comorbidities"; the two fields feed different parts of the note template. Excludes current medications (see medication_history).'),
    ("past_surgical_history", 'Relevant past surgical history. "No Past Surgical History" if explicitly stated as such.'),
    ("social_family_history", "Social, marital, or family history."),
    ("examination_finding", "Examination findings, including vital signs where documented."),
    ("plan_of_care", "Short/long term plan of care."),
    ("procedures_planned", "Any investigations, imaging, labs, or procedures planned/ordered, even if also mentioned in the plan of care."),
    ("referrals", "Referrals made or recommended."),
    ("follow_up_timing", "When the patient should follow up."),
    ("patient_instructions", "Instructions given to the patient."),
    ("outcome_progress_status", 'Outcome/progress status if this is a follow-up visit, else "".'),
]

_DENTAL_FIELDS = [
    ("chief_complaint", _CC),
    ("onset", _ONSET), ("course", _COURSE), ("duration", _DURATION),
    ("past_medical_history", 'Relevant past medical history mentioned (e.g. systemic conditions, current medications affecting treatment). "No Past Medical History" if explicitly stated as such.'),
    ("consent_obtained", 'Legal/ethical consent status — "Yes", "No", "Verbal", or "Not Applicable".'),
    ("treatment_plan_status", 'Milestone of the overall care plan — "Started", "In-Progress", or "Completed".'),
    ("charge_category", 'Billing tier — "Standard", "Insurance", or "Discounted".'),
    ("patient_education_given", 'Whether post-procedure care instructions were communicated — "Yes" or "No".'),
    ("radiographs_taken", 'Which X-ray type(s) were captured and at which stage — Pre-treatment / Intra-treatment / Post-treatment — using modalities such as PA (Periapical), BW (Bitewing), OPG (Panoramic), or CBCT (e.g. "Pre-treatment: Periapical").'),
    ("tooth_number", "Tooth identifier(s) using Universal (1-32) or FDI notation."),
    ("subspecialty_focus", "Dental branch governing the procedure (Endodontics, Periodontics, Orthodontics, Prosthodontics, Surgery, Pedo, or General)."),
    ("procedure_executed", "Short summary of the actual clinical intervention performed."),
    ("materials_inventory", "Specific brands, sizes, or materials used."),
    ("re_make", 'Whether a restoration/appliance/impression had to be redone — "Yes" or "No".'),
    ("complication_if_present", 'Any adverse events or unexpected issues, or "" if none.'),
    ("internal_referral", "Whether patient was referred to another specialist within the practice, and which specialty."),
    ("service_leakage_alert", 'Lost revenue/care gap — External Referral Reason or Patient Refusal, or "" if none.'),
    ("next_visit_tooth_number", "Target tooth for the upcoming appointment."),
    ("next_visit_treatment", "Planned procedure/goal for the next appointment."),
    ("retention_priority", 'Triage label for how critical it is to retain this patient (e.g. "High", "Medium", "Low").'),
    ("followup_timeline", "Scheduled timeframe for the next check-in."),
]

_DEPT = {
    "er": dict(persona="a medical expert", visit="an Emergency Room (ER) encounter",
               template="ER", fields=_ER_FIELDS),
    "opd": dict(persona="a medical expert", visit="an Outpatient Department (OPD) visit",
                template="OPD", fields=_OPD_FIELDS),
    "dental": dict(persona="a dental clinical documentation expert", visit="a dental visit",
                   template="Dental", fields=_DENTAL_FIELDS),
}

_EXTRACTION_EXAMPLES = {
    "er": {
        "chief_complaint": "Right lower quadrant abdominal pain", "onset": "2 days ago",
        "course": "Progressively worsening", "duration": "2 days",
        "past_medical_history": "Hypertension", "past_surgical_history": "No Past Surgical History",
        "medication_history": "Amlodipine 5mg once daily", "allergy_status": "No Known Allergies",
        "family_history": "", "social_history": "Non-smoker", "airway": "Patent",
        "breathing": "Spontaneous, saturating 98% on room air",
        "circulation": "BP 128/82, HR 88, warm and well perfused",
        "disability": "GCS 15, alert and oriented", "exposure": "No external injuries noted",
        "examination": "Tenderness and guarding at McBurney's point, positive Rovsing's sign",
        "investigations": "WBC mildly elevated. Abdominal ultrasound pending.",
        "plan_of_care": "Laparoscopic appendectomy", "requested_consultation": "General Surgery",
        "discharge_instructions": "Not applicable due to admission",
        "patient_disposition": "Admitted for surgery", "sick_leave": "Not applicable",
    },
    "opd": {
        "allergy_status": "No known drug allergies", "chief_complaint": "Persistent cough and fever",
        "onset": "Gradual, started 5 days ago", "course": "Gradually worsening", "duration": "5 days",
        "comorbidities": "Type 2 diabetes", "medication_history": "Metformin 500mg twice daily",
        "past_medical_history": "Type 2 diabetes; seasonal allergies",
        "past_surgical_history": "No Past Surgical History",
        "social_family_history": "Non-smoker, lives with family",
        "examination_finding": "Temp 38.2C, bilateral crackles on auscultation",
        "plan_of_care": "Start antibiotics, symptomatic treatment",
        "procedures_planned": "CBC; Chest X-ray", "referrals": "", "follow_up_timing": "1 week",
        "patient_instructions": "Rest, hydration, return if breathless", "outcome_progress_status": "",
    },
    "dental": {
        "chief_complaint": "Severe pain on the lower left back tooth when chewing", "onset": "1 week ago",
        "course": "Progressively worsening", "duration": "1 week",
        "past_medical_history": "Controlled hypertension, on Amlodipine", "consent_obtained": "Yes",
        "treatment_plan_status": "In-Progress", "charge_category": "Insurance",
        "patient_education_given": "Yes", "radiographs_taken": "Pre-treatment: Periapical",
        "tooth_number": "FDI 36", "subspecialty_focus": "Endodontics",
        "procedure_executed": "Root canal treatment, first sitting",
        "materials_inventory": "Rotary NiTi files, calcium hydroxide dressing", "re_make": "No",
        "complication_if_present": "", "internal_referral": "", "service_leakage_alert": "",
        "next_visit_tooth_number": "FDI 36", "next_visit_treatment": "Root canal obturation",
        "retention_priority": "High", "followup_timeline": "1 week",
    },
}


def _dept(department: str) -> str:
    d = (department or "").strip().lower()
    return d if d in _DEPT else "opd"


def get_extraction_prompt(department: str, text: str, is_conversation: bool = False) -> str:
    d = _dept(department)
    cfg = _DEPT[d]
    source = "doctor-patient conversation" if is_conversation else "clinical text"
    label = "Conversation" if is_conversation else "Text"
    fields_doc = "\n".join(f'  - "{k}": {desc}' for k, desc in cfg["fields"])
    example = json.dumps(
        {"json_data": _EXTRACTION_EXAMPLES[d],
         "summary": "Fields extracted directly from what was stated; unmentioned fields left blank."},
        indent=2, ensure_ascii=False)
    hint = f"\n{_CONVERSATION_HINT}\n" if is_conversation else ""
    return f"""
You are {cfg['persona']}. Given the following {source} from {cfg['visit']}, extract the fields
below for the {cfg['template']} Clinical Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings):
{fields_doc}
- "summary": A short string explaining the rationale behind the extraction.

Leave a field as "" if it is not mentioned. Do not invent information that is not present or
reasonably implied.
Do NOT include diagnoses or ICD-10 codes — a separate step produces those.
{hint}
{label}: {text}

Example output:
{example}
"""


# ── Step 2: diagnosis ──────────────────────────────────────────────────────

def _history_block(patient_history_text: str) -> str:
    if not patient_history_text:
        return ""
    return f"""

Patient History (this patient's visits over the last year, most recent first).
Use it to inform your diagnosis recommendations — e.g. recognizing a chronic or
recurring condition, a relevant prior diagnosis, or a comorbidity that changes the
differential for today's presentation. If none of it is relevant to today's visit,
ignore it and say so in "history_notes".

{patient_history_text}
"""


_HISTORY_DOC = """  - "history_considered": "Yes" if the patient history block genuinely changed or
    supported your recommendation, otherwise "No". If no patient history block was
    provided, this must always be "No" — never claim you considered history you weren't given.
  - "history_notes": If history_considered is "Yes", a short note citing the SPECIFIC prior
    diagnosis code(s) and date(s) that influenced your recommendation. If more than one prior
    diagnosis is relevant, name all of them. Otherwise ""."""

_ER_EXTRA_DOC = """  - "differential_diagnoses": A list of 3-6 clinically reasonable alternatives
    formatted "<Australian ICD-10-AM code> - <exact catalogue description>", most important first.
    Select only from the supplied catalogue options; do not invent or substitute diagnoses or codes.
  - "triage_priority": ALWAYS populate this — the patient's triage urgency. If a triage level
    wasn't explicitly stated, infer your best classification from the clinical context (vitals,
    primary survey findings, presentation) using the Emergency Severity Index (1 = most urgent ...
    5 = least urgent), with a brief inline justification, e.g. "ESI 2 - Emergent (stable vitals but
    high-risk presentation)". This is clinical judgement, not a transcription — never leave it blank."""

_DIAGNOSIS_EXAMPLES = {
    "er": {
        "icd10_codes": [
            "K35.8 - Acute appendicitis, other and unspecified",
            "R10.3 - Pain localised to other parts of lower abdomen",
        ],
        "differential_diagnoses": [
            "K35.2 - Acute appendicitis with generalised peritonitis",
            "I88.0 - Nonspecific mesenteric lymphadenitis",
            "N23 - Unspecified renal colic",
            "N83.5 - Torsion of ovary, ovarian pedicle and fallopian tube",
            "K50.9 - Crohn's disease, unspecified",
        ],
        "triage_priority": "ESI 3 - Urgent (stable vitals, localized peritoneal signs, needs timely surgical evaluation)",
        "reasoning": "Progressive RLQ pain with tenderness and guarding supports acute appendicitis. R10.3 records the documented lower-abdominal pain.",
        "history_considered": "No", "history_notes": "",
    },
    "opd": {
        "icd10_codes": [
            "J18.9 - Pneumonia, unspecified",
            "J20.9 - Acute bronchitis, unspecified",
        ],
        "reasoning": "Fever with bilateral crackles over a gradually worsening 5-day course, with a chest X-ray ordered, favours pneumonia over simple bronchitis; bronchitis stays as the alternative.",
        "history_considered": "No", "history_notes": "",
    },
    "dental": {
        "icd10_codes": [
            "K04.0 - Pulpitis",
            "K04.4 - Acute apical periodontitis of pulpal origin",
        ],
        "reasoning": "Severe pain on chewing localized to tooth 36 with a pre-treatment periapical taken and root canal started is consistent with pulpitis; apical periodontitis is the next most likely.",
        "history_considered": "No", "history_notes": "",
    },
}


def get_diagnosis_prompt(
    department: str,
    text: str,
    is_conversation: bool = False,
    patient_history_text: str = "",
    catalogue_candidates: list[tuple[str, str]] | None = None,
) -> str:
    d = _dept(department)
    cfg = _DEPT[d]
    source = "doctor-patient conversation" if is_conversation else "clinical text"
    label = "Conversation" if is_conversation else "Text"
    code_kind = "ICD-10-AM"
    er_doc = f"\n{_ER_EXTRA_DOC}" if d == "er" else ""
    if catalogue_candidates is None:
        candidates_doc = (
            "No catalogue candidates were supplied. Return an empty "
            '"icd10_codes" list and do not invent a code or diagnosis.'
        )
        example_data = _DIAGNOSIS_EXAMPLES[d]
    else:
        candidates_doc = (
            "For icd10_codes"
            + (
                " and differential_diagnoses"
                if d == "er"
                else ""
            )
            + ", select ONLY from these entries retrieved from the Australian "
            "ICD-10-AM catalogue. Copy each code and description exactly. Do "
            "not use medical knowledge to add or substitute codes or diagnosis "
            "labels. If none fit the evidence, return an empty list."
        )
        candidates_doc += "\n" + "\n".join(
            f"- {code} - {description}"
            for code, description in catalogue_candidates
        )
        example_data = {
            "icd10_codes": (
                [f"{catalogue_candidates[0][0]} - {catalogue_candidates[0][1]}"]
                if catalogue_candidates
                else []
            ),
            "reasoning": "Explain which documented findings support the selected catalogue entry.",
            "history_considered": "No",
            "history_notes": "",
        }
        if d == "er":
            example_data.update({
                "differential_diagnoses": (
                    [f"{catalogue_candidates[0][0]} - {catalogue_candidates[0][1]}"]
                    if catalogue_candidates
                    else []
                ),
                "triage_priority": _DIAGNOSIS_EXAMPLES["er"]["triage_priority"],
            })
    example = json.dumps(
        {"json_data": example_data, "summary": "..."},
        indent=2,
        ensure_ascii=False,
    )
    return f"""
You are {cfg['persona']} acting as a clinical coding and diagnostic-reasoning assistant. Given the
following {source} from {cfg['visit']}, recommend diagnoses. Think through the case before answering.
Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys:
  - "icd10_codes": A list of strings, each formatted
    "<code> - <exact catalogue description>", containing your recommended
    {code_kind} codes, ordered from MOST to LEAST likely (the first entry is
    your leading diagnosis). Return up to 5 entries, selecting only from the
    supplied catalogue options. If no supplied entry fits the evidence, return
    an empty list.
  - "reasoning": ALWAYS populate this — 2-4 sentences explaining WHY the leading code was chosen over
    the alternatives, citing the concrete symptoms, findings, or results from TODAY'S text that support
    it. If history_considered is "Yes", weave in the prior diagnosis/date that reinforced the choice.{er_doc}
{_HISTORY_DOC}
- "summary": One short sentence summarizing the recommendation.

Base the recommendation only on what the text supports; do not invent findings.
The server enforces the supplied catalogue candidate list. Any unlisted code
or diagnosis will be excluded from the recommendations.
{_history_block(patient_history_text)}
{label}: {text}

Catalogue options:
{candidates_doc}

Example output:
{example}
"""