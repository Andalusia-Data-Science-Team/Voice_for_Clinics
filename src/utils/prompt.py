def _specialty_context_line(specialty):
    """Shared context line inserted into refine prompts when a specialty is set."""
    if not specialty:
        return ""
    return (
        f"\nMEDICAL SPECIALTY: {specialty}\n"
        f"Pay particular attention to {specialty}-specific terminology, abbreviations, "
        f"procedure names, and drug names common in this specialty when correcting the text.\n"
    )


def _history_block(patient_history_text):
    """Shared helper — the scoped patient-history context block, or "" if none."""
    if not patient_history_text:
        return ""
    return f"""

Patient History (this patient's visits over the last year, most recent
first). Use this ONLY to inform your "icd10_codes" and "final_diagnosis"
recommendations — for example, recognizing a chronic or recurring
condition, a relevant prior diagnosis, or a comorbidity that changes the
differential for today's presentation. Do NOT use it to fill in any other
field (chief complaint, examination, plan, etc.) — those must come only
from today's text. If none of this history is actually relevant to today's
visit, ignore it and say so in "history_notes".

{patient_history_text}
"""


_HISTORY_FIELDS_DOC = """  - "history_considered": "Yes" if the patient history block above genuinely
    changed or supported your icd10_codes/final_diagnosis recommendation,
    otherwise "No". If no patient history block was provided above, this
    must always be "No" — never claim you considered history that wasn't given to you.
    - "history_notes": If history_considered is "Yes", a short note citing the
    SPECIFIC prior diagnosis code(s) and date(s) that influenced your
    recommendation (e.g. "D64.8 - Other specified anaemias documented
    6/17/2026, matching today's fatigue/pallor/palpitations"). If more than
    one prior diagnosis in the history is relevant to today's presentation,
    name all of them, not just one — the doctor needs to see the competing
    evidence, not just your final pick. Otherwise ""."""


_FINAL_DIAGNOSIS_RATIONALE_DOC = """  - "final_diagnosis_rationale": ALWAYS populate this — a short (1-2
    sentence) clinician-readable explanation of WHY you chose this specific
    final_diagnosis, citing the concrete symptoms, exam findings, or results
    from TODAY'S text that support it (e.g. "Fatigue, pallor, and
    palpitations are consistent with anaemia; CBC was ordered to confirm").
    If history_considered is "Yes", also weave in the specific prior
    diagnosis/date that reinforced the choice, rather than just repeating
    history_notes verbatim — this should read as ONE coherent justification
    a doctor can sanity-check at a glance. Never leave this blank if
    final_diagnosis is populated."""


def get_refine_arabic_prompt_deepseek(raw_text, specialty=None):
    specialty_line = _specialty_context_line(specialty)
    return f"""
Act as a senior medical transcription editor specializing in Arabic healthcare documentation.
{specialty_line}
**ORIGINAL TRANSCRIPTION:**
{raw_text}

**EDITING TASKS:**
- Correct grammatical errors and awkward phrasing
- Improve sentence structure and flow
- Maintain all medical facts and clinical details
- De-identify speaker references when possible
- Ensure professional medical Arabic standards
- Enhance readability for medical professionals

**CRITICAL RULES:**
→ Output ONLY the corrected Arabic text
→ No additional commentary or explanations
→ Preserve the original medical meaning completely
→ Use formal Arabic appropriate for medical records
→ Ensure refining any mentioned medications
→ Do NOT add, infer, or make more specific any clinical detail that was not explicitly stated — e.g. do not add a side (left/right), a position (upper/lower), a severity, or a location that the original text did not mention. Fix language only; never add medical information.
→ Do not use asterisks (*) anywhere in the output, for any reason

**CORRECTED MEDICAL TEXT:**
"""


def get_refine_english_prompt_deepseek(translated_text, specialty=None):
    specialty_line = _specialty_context_line(specialty)
    return f"""
    Correct the grammar and structure of this English medical text.
    Return only the corrected English text with no additional explanations.
    Keep brand and generic drug names unchanged
    Ensure refining any mentioned medications
    Do not add, infer, or make more specific any clinical detail that was not explicitly stated —
    e.g. do not add a side (left/right), a position (upper/lower), a severity, or a location that
    the original text did not mention. Fix language only; never add medical information.
    Do not use asterisks (*) anywhere in the output, for any reason
    {specialty_line}
    ORIGINAL TEXT:
    \"\"\"{translated_text}\"\"\"
    **ONLY return the refinment no more**
    """


# --- Conversation Mode Prompts ---
def get_refine_arabic_prompt_deepseek_conversation(raw_text, specialty=None):
    specialty_line = _specialty_context_line(specialty)
    return f"""
    Act as a clinical transcription editor. Refine this Arabic medical conversation for accuracy and clarity.
    {specialty_line}
    **CONVERSATION TO PROCESS:**
    {raw_text}

    **SPEAKER IDENTIFICATION:**
    - الدكتور: Medical professional (asks questions, examines, diagnoses, prescribes)
    - المريض: Patient (describes symptoms, answers questions, shares history)

    **QUALITY STANDARDS:**
    1. Accuracy: Preserve all medical content, terminology, and clinical context
    2. Clarity: Fix grammatical errors and improve sentence flow
    3. Format: Clearly label each speaker turn with plain text "الدكتور:" or "المريض:" — no bold, no asterisks
    4. Anonymization: Minimize personal identifiers while keeping dialogue intact
    5. Professionalism: Use formal medical Arabic appropriate for clinical documentation

    **CRITICAL:**
    → Output ONLY the refined conversation in Arabic
    → No additional text, explanations, or commentary
    → Maintain original dialogue sequence and medical meaning
    → Each speaker turn must be labeled as plain text "الدكتور:" or "المريض:" — never wrap the label in asterisks or any other markup
    → Keep brand and generic drug names unchanged
    → Ensure refining any mentioned medications
    → Do NOT add, infer, or make more specific any clinical detail that was not explicitly stated — e.g. do not add a side (left/right), a position (upper/lower), a severity, or a location that the original text did not mention. Fix language only; never add medical information.
    → Do not use asterisks (*) anywhere in the output, for any reason — not around speaker labels, not anywhere else

    **REFINED CLINICAL CONVERSATION:**
"""


def get_refine_english_prompt_deepseek_conversation(translated_text, specialty=None):
    specialty_line = _specialty_context_line(specialty)
    return f"""
Refine this English medical conversation while preserving the dialogue structure.
{specialty_line}
Instructions:
1. Correct grammar, punctuation, and phrasing for natural spoken English.
2. Ensure medical terminology and tone are appropriate for a clinical dialogue.
3. Keep brand and generic drug names unchanged. 
4. Ensure refining any mentioned medications
5. Maintain the natural conversational flow.
6. Use "Doctor:" and "Patient:" exactly as shown below (capitalize only the first letter, plain text — no asterisks, no bold).
7. Do NOT add, infer, or make more specific any clinical detail that was not explicitly stated — e.g. do not add a side (left/right), a position (upper/lower), a severity, or a location that the original text did not mention. Fix language only; never add medical information.
8. Do not use asterisks (*) anywhere in the output, for any reason.
9. Return ONLY the refined conversation in dialogue format — no introductions, explanations, or extra text.

Example format:
Doctor: Good morning. What brings you in today?
Patient: I’ve had chest pain for a few days.

Doctor: What kind of pain?
Patient: A pressure in the center of my chest, worse when I climb stairs.

ORIGINAL TEXT:
\"\"\"{translated_text}\"\"\"

Only return the dialogue no more — plain text speaker labels, no asterisks anywhere
"""


def get_translation_prompt_deepseek(refined_text):
    return f"""
**TASK:** Translate Arabic medical text to English

**SOURCE TEXT (Arabic):**
{refined_text}

**TRANSLATION REQUIREMENTS:**
- Translate ALL Arabic text to English
- Preserve medical terminology accurately
- Maintain professional medical language
- Keep the structure and meaning identical

**CRITICAL RULES:**
→ Output MUST be in ENGLISH only
→ No Arabic characters or words in the output
→ Return ONLY the English translation
→ No additional commentary or explanations
→ Do not use astrisks at all
→ If the text is already English, return it as-is

**ENGLISH TRANSLATION:**
"""


def get_translation_prompt_deepseek_conversation(refined_text):
    return f"""
**TASK:** Translate Arabic medical conversation to English

**SOURCE CONVERSATION (Arabic):**
{refined_text}

**TRANSLATION REQUIREMENTS:**
- Translate ALL Arabic dialogue to English
- Preserve speaker labels exactly: **الدكتور:** → **Doctor:** and **المريض:** → **Patient:**
- Maintain accurate medical terminology
- Keep conversational flow and structure

**SPEAKER LABEL MAPPING:**
- **الدكتور:** must become **Doctor:**
- **المريض:** must become **Patient:**

**CRITICAL RULES:**
→ Output MUST be in ENGLISH only
→ No Arabic characters in the final output
→ Preserve the dialogue format with English speaker labels
→ Return ONLY the translated conversation
→ No additional text or explanations
→ Do not use astrisks at all

**ENGLISH TRANVERSATION:**
"""

 
# ═══════════════════════════════ ER — extraction ═══════════════════════════
 
def get_er_extraction_prompt(translated_text, patient_history_text=""):
    return f"""
You are a medical expert. Given the following clinical text from an
Emergency Room (ER) encounter, extract the fields below for the ER Clinical
Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "chief_complaint": Primary reason for the visit/procedure. All patient complaints, signs, and symptoms should be captured here, even if they are not the main reason for the visit.
  - "onset": when and/or how the symptom or complaint started.
    Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
    Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "past_medical_history": Relevant past medical history. "No Past Medical History" if explicitly stated as such.
  - "past_surgical_history": Relevant past surgical history. "No Past Surgical History" if explicitly stated as such.
  - "medication_history": Current home medications the patient is taking. "No Medication History" if explicitly stated as such.
  - "allergy_status": Documented drug/food/other allergies. "No Known Allergies" if explicitly stated as such.
  - "family_history": Relevant family history (e.g. hereditary/familial conditions), if mentioned.
  - "social_history": Relevant social history — smoking, alcohol, occupation, living situation — if mentioned.
  - "airway": Primary survey — Airway (A) findings.
  - "breathing": Primary survey — Breathing (B) findings.
  - "circulation": Primary survey — Circulation (C) findings.
  - "disability": Primary survey — Disability (D), neuro status.
  - "exposure": Primary survey — Exposure (E) findings.
  - "examination": General/focused physical examination findings. Do not include laboratory, ECG, or imaging results here — those belong in "investigations".
  - "investigations": Relevant laboratory, ECG, or imaging results mentioned (e.g. troponin, ECG findings, CBC, renal function tests). Leave as "" if none mentioned.
  - "plan_of_care": The intended management plan (e.g. admission, monitoring, treatment, further workup). Do not include specific specialty consultations/referrals here — those belong in "requested_consultation".
  - "requested_consultation": Any specialty consultations or referrals requested (e.g. "Cardiology", "Infectious Disease", "ICU consultant review"), if mentioned.
  - "discharge_instructions": Instructions given for discharge. If the patient is being admitted rather than discharged, state that discharge instructions are not applicable due to admission.
  - "patient_disposition": The patient's disposition/destination from the ER (e.g. "Admitted to CCU", "Admitted to ICU", "Discharged home", "Transferred").
  - "sick_leave": Sick leave duration/details, if mentioned. State "Not applicable" if the patient is being admitted rather than discharged.
  - "triage_priority": ALWAYS populate this — the patient's triage urgency
    classification. If a triage level wasn't explicitly stated, infer your
    best classification from the clinical context (vitals, primary survey
    findings, presentation) using a standard scale, e.g. "ESI 2 - Emergent"
    (Emergency Severity Index, 1 = most urgent / resuscitation to 5 = least
    urgent / non-urgent), with a brief inline justification, e.g. "ESI 2 -
    Emergent (stable vitals but high-risk presentation)". This is your
    clinical judgement, not a transcription — never leave it blank.
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 codes WITH descriptions, based on your own
    clinical judgement of the case (e.g. ["K35.80 - Unspecified acute
    appendicitis", "R10.31 - Right lower quadrant pain"]). This is your
    recommendation, not a transcription — never return an empty list if the
    text describes any clinical presentation at all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working diagnosis, formatted EXACTLY as "<ICD-10 code> - <diagnosis
    text>", e.g. "K35.80 - Acute appendicitis". Never leave it blank if the
    text describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned
in the text. Do not invent information that is not present or reasonably
implied.
{_history_block(patient_history_text)}
Text: {translated_text}
 
Example output:
{{
  "json_data": {{
    "chief_complaint": "Right lower quadrant abdominal pain",
    "onset": "2 days ago",
    "course": "Progressively worsening",
    "duration": "2 days",
    "past_medical_history": "Hypertension",
    "past_surgical_history": "No past surgical history",
    "medication_history": "Amlodipine 5mg once daily",
    "allergy_status": "No known drug allergies",
    "family_history": "",
    "social_history": "Non-smoker",
    "airway": "Patent",
    "breathing": "Spontaneous, saturating 98% on room air",
    "circulation": "BP 128/82, HR 88, warm and well perfused",
    "disability": "GCS 15, alert and oriented",
    "exposure": "No external injuries noted",
    "examination": "Tenderness and guarding at McBurney's point, positive Rovsing's sign",
    "investigations": "WBC mildly elevated. Abdominal ultrasound pending.",
    "plan_of_care": "Laparoscopic appendectomy",
    "requested_consultation": "General Surgery",
    "discharge_instructions": "Wound care, avoid heavy lifting for 2 weeks, return if fever or worsening pain",
    "patient_disposition": "Admitted for surgery",
    "sick_leave": "5 days",
    "triage_priority": "ESI 3 - Urgent (stable vitals, localized peritoneal signs, needs timely surgical evaluation)",
    "icd10_codes": ["K35.80 - Unspecified acute appendicitis", "R10.31 - Right lower quadrant pain"],
    "final_diagnosis": "K35.80 - Acute appendicitis",
    "final_diagnosis_rationale": "Tenderness and guarding at McBurney's point with a positive Rovsing's sign are classic for acute appendicitis, and the planned laparoscopic appendectomy is consistent with that diagnosis.",
    "history_considered": "No",
    "history_notes": ""
  }},
  "summary": "Chief complaint and history extracted from the presenting complaint. Primary survey (ABCDE) taken from the pre-operative assessment. ICD-10 codes and final diagnosis are the model's clinical recommendation based on the presentation."
}}
"""
 
 
def get_er_extraction_prompt_conversation(translated_text, patient_history_text=""):
    return f"""
You are a medical expert. Given the following doctor-patient conversation
from an Emergency Room (ER) encounter, extract the fields below for the ER
Clinical Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "chief_complaint": Primary reason for the visit — from the patient's opening statements. All patient complaints, signs, and symptoms should be captured here, even if they are not the main reason for the visit.
  - "onset": when and/or how the symptom or complaint started.
    Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
    Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "past_medical_history": Relevant past medical history — from patient's answers. "No Past Medical History" if explicitly stated as such.
  - "past_surgical_history": Relevant past surgical history — from patient's answers. "No Past Surgical History" if explicitly stated as such.
  - "medication_history": Current home medications, from patient's answers. "No Medication History" if explicitly stated as such.
  - "allergy_status": Documented drug/food/other allergies, from patient's answers. "No Known Allergies" if explicitly stated as such.
  - "family_history": Relevant family history, from patient's answers, if mentioned.
  - "social_history": Relevant social history — smoking, alcohol, occupation, living situation — from patient's answers, if mentioned.
  - "airway": Primary survey — Airway (A) findings, from doctor's assessment.
  - "breathing": Primary survey — Breathing (B) findings.
  - "circulation": Primary survey — Circulation (C) findings.
  - "disability": Primary survey — Disability (D), neuro status.
  - "exposure": Primary survey — Exposure (E) findings.
  - "examination": General/focused examination findings, from doctor's statements. Do not include laboratory, ECG, or imaging results here — those belong in "investigations".
  - "investigations": Relevant laboratory, ECG, or imaging results mentioned in the conversation. Leave as "" if none mentioned.
  - "plan_of_care": The intended management plan, from doctor's recommendations. Do not include specific specialty consultations/referrals here — those belong in "requested_consultation".
  - "requested_consultation": Any specialty consultations or referrals requested, if mentioned.
  - "discharge_instructions": Instructions given for discharge. If the patient is being admitted rather than discharged, state that discharge instructions are not applicable due to admission.
  - "patient_disposition": The patient's disposition/destination from the ER (e.g. "Admitted to CCU", "Admitted to ICU", "Discharged home", "Transferred").
  - "sick_leave": Sick leave duration/details, if mentioned. State "Not applicable" if the patient is being admitted rather than discharged.
  - "triage_priority": ALWAYS populate this — the patient's triage urgency
    classification. If a triage level wasn't explicitly stated, infer your
    best classification from the clinical context (vitals, primary survey
    findings, presentation) using a standard scale, e.g. "ESI 2 - Emergent"
    (Emergency Severity Index, 1 = most urgent / resuscitation to 5 = least
    urgent / non-urgent), with a brief inline justification. This is your
    clinical judgement, not a transcription — never leave it blank.
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 codes WITH descriptions, based on doctor's
    diagnosis and patient's symptoms. Never return an empty list if the
    conversation describes any clinical presentation at all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working diagnosis, formatted EXACTLY as "<ICD-10 code> - <diagnosis
    text>", from doctor's conclusions. Never leave it blank if the
    conversation describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction and how it was identified from the conversation.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned.
Do not invent information.
{_history_block(patient_history_text)}
Conversation: {translated_text}
"""
 
 
# ═══════════════════════════════ OPD — extraction ═══════════════════════════
 
def get_opd_extraction_prompt(translated_text, patient_history_text=""):
    return f"""
You are a medical expert. Given the following clinical text from an
Outpatient Department (OPD) visit, extract the fields below for the OPD
Clinical Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "allergy_status": Documented drug/food/other allergies.
  - "chief_complaint": Primary reason for the visit. All patient complaints, signs, and symptoms should be captured here, even if they are not the main reason for the visit.
  - "onset": when and/or how the symptom or complaint started.
    Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
    Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "comorbidities": Existing comorbid conditions.
  - "medication_history": Current medications (part of medical history). "No Medication History" if explicitly stated as such.
  - "past_medical_history": Relevant past medical history — INCLUDE any ongoing
    chronic conditions the patient reports (e.g. diabetes, hypertension,
    asthma) even though the same condition is also captured under
    "comorbidities" above; the two fields feed different parts of the note
    template. Excludes current medications (see medication_history).
  - "past_surgical_history": Relevant past surgical history. "No Past Surgical History" if explicitly stated as such.
  - "social_family_history": Social, marital, or family history.
  - "examination_finding": Examination findings, including vital signs where documented.
  - "plan_of_care": Short/long term plan of care.
  - "procedures_planned": Any investigations, imaging, labs, or procedures planned/ordered, even if also mentioned in the plan of care.
  - "referrals": Referrals made or recommended.
  - "follow_up_timing": When the patient should follow up.
  - "patient_instructions": Instructions given to the patient.
  - "outcome_progress_status": Outcome/progress status if this is a follow-up visit, else "".
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 codes WITH descriptions, based on your own
    clinical judgement of the case. Never return an empty list if the text
    describes any clinical presentation at all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working diagnosis, formatted EXACTLY as "<ICD-10 code> - <diagnosis
    text>", e.g. "J18.9 - Community-acquired pneumonia". Never leave it
    blank if the text describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned
in the text. Do not invent information that is not present or reasonably
implied.
{_history_block(patient_history_text)}
Text: {translated_text}
 
Example output:
{{
  "json_data": {{
    "allergy_status": "No known drug allergies",
    "chief_complaint": "Persistent cough and fever",
    "onset": "Gradual, started 5 days ago",
    "course": "Gradually worsening",
    "duration": "5 days",
    "comorbidities": "Type 2 diabetes",
    "medication_history": "Metformin 500mg twice daily",
    "past_medical_history": "Type 2 diabetes; seasonal allergies",
    "past_surgical_history": "No past surgical history",
    "social_family_history": "Non-smoker, lives with family",
    "examination_finding": "Temp 38.2C, bilateral crackles on auscultation",
    "plan_of_care": "Start antibiotics, symptomatic treatment",
    "procedures_planned": "CBC; Chest X-ray",
    "referrals": "None",
    "follow_up_timing": "1 week",
    "patient_instructions": "Rest, hydration, return if breathless",
    "outcome_progress_status": "",
    "icd10_codes": ["J18.9 - Pneumonia, unspecified organism"],
    "final_diagnosis": "J18.9 - Community-acquired pneumonia",
    "final_diagnosis_rationale": "Fever with bilateral crackles on auscultation over a gradually worsening 5-day course, backed by the ordered chest X-ray, supports a community-acquired pneumonia rather than a simple upper respiratory infection.",
    "history_considered": "No",
    "history_notes": ""
  }},
  "summary": "Chief complaint, comorbidities, and medication history extracted directly from the transcript. ICD-10 codes and final diagnosis are the model's clinical recommendation based on the presentation."
}}
 
Example output WHEN a relevant patient history block IS provided (e.g. this
patient has a documented visit for "E11.65 - Type 2 diabetes mellitus with
poor control" within the last year, and today's presentation is consistent
with a diabetes-related complication):
{{
  "json_data": {{
    "...": "... (other fields as normal) ...",
    "icd10_codes": ["E11.65 - Type 2 diabetes mellitus with poor control", "E11.9 - Type 2 diabetes mellitus without complications"],
    "final_diagnosis": "E11.65 - Type 2 diabetes mellitus with poor control",
    "final_diagnosis_rationale": "Today's presentation is consistent with a diabetes-related complication, and the patient has a documented history of Type 2 diabetes mellitus with poor control from a prior visit — carried forward as the likely underlying diagnosis.",
    "history_considered": "Yes",
    "history_notes": "Patient has a documented history of Type 2 diabetes mellitus with poor control from a prior visit — carried forward as the likely underlying diagnosis given today's presentation."
  }},
  "summary": "..."
}}
"""
 
 
def get_opd_extraction_prompt_conversation(translated_text, patient_history_text=""):
    return f"""
You are a medical expert. Given the following doctor-patient conversation
from an Outpatient Department (OPD) visit, extract the fields below for the
OPD Clinical Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "allergy_status": Documented drug/food/other allergies.
  - "chief_complaint": Primary reason for the visit — from patient's initial statements. All patient complaints, signs, and symptoms should be captured here, even if they are not the main reason for the visit.
  - "onset": when and/or how the symptom or complaint started.
      Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
      Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "medication_history": Current medications mentioned by patient or prescribed by doctor. "No Medication History" if explicitly stated as such.
  - "past_medical_history": Relevant past medical history — INCLUDE any ongoing
    chronic conditions the patient reports (e.g. diabetes, hypertension,
    asthma) even though the same condition is also captured under
    "comorbidities" above; the two fields feed different parts of the note
    template. Excludes current medications (see medication_history).
  - "past_surgical_history": Relevant past surgical history from patient's answers. "No Past Surgical History" if explicitly stated as such.
  - "social_family_history": Social, marital, or family history.
  - "examination_finding": Examination findings discussed by the doctor, including vital signs where documented.
  - "plan_of_care": Short/long term plan of care, from doctor's recommendations.
  - "procedures_planned": Any investigations, imaging, labs, or procedures planned/ordered, even if also mentioned in the plan of care.
  - "referrals": Referrals made or recommended.
  - "follow_up_timing": When the patient should follow up.
  - "patient_instructions": Instructions given to the patient, from doctor's closing remarks.
  - "outcome_progress_status": Outcome/progress status if this is a follow-up visit, else "".
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 codes WITH descriptions, based on doctor's
    diagnosis and patient's symptoms. Never return an empty list if the
    conversation describes any clinical presentation at all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working diagnosis, formatted EXACTLY as "<ICD-10 code> - <diagnosis
    text>", from doctor's conclusions. Never leave it blank if the
    conversation describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction and how it was identified from the conversation.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned.
Do not invent information.
{_history_block(patient_history_text)}
Conversation: {translated_text}
"""
 
 
# ══════════════════════════════ Dental — extraction ═════════════════════════
 
def get_dental_extraction_prompt(translated_text, patient_history_text=""):
    return f"""
You are a dental clinical documentation expert. Given the following clinical
text from a dental visit, extract the fields below for the Dental Clinical
Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "chief_complaint": Primary reason for the visit. All patient complaints, signs, and symptoms should be captured here, even if they are not the main reason for the visit.
  - "onset": when and/or how the symptom or complaint started.
      Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
      Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "past_medical_history": Relevant past medical history mentioned (e.g. systemic conditions, current medications affecting treatment). "No Past Medical History" if explicitly stated as such.
  - "consent_obtained": Legal/ethical consent status — "Yes", "No", "Verbal", or "Not Applicable".
  - "treatment_plan_status": Milestone of the overall care plan — "Started", "In-Progress", or "Completed".
  - "charge_category": Billing tier — "Standard", "Insurance", or "Discounted".
  - "patient_education_given": Whether post-procedure care instructions were communicated — "Yes" or "No".
  - "radiographs_taken": Which X-ray type(s) were captured and at which stage — Pre-treatment / Intra-treatment / Post-treatment — using modalities such as PA (Periapical), BW (Bitewing), OPG (Panoramic), or CBCT (e.g. "Pre-treatment: Periapical").
  - "tooth_number": Tooth identifier(s) using Universal (1-32) or FDI notation.
  - "subspecialty_focus": Dental branch governing the procedure (Endodontics, Periodontics, Orthodontics, Prosthodontics, Surgery, Pedo, or General).
  - "procedure_executed": Short summary of the actual clinical intervention performed.
  - "materials_inventory": Specific brands, sizes, or materials used.
  - "re_make": Whether a restoration/appliance/impression had to be redone — "Yes" or "No".
  - "complication_if_present": Any adverse events or unexpected issues, or "" if none.
  - "internal_referral": Whether patient was referred to another specialist within the practice, and which specialty.
  - "service_leakage_alert": Lost revenue/care gap — External Referral Reason or Patient Refusal, or "" if none.
  - "next_visit_tooth_number": Target tooth for the upcoming appointment.
  - "next_visit_treatment": Planned procedure/goal for the next appointment.
  - "retention_priority": Triage label for how critical it is to retain this patient (e.g. "High", "Medium", "Low").
  - "followup_timeline": Scheduled timeframe for the next check-in.
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 (or relevant dental diagnostic) codes WITH
    descriptions, based on your own clinical judgement of the case. Never
    return an empty list if the text describes any clinical presentation at
    all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working dental diagnosis, formatted EXACTLY as "<ICD-10 code> -
    <diagnosis text>", e.g. "K04.0 - Irreversible pulpitis, tooth 36". Never
    leave it blank if the text describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned
in the text. Do not invent information that is not present or reasonably
implied.
{_history_block(patient_history_text)}
Text: {translated_text}
 
Example output:
{{
  "json_data": {{
    "chief_complaint": "Severe pain on the lower left back tooth when chewing",
    "onset": "1 week ago",
    "course": "Progressively worsening",
    "duration": "1 week",
    "past_medical_history": "Controlled hypertension, on Amlodipine",
    "consent_obtained": "Yes",
    "treatment_plan_status": "In-Progress",
    "charge_category": "Insurance",
    "patient_education_given": "Yes",
    "radiographs_taken": "Pre-treatment: Periapical",
    "tooth_number": "FDI 36",
    "subspecialty_focus": "Endodontics",
    "procedure_executed": "Root canal treatment, first sitting",
    "materials_inventory": "Rotary NiTi files, calcium hydroxide dressing",
    "re_make": "No",
    "complication_if_present": "",
    "internal_referral": "",
    "service_leakage_alert": "",
    "next_visit_tooth_number": "FDI 36",
    "next_visit_treatment": "Root canal obturation",
    "retention_priority": "High",
    "followup_timeline": "1 week",
    "icd10_codes": ["K04.0 - Pulpitis"],
    "final_diagnosis": "K04.0 - Irreversible pulpitis, tooth 36",
    "final_diagnosis_rationale": "Severe pain on chewing localized to tooth 36, together with a positive periapical finding, is consistent with irreversible pulpitis rather than reversible pulpal inflammation.",
    "history_considered": "No",
    "history_notes": ""
  }},
  "summary": "Chief complaint and procedure details extracted from the documented clinical steps. ICD-10 codes and final diagnosis are the model's clinical recommendation based on the presentation."
}}
"""
 
 
def get_dental_extraction_prompt_conversation(translated_text, patient_history_text=""):
    return f"""
You are a dental clinical documentation expert. Given the following
doctor-patient conversation from a dental visit, extract the fields below for
the Dental Clinical Note template. Return a JSON object with two fields:
- "json_data": A dictionary with exactly these keys (all strings, except
  icd10_codes which is a list of strings):
  - "chief_complaint": Primary reason for the visit — from the patient's opening statements.
  - "onset": when and/or how the symptom or complaint started.
      Include the reported starting point and whether the onset was sudden, gradual, acute, or insidious, if explicitly mentioned.
      Do not infer the onset if it is not stated.
  - "course": Extract how the symptom or complaint has changed or behaved over time since it started.
    Examples include improving, worsening, progressively worsening, stable, intermittent, recurrent, persistent, fluctuating, or unchanged.
    Only report the course if it is explicitly stated or clearly described in the conversation.
    Do not infer a course from the severity of the symptom alone.
  - "duration": Extract how long the symptom or complaint has been present.
    Include the exact duration when available, such as hours, days, weeks, months, or years. including any recent change in severity if mentioned (e.g. "2 months, worse over the last 3 weeks").
    Do not calculate or infer a duration unless it is explicitly supported by the patient's statement.
  - "past_medical_history": Relevant past medical history mentioned (e.g. systemic conditions, current medications affecting treatment). "No Past Medical History" if explicitly stated as such.
  - "consent_obtained": Legal/ethical consent status — "Yes", "No", "Verbal", or "Not Applicable".
  - "treatment_plan_status": Milestone of the overall care plan — "Started", "In-Progress", or "Completed".
  - "charge_category": Billing tier — "Standard", "Insurance", or "Discounted".
  - "patient_education_given": Whether post-procedure care instructions were communicated — "Yes" or "No".
  - "radiographs_taken": Which X-ray type(s) were captured and at which stage — Pre-treatment / Intra-treatment / Post-treatment — using modalities such as PA (Periapical), BW (Bitewing), OPG (Panoramic), or CBCT.
  - "tooth_number": Tooth identifier(s) using Universal (1-32) or FDI notation.
  - "subspecialty_focus": Dental branch governing the procedure.
  - "procedure_executed": Short summary of the actual clinical intervention performed, from doctor's statements.
  - "materials_inventory": Specific brands, sizes, or materials used.
  - "re_make": Whether a restoration/appliance/impression had to be redone — "Yes" or "No".
  - "complication_if_present": Any adverse events or unexpected issues mentioned, or "" if none.
  - "internal_referral": Whether patient was referred internally, and which specialty.
  - "service_leakage_alert": External Referral Reason or Patient Refusal, or "" if none.
  - "next_visit_tooth_number": Target tooth for the upcoming appointment.
  - "next_visit_treatment": Planned procedure/goal for the next appointment.
  - "retention_priority": Triage label for how critical it is to retain this patient.
  - "followup_timeline": Scheduled timeframe for the next check-in.
  - "icd10_codes": ALWAYS populate this — a list of your recommended
    possible/differential ICD-10 (or relevant dental diagnostic) codes WITH
    descriptions. Never return an empty list if the conversation describes
    any clinical presentation at all.
  - "final_diagnosis": ALWAYS populate this — your single best-supported
    working dental diagnosis, formatted EXACTLY as "<ICD-10 code> -
    <diagnosis text>", from doctor's conclusions. Never leave it blank if
    the conversation describes any clinical presentation at all.
{_FINAL_DIAGNOSIS_RATIONALE_DOC}
{_HISTORY_FIELDS_DOC}
- "summary": A short string explaining the rationale behind the extraction and how it was identified from the conversation.
 
For every field EXCEPT icd10_codes, final_diagnosis, final_diagnosis_rationale,
history_considered, and history_notes: leave it as "" if it is not mentioned.
Do not invent information.
{_history_block(patient_history_text)}
Conversation: {translated_text}
"""