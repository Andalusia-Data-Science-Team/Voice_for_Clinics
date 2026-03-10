def get_refine_arabic_prompt_deepseek(raw_text):
    return f"""
Act as a senior medical transcription editor specializing in Arabic healthcare documentation.

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
→ Do not use astrisks at all

**CORRECTED MEDICAL TEXT:**
"""


def get_refine_english_prompt_deepseek(translated_text):
    return f"""
    Correct the grammar and structure of this English medical text.
    Return only the corrected English text with no additional explanations.
    Keep brand and generic drug names unchanged
    Ensure refining any mentioned medications
    Do not use astrisks at all
    ORIGINAL TEXT:
    \"\"\"{translated_text}\"\"\"
    **ONLY return the refinment no more**
    """


# --- Conversation Mode Prompts ---
def get_refine_arabic_prompt_deepseek_conversation(raw_text):
    return f"""
    Act as a clinical transcription editor. Refine this Arabic medical conversation for accuracy and clarity.

    **CONVERSATION TO PROCESS:**
    {raw_text}

    **SPEAKER IDENTIFICATION:**
    - **الدكتور:** Medical professional (asks questions, examines, diagnoses, prescribes)
    - **المريض:** Patient (describes symptoms, answers questions, shares history)

    **QUALITY STANDARDS:**
    1. **Accuracy:** Preserve all medical content, terminology, and clinical context
    2. **Clarity:** Fix grammatical errors and improve sentence flow
    3. **Format:** Clearly label each speaker turn with **الدكتور:** or **المريض:**
    4. **Anonymization:** Minimize personal identifiers while keeping dialogue intact
    5. **Professionalism:** Use formal medical Arabic appropriate for clinical documentation

    **CRITICAL:**
    → Output ONLY the refined conversation in Arabic
    → No additional text, explanations, or commentary
    → Maintain original dialogue sequence and medical meaning
    → Each speaker turn must be properly labeled
    → Keep brand and generic drug names unchanged
    → Ensure refining any mentioned medications
    → Do not use astrisks at all

    **REFINED CLINICAL CONVERSATION:**
"""


def get_refine_english_prompt_deepseek_conversation(translated_text):
    return f"""
Refine this English medical conversation while preserving the dialogue structure.

Instructions:
1. Correct grammar, punctuation, and phrasing for natural spoken English.
2. Ensure medical terminology and tone are appropriate for a clinical dialogue.
3. Keep brand and generic drug names unchanged. 
4. Ensure refining any mentioned medications
5. Maintain the natural conversational flow.
6. Use **Doctor:** and **Patient:** exactly as shown below (capitalize only the first letter).
7. Return ONLY the refined conversation in dialogue format — no introductions, explanations, or extra text.

Example format:
Doctor: Good morning. What brings you in today?
Patient: I’ve had chest pain for a few days.

Doctor: What kind of pain?
Patient: A pressure in the center of my chest, worse when I climb stairs.

ORIGINAL TEXT:
\"\"\"{translated_text}\"\"\"

Only return the dialogue no more **Without any additional text or astrisks**
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


def get_extraction_prompt_llama(translated_text):
    return f"""
You are a medical expert Given the following medical text, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing the following medical features:
  - "chief_complaint": The primary reason for the visit (string) you will get a diagnose of a patient so you must output a chief complain.
  - "icd10_codes": A list of ICD-10 codes with descriptions (list of strings) RECOMMEND RELATED ICD10 codes as most as you can.
  - "history_of_illness": Details of the patient's medical history (string).
  - "current_medication": Current medications prescribed or taken (string).
  - "imaging_results": Results from imaging studies (string).
  - "plan": Treatment or management plan (string).
  - "assessment": Clinical assessment or diagnosis (string).
  - "follow_up": Follow-up instructions (string).
- "reasoning": A string explaining the rationale behind the extracted features.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the text.

Text: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient has a history of asthma and seasonal allergies.",
    "current_medication": "Albuterol inhaler, Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray shows no consolidation.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol as needed.",
    "assessment": "Influenza with acute respiratory symptoms",
    "follow_up": "Return in 7 days or sooner if symptoms worsen."
  }},
  "reasoning": "The text describes a patient with cough and fever, leading to a diagnosis of influenza. ICD-10 codes J11.1 and R05 are assigned based on the symptoms. The history of asthma and allergies is noted. Current medications include Oseltamivir for influenza and Albuterol for asthma. Chest X-ray is normal, supporting a viral etiology. The plan includes antiviral treatment and symptom management, with a follow-up in 7 days."
}}
"""


# --- Conversation Mode Extraction ---
def get_extraction_prompt_llama_conversation(translated_text):
    return f"""
You are a medical expert. Given the following medical conversation between a doctor and patient, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing the following medical features:
  - "chief_complaint": The primary reason for the visit (string) - extract from patient's initial statements.
  - "icd10_codes": A list of ICD-10 codes with descriptions (list of strings) - based on doctor's diagnosis and patient's symptoms.
  - "history_of_illness": Details of the patient's medical history (string) - gather from patient's responses about their medical background.
  - "current_medication": Current medications prescribed or taken (string) - mentioned by patient or prescribed by doctor.
  - "imaging_results": Results from imaging studies (string) - discussed by doctor.
  - "plan": Treatment or management plan (string) - extract from doctor's recommendations.
  - "assessment": Clinical assessment or diagnosis (string) - based on doctor's conclusions.
  - "follow_up": Follow-up instructions (string) - extract from doctor's final instructions.
  - "conversation_summary": A brief summary of the consultation (string) - optional overview of the interaction.
- "reasoning": A string explaining the rationale behind the extracted features and how they were identified from the conversation.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the conversation.

Conversation: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever for 5 days",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient reports history of asthma since childhood and seasonal allergies.",
    "current_medication": "Albuterol inhaler as needed, Doctor prescribed Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray performed today shows no consolidation or pneumonia.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol inhaler as needed for breathing difficulty, rest and fluids.",
    "assessment": "Influenza with acute respiratory symptoms, no complications noted.",
    "follow_up": "Return in 7 days for re-evaluation, or sooner if symptoms worsen or breathing difficulty increases.",
    "conversation_summary": "Patient presented with 5-day history of cough and fever. Examination and chest X-ray ruled out pneumonia. Diagnosed with influenza and prescribed antiviral treatment."
  }},
  "reasoning": "The chief complaint was identified from the patient's opening statement about cough and fever. ICD-10 codes were assigned based on the doctor's diagnosis of influenza. Medical history was gathered from patient's responses about asthma and allergies. Current medication includes the patient's existing inhaler and newly prescribed Oseltamivir. Imaging results were shared by the doctor during the consultation. The treatment plan and follow-up instructions were extracted from the doctor's recommendations at the end of the visit."
}}
"""


def get_dynamic_extraction_prompt_llama(translated_text):
    return f"""
You are a medical expert Given the following medical text, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing this medical features:
  chief_complaint, icd10_codes, history_of_illness, current_medication,
  imaging_results, plan, assessment, follow_up
- "reasoning": A string explaining the rationale behind the extracted features.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the text.

Text: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient has a history of asthma and seasonal allergies.",
    "current_medication": "Albuterol inhaler, Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray shows no consolidation.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol as needed.",
    "assessment": "Influenza with acute respiratory symptoms",
    "follow_up": "Return in 7 days or sooner if symptoms worsen."
  }},
  "reasoning": "The text describes a patient with cough and fever, leading to a diagnosis of influenza. ICD-10 codes J11.1 and R05 are assigned based on the symptoms. The history of asthma and allergies is noted. Current medications include Oseltamivir for influenza and Albuterol for asthma. Chest X-ray is normal, supporting a viral etiology. The plan includes antiviral treatment and symptom management, with a follow-up in 7 days."
}}
"""

# --- Conversation Mode Dynamic Extraction ---
def get_dynamic_extraction_prompt_llama_conversation(translated_text):
    return f"""
You are a medical expert. Given the following medical conversation between a doctor and patient, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing these medical features:
  chief_complaint, icd10_codes, history_of_illness, current_medication,
  imaging_results, plan, assessment, follow_up
- "reasoning": A string explaining the rationale behind the extracted features and how they were identified from the conversation.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the conversation.

Instructions:
- Extract information from both doctor's statements and patient's responses
- Chief complaint should come from patient's initial description
- Medical history should be gathered from patient's answers
- Treatment plans and assessments should come from doctor's recommendations
- Add "conversation_summary" field if not in features list

Conversation: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever for 5 days",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient reports history of asthma and seasonal allergies.",
    "current_medication": "Albuterol inhaler, Doctor prescribed Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray shows no consolidation.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol as needed.",
    "assessment": "Influenza with acute respiratory symptoms",
    "follow_up": "Return in 7 days or sooner if symptoms worsen."
    "conversation_summary": "Patient presented with 5-day history of cough and fever. Examination and chest X-ray ruled out pneumonia. Diagnosed with influenza and prescribed antiviral treatment."
  }},
  "reasoning": "Chief complaint identified from patient's opening description. Medical history extracted from patient's responses about previous conditions. Doctor's diagnosis informed ICD-10 code selection. Treatment plan based on doctor's recommendations during consultation. Follow-up instructions from doctor's closing remarks."
}}
"""


# --- Question Generation Prompts ---
def get_question_generation_prompt_llama(translated_text):
    return f"""
    You are a medical expert. Based on the following medical dictation by a doctor according to the patient condition, generate a list of essential medical questions that the doctor MUST ask or should have asked the patient.

    For each question:
    1. Determine if the answer is already provided in the text
    2. If answered, extract the answer
    3. If not answered, mark it as "needs_asking" so the doctor knows to ask the patient

    Return a JSON object with:
    - "questions": A list of objects, each containing:
      - "question": The medical question (string)
      - "answer": The answer if found in text, or null if not mentioned (string or null)
      - "needs_asking": true if doctor needs to ask this, false if already answered (boolean)
      - "category": The category of the question - one of: "chief_complaint", "history", "medications", "allergies", "vital_signs", "physical_exam", "assessment", "plan" (string)
    - "reasoning": Brief explanation of the analysis (string)

    **MEDICAL CATEGORIES TO COVER:**
    - Chief Complaint: Why is the patient here?
    - Medical History: Past illnesses, surgeries, chronic conditions
    - Current Medications: What drugs is the patient taking?
    - Allergies: Drug or food allergies
    - Vital Signs: Blood pressure, temperature, pulse, etc.
    - Physical Examination: Relevant findings
    - Assessment/Diagnosis: What's the clinical impression?
    - Plan: Treatment, follow-up, referrals

    ** The questions must be generated differently based on patient condition **

    Text: {translated_text}

    Example output:
    {{
      "questions": [
        {{
          "question": "What is the patient's chief complaint?",
          "answer": "Persistent cough and fever for 5 days",
          "needs_asking": false,
          "category": "chief_complaint"
        }},
        {{
          "question": "Does the patient have any known drug allergies?",
          "answer": null,
          "needs_asking": true,
          "category": "allergies"
        }},
        {{
          "question": "What is the patient's current blood pressure?",
          "answer": "120/80 mmHg",
          "needs_asking": false,
          "category": "vital_signs"
        }},
        {{
          "question": "Has the patient had any recent surgeries?",
          "answer": null,
          "needs_asking": true,
          "category": "history"
        }}
      ],
      "reasoning": "Generated essential medical questions covering all standard categories. Some questions were answered in the dictation (chief complaint, vital signs), while others need to be asked (allergies, surgical history) to complete the medical record."
    }}
    """


def get_question_generation_prompt_llama_conversation(translated_text):
    return f"""
    You are a medical expert. Based on the following medical conversation between a doctor and patient, generate a list of essential medical questions that were either:
    1. Already asked and answered during the conversation
    2. Should have been asked according to patient condition but were not

    For each question:
    1. Check if it was discussed in the conversation
    2. If answered, extract the answer from patient's response
    3. If not asked, mark it as "needs_asking" so the doctor knows to ask

    Return a JSON object with:
    - "questions": A list of objects, each containing:
      - "question": The medical question (string)
      - "answer": The answer from conversation, or null if not discussed (string or null)
      - "needs_asking": true if doctor should still ask this, false if already covered (boolean)
      - "category": The category - one of: "chief_complaint", "history", "medications", "allergies", "vital_signs", "physical_exam", "assessment", "plan" (string)
    - "reasoning": Brief explanation of the conversation analysis (string)

    **MEDICAL CATEGORIES TO COVER:**
    - Chief Complaint: Why is the patient here?
    - Medical History: Past illnesses, surgeries, chronic conditions
    - Current Medications: What drugs is the patient taking?
    - Allergies: Drug or food allergies
    - Vital Signs: Blood pressure, temperature, pulse, etc.
    - Physical Examination: Relevant findings
    - Assessment/Diagnosis: What's the clinical impression?
    - Plan: Treatment, follow-up, referrals

    ** The questions must be generated differently based on patient condition **

    Conversation: {translated_text}

    Example output:
    {{
      "questions": [
        {{
          "question": "What brings you to the clinic today?",
          "answer": "Persistent cough and fever for 5 days",
          "needs_asking": false,
          "category": "chief_complaint"
        }},
        {{
          "question": "Do you have any chronic medical conditions?",
          "answer": "Patient mentioned history of asthma since childhood",
          "needs_asking": false,
          "category": "history"
        }},
        {{
          "question": "Are you allergic to any medications?",
          "answer": null,
          "needs_asking": true,
          "category": "allergies"
        }},
        {{
          "question": "What medications are you currently taking?",
          "answer": "Albuterol inhaler as needed",
          "needs_asking": false,
          "category": "medications"
        }},
        {{
          "question": "Have you had any recent weight changes?",
          "answer": null,
          "needs_asking": true,
          "category": "history"
        }}
      ],
      "reasoning": "Analyzed the doctor-patient conversation. The doctor asked about chief complaint, medical history, and current medications which the patient answered. However, important questions about allergies and recent weight changes were not discussed and should be asked to complete the medical assessment."
    }}
    """

# def get_extraction_prompt_deepseek(translated_text):
#     return f"""
#     Extract patient information from this medical text into exactly two sections:

#     # SECTION 1: PATIENT DATA (JSON FORMAT)
#     ```json
#     {{
#     "chief_complaint": "",
#     "icd10_codes": [
#         "Code1 - Description",
#         "Code2 - Description"
#     ],
#     "history_of_illness": "",
#     "current_medication": "",
#     "imaging_results": "",
#     "plan": "",
#     "assessment": "",
#     "follow_up": ""
#     }}
#     ```

#     # SECTION 2: ANALYSIS NOTES
#     Brief justification for data extraction and ICD10 code selections.

#     Rules:
#     - Use 'null' for missing data
#     - Include all relevant ICD10 codes with descriptions
#     - Properly format JSON with no explanatory text inside

#     TEXT:
#     \"\"\"{translated_text}\"\"\"
#     """


# def get_refine_arabic_prompt_llama(raw_text):
#     return f"""
# SYSTEM: You are a medical language processor that outputs ONLY corrected text with NO explanations.

# USER: Correct this Arabic medical text:
# \"\"\"{raw_text}\"\"\"

# ASSISTANT:
# """


# def get_translation_prompt_llama(refined_text):
#     return f"""
# SYSTEM: You are a translation system that outputs ONLY the English translation with NO explanations.

# USER: Translate to English:
# \"\"\"{refined_text}\"\"\"

# ASSISTANT:
# """

# def get_refine_english_prompt_llama(translated_text):
#     return f"""
# SYSTEM: You are a text processor that outputs ONLY the corrected English text with NO explanations.

# USER: Correct this medical text:
# \"\"\"{translated_text}\"\"\"

# ASSISTANT:
# """


# def get_refine_english_prompt_deepseek_conv(translated_text):
#     return f"""
#     Refine this English medical conversation while preserving speaker labels:
#     1. Fix any grammatical errors or awkward phrasing
#     2. Ensure medical terminology is used correctly
#     3. Maintain the conversational flow and natural dialogue
#     4. Preserve the **DOCTOR:** and **PATIENT:** labels
    
#     Return ONLY the refined English conversation without any additional commentary.
#     ORIGINAL TEXT:
# \"\"\"{translated_text}\"\"\"
#     """


# def get_refine_arabic_prompt_llama_conv(raw_text):
#     return f"""
# SYSTEM: You are an Arabic medical transcription system that outputs ONLY formatted text with speaker labels. Any additional text will trigger system errors.

# USER: Format with **DOCTOR:** and **PATIENT:** labels:
# \"\"\"{raw_text}\"\"\"

# ASSISTANT:
# """


# def get_refine_english_prompt_llama_conv(translated_text):
#     return f"""
# SYSTEM: You are a text processor that outputs ONLY the refined English conversation with speaker labels. No explanations.

# USER: Refine this conversation:
# \"\"\"{translated_text}\"\"\"

# ASSISTANT:
# """


# def get_refine_arabic_prompt_deepseek_conv(raw_text):
#     return f"""
#     Analyze this Arabic medical conversation in different dialects and format it properly it usually starts with greeting from one of the speakers:
    
#     1. Identify speakers based on these clear role indicators:
#        - DOCTOR: The person who asks questions, examines the patient, provides diagnoses, and recommends treatments
#        - PATIENT: The person who describes symptoms, answers questions, explains concerns, and shares their medical history
    
#     2. Format the conversation by labeling each speaker as **DOCTOR:** or **PATIENT:** before their dialogue
    
#     3. Return ONLY the properly formatted Arabic dialogue with speaker labels, without any additional commentary or explanation
    
#     ORIGINAL TEXT:
#     \"\"\"{raw_text}\"\"\"
#     """


# def get_translation_prompt_deepseek_conv(refined_text):
#     return f"""
#     Translate this Arabic medical conversation to English.
#     Preserve speaker labels (**DOCTOR:** and **PATIENT:**).
#     Return only the English translation without commentary.

#     ARABIC TEXT:
#     \"\"\"{refined_text}\"\"\"
#     """


# def get_translation_prompt_llama_conv(refined_text):
#     return f"""
# SYSTEM: You are a translation system that outputs ONLY the English translation with speaker labels preserved. No explanations.

# USER: Translate with **DOCTOR:** and **PATIENT:** labels:
# \"\"\"{refined_text}\"\"\"

# ASSISTANT:
# """