def _specialty_context_line(specialty):
    """Shared context line inserted into refine prompts when a specialty is set."""
    if not specialty:
        return ""
    return (
        f"\nMEDICAL SPECIALTY: {specialty}\n"
        f"Pay particular attention to {specialty}-specific terminology, abbreviations, "
        f"procedure names, and drug names common in this specialty when correcting the text.\n"
    )


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
