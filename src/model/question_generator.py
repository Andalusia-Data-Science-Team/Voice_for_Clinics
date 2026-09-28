import logging
from model.department_fields import get_department_fields

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return str(value).strip() == ""


class QuestionGenerator:
    """
    Builds the "questions for the doctor" list directly from the already-
    extracted feature JSON — deterministically, with no separate LLM call.

    This intentionally replaces an earlier version that asked the LLM to
    re-scan the raw transcript for missing fields: that second, independent
    pass could disagree with the extraction step (flagging a field as
    "not mentioned" even though it had just been successfully extracted).
    Deriving the question list from the same `extracted_features` dict the
    Feature extraction card is built from guarantees the two can never
    contradict each other.

    icd10_codes and final_diagnosis are never asked about here — they're
    always AI-recommended by the extraction step (see the extraction
    prompts), not something the doctor is expected to dictate.
    """

    @staticmethod
    def generate(extracted_features: dict, department: str = "opd"):
        """
        Args:
            extracted_features: The `json_data` dict returned by
                ExtractFeature.extract() for this same transcript.
            department: "er", "opd", or "dental" — selects which field
                checklist to check against. Defaults to "opd".

        Returns:
            Tuple: (questions: list[dict], reasoning: str)
            Each question dict has: question, answer (always None),
            needs_asking (always True), category (the field's display label).
        """
        fields = get_department_fields(department)
        src = extracted_features or {}

        questions = []
        for f in fields:
            if _is_empty(src.get(f["key"])):
                questions.append({
                    "question": f["question"],
                    "answer": None,
                    "needs_asking": True,
                    "category": f["label"],
                })

        logger.info(
            f"[QuestionGenerator] {len(questions)}/{len(fields)} field(s) missing for department={department}"
        )
        return questions, ""