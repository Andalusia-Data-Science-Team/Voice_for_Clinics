"""
evaluation/engine.py
====================
Automated evaluation engine:
- Evaluates pipeline sessions against ground truth GoldCase documents
- Computes per-field accuracy, similarity, completeness, and diagnosis matches
- Generates composite scores and persists evaluation runs into MongoDB
- Supports single-session evaluation as well as batch evaluation
"""
from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from db.repositories import GoldDatasetRepository, EvalRepository, SessionRepository
from schemas.gold_schemas import GoldCase
from evaluation.metrics import (
    FieldScore,
    compare_fields,
    compare_diagnosis,
    calc_accuracy,
    calc_completeness,
    calc_f1,
    fuzzy_similarity,
    compute_composite_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Complete evaluation report for a single case vs session execution."""
    eval_id: str
    case_id: str
    session_id: str
    model_id: str
    department: str
    field_scores: List[FieldScore]
    extraction_accuracy: float
    extraction_completeness: float
    f1_score: float
    diagnosis_match: bool
    total_score: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass instance to a clean JSON-serializable dictionary."""
        d = asdict(self)
        d["field_diffs"] = {
            f.field_name: {
                "expected": f.expected,
                "actual": f.actual,
                "match": f.match,
                "similarity": f.similarity,
            }
            for f in self.field_scores
        }
        return d


class EvaluationEngine:
    """Engine orchestrating ground-truth comparison and evaluation persistence."""

    def __init__(
        self,
        gold_repo: GoldDatasetRepository,
        eval_repo: EvalRepository,
        session_repo: SessionRepository,
    ):
        self._gold = gold_repo
        self._eval = eval_repo
        self._sessions = session_repo

    async def evaluate_session(
        self,
        session_id: str,
        case_id: str,
        similarity_threshold: float = 0.85,
    ) -> EvalResult:
        """Compare one completed session against a specific gold case and persist results."""
        gold = await self._gold.get_case(case_id)
        if not gold:
            raise ValueError(f"Gold case '{case_id}' not found in database")

        session = await self._sessions.get_session(session_id)
        if not session:
            raise ValueError(f"Pipeline session '{session_id}' not found in database")

        expected_extraction = gold.get("expected_extraction") or {}
        actual_extraction = session.get("extracted_features") or {}

        field_scores = compare_fields(
            expected=expected_extraction,
            actual=actual_extraction,
            similarity_threshold=similarity_threshold,
        )

        acc = calc_accuracy(field_scores)
        comp = calc_completeness(field_scores)
        f1 = calc_f1(field_scores)

        diag_match = compare_diagnosis(
            expected=gold.get("expected_diagnosis"),
            actual=session.get("diagnosis"),
        )

        avg_sim = (
            sum(s.similarity for s in field_scores) / len(field_scores)
            if field_scores else 1.0
        )

        composite = compute_composite_score(
            accuracy=acc,
            completeness=comp,
            diagnosis_match=diag_match,
            avg_similarity=avg_sim,
        )

        eval_id = str(uuid.uuid4())
        model_id = session.get("extraction_model") or session.get("reasoning_model") or "unknown"

        result = EvalResult(
            eval_id=eval_id,
            case_id=case_id,
            session_id=session_id,
            model_id=model_id,
            department=gold.get("department", session.get("department", "unknown")),
            field_scores=field_scores,
            extraction_accuracy=acc,
            extraction_completeness=comp,
            f1_score=f1,
            diagnosis_match=diag_match,
            total_score=composite,
        )

        # Persist eval run into MongoDB
        await self._eval.save_run(result.to_dict())
        logger.info(
            "[Evaluation] Finished eval_id=%s for case=%s session=%s (accuracy=%.2f, score=%.2f)",
            eval_id, case_id, session_id, acc, composite,
        )
        return result

    async def run_batch(
        self,
        department: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[EvalResult]:
        """Fetch all active gold cases, search for matching sessions, and produce evaluation results."""
        cases = await self._gold.list_cases(department=department, active_only=True)
        results: List[EvalResult] = []

        for case_doc in cases:
            case_id = case_doc["case_id"]
            # Look up recent session with this case_id or session_id
            session = await self._sessions.get_session(case_id)
            if not session:
                # Also try finding session where session_id matches case_id pattern
                candidate_sessions = await self._sessions.list_sessions(
                    department=case_doc.get("department"),
                    limit=50,
                )
                session = next(
                    (s for s in candidate_sessions if s.get("session_id") == case_id),
                    None,
                )

            if session:
                res = await self.evaluate_session(session["session_id"], case_id)
                results.append(res)

        return results
