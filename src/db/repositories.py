"""
Typed repository classes — wrappers around Motor collections
that enforce schema validation before writes and deliver high-performance querying.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from schemas.output_schemas import PipelineSessionOutput
from schemas.gold_schemas import GoldCase
from db.models import GoldCaseModel, EvalRunModel

logger = logging.getLogger(__name__)


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB _id ObjectId and datetime fields into JSON-serializable primitives."""
    res = dict(doc)
    if "_id" in res:
        res["_id"] = str(res["_id"])
    if "created_at" in res and isinstance(res["created_at"], datetime):
        res["created_at"] = res["created_at"].isoformat()
    if "updated_at" in res and isinstance(res["updated_at"], datetime):
        res["updated_at"] = res["updated_at"].isoformat()
    return res


def _serialize_mongo_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if doc is None:
        return None
    return _serialize_doc(doc)


class SessionRepository:
    """CRUD for sessions + pipeline_runs collections with optional high-performance write-behind buffer."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        batch_size: int = 20,
        flush_interval_seconds: float = 2.0,
        enable_buffer: bool = False,
    ):
        self._db = db
        self._sessions = db.sessions
        self._runs = db.pipeline_runs
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.1, flush_interval_seconds)
        self._enable_buffer = enable_buffer
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task[Any]] = None

        if self._enable_buffer:
            self._flush_task = asyncio.create_task(self._periodic_flusher())

    async def _periodic_flusher(self) -> None:
        """Periodic background task to flush buffer every T seconds."""
        try:
            while True:
                await asyncio.sleep(self._flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            await self.flush()

    async def flush(self) -> int:
        """Flush all buffered session writes to MongoDB."""
        async with self._lock:
            if not self._buffer:
                return 0
            to_flush = list(self._buffer)
            self._buffer.clear()

        flushed_count = 0
        try:
            from pymongo import UpdateOne
            operations = []
            for doc in to_flush:
                session_id = doc.get("session_id")
                if session_id:
                    operations.append(
                        UpdateOne(
                            {"session_id": session_id},
                            {"$set": doc},
                            upsert=True,
                        )
                    )
            if operations:
                res = await self._sessions.bulk_write(operations, ordered=False)
                flushed_count = (res.upserted_count or 0) + (res.modified_count or 0)
                logger.debug("[SessionRepo] Flushed %d buffered writes to MongoDB", len(operations))
        except Exception as exc:
            logger.error("[SessionRepo] Bulk flush failed: %s", exc)
            # Re-queue on failure so writes are not lost
            async with self._lock:
                self._buffer = to_flush + self._buffer
        return flushed_count

    async def close(self) -> None:
        """Cancel periodic flusher and flush remaining items."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def save_session(
        self,
        session: PipelineSessionOutput,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a validated PipelineSessionOutput to MongoDB sessions collection."""
        meta_dict = meta or {}
        now = datetime.now(timezone.utc)
        doc = {
            **session.model_dump(),
            **meta_dict,  # ip, user_agent, sso_user, clinician_name, etc.
            "created_at": now,
            "updated_at": now,
        }
        session_id = doc.get("session_id") or ""

        # High-performance write-behind buffer path
        if self._enable_buffer and session_id:
            async with self._lock:
                self._buffer.append(doc)
                should_flush = len(self._buffer) >= self._batch_size
            if should_flush:
                asyncio.create_task(self.flush())
            return session_id

        # Direct write path
        if session_id:
            res = await self._sessions.update_one(
                {"session_id": session_id},
                {"$set": doc},
                upsert=True,
            )
            if res.upserted_id:
                return str(res.upserted_id)
            existing = await self._sessions.find_one({"session_id": session_id}, {"_id": 1})
            return str(existing["_id"]) if existing and "_id" in existing else str(session_id)

        result = await self._sessions.insert_one(doc)
        return str(result.inserted_id) if result and result.inserted_id else str(session_id)

    async def save_pipeline_step(
        self,
        session_id: str,
        step: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        timings: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> str:
        """Record an individual pipeline step execution inside pipeline_runs."""
        doc = {
            "session_id": session_id,
            "step_name": step,
            "input_data": input_data,
            "output_data": output_data,
            "timings": timings,
            "usage_meta": usage,
            "error": error,
            "created_at": datetime.now(timezone.utc),
        }
        result = await self._runs.insert_one(doc)
        return str(result.inserted_id)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetch session by session_id."""
        doc = await self._sessions.find_one({"session_id": session_id})
        return _serialize_mongo_doc(doc)

    async def list_sessions(
        self,
        limit: int = 100,
        department: Optional[str] = None,
        clinician_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List sessions sorted newest first with optional filtering."""
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department
        if clinician_name:
            query["clinician_name"] = clinician_name

        cursor = self._sessions.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize_doc(d) for d in docs]

    async def list_pipeline_runs(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all pipeline steps for a given session."""
        cursor = self._runs.find({"session_id": session_id}).sort("created_at", 1)
        docs = await cursor.to_list(length=100)
        return [_serialize_doc(d) for d in docs]


class FeedbackRepository:
    """CRUD for the feedback collection."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db.feedback

    async def save(self, entry: Dict[str, Any]) -> str:
        """Insert clinician feedback into feedback collection."""
        doc = dict(entry)
        if "created_at" not in doc:
            doc["created_at"] = datetime.now(timezone.utc)
        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    async def list_recent(
        self,
        limit: int = 500,
        department: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent feedback documents, newest first."""
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department

        cursor = self._col.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize_doc(d) for d in docs]


class GoldDatasetRepository:
    """Repository for gold standard clinical test cases (Phase 4 Evaluation Store)."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db.gold_cases

    async def upsert_case(self, case: GoldCase) -> str:
        """Create or update a gold case by case_id."""
        now = datetime.now(timezone.utc)
        doc = case.model_dump()
        doc["updated_at"] = now
        if not doc.get("created_at"):
            doc["created_at"] = now

        await self._col.update_one(
            {"case_id": case.case_id},
            {"$set": doc},
            upsert=True,
        )
        return case.case_id

    # Backward-compatible alias
    save_case = upsert_case

    async def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        doc = await self._col.find_one({"case_id": case_id})
        return _serialize_mongo_doc(doc)

    async def list_cases(
        self,
        department: Optional[str] = None,
        specialty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = True,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if active_only:
            query["is_active"] = {"$ne": False}
        if department:
            query["department"] = department
        if specialty:
            query["specialty"] = specialty
        if tags:
            query["tags"] = {"$all": tags}

        cursor = self._col.find(query).sort("case_id", 1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize_doc(d) for d in docs]

    async def import_from_xlsx(self, file_path: str) -> int:
        """Bulk import from Excel workbook using importers module."""
        from evaluation.importers import load_gold_cases_from_xlsx
        cases = load_gold_cases_from_xlsx(file_path)
        count = 0
        for case in cases:
            await self.upsert_case(case)
            count += 1
        return count


class EvalRepository:
    """Repository for evaluation benchmarking results (Phase 4 Engine)."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self._col = db.eval_runs

    async def save_run(self, run: Dict[str, Any]) -> str:
        """Save an evaluation run result dictionary."""
        doc = dict(run)
        if "created_at" not in doc:
            doc["created_at"] = datetime.now(timezone.utc)
        elif isinstance(doc["created_at"], str):
            try:
                doc["created_at"] = datetime.fromisoformat(doc["created_at"])
            except ValueError:
                pass

        eval_id = doc.get("eval_id")
        if eval_id:
            await self._col.update_one(
                {"eval_id": eval_id},
                {"$set": doc},
                upsert=True,
            )
            return str(eval_id)

        result = await self._col.insert_one(doc)
        return str(result.inserted_id)

    # Backward-compatible alias
    async def save_eval(self, eval_run: EvalRunModel) -> str:
        return await self.save_run(eval_run.model_dump())

    async def get_eval(self, eval_id: str) -> Optional[Dict[str, Any]]:
        doc = await self._col.find_one({"eval_id": eval_id})
        return _serialize_mongo_doc(doc)

    async def get_runs_for_case(self, case_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        cursor = self._col.find({"case_id": case_id}).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize_doc(d) for d in docs]

    list_by_case = get_runs_for_case

    async def list_recent_runs(
        self,
        department: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department
        if model_id:
            query["model_id"] = model_id

        cursor = self._col.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize_doc(d) for d in docs]

    async def aggregate_report(self, department: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate evaluation metrics across all runs."""
        query: Dict[str, Any] = {}
        if department:
            query["department"] = department

        cursor = self._col.find(query)
        docs = await cursor.to_list(length=1000)
        if not docs:
            return {
                "total_runs": 0,
                "mean_accuracy": 0.0,
                "mean_completeness": 0.0,
                "mean_total_score": 0.0,
                "diagnosis_match_rate": 0.0,
                "department": department or "all",
            }

        n = len(docs)
        tot_acc = sum(float(d.get("extraction_accuracy", 0.0)) for d in docs)
        tot_comp = sum(float(d.get("extraction_completeness", 0.0)) for d in docs)
        tot_score = sum(float(d.get("total_score", 0.0)) for d in docs)
        tot_diag = sum(1 for d in docs if d.get("diagnosis_match") is True)

        return {
            "total_runs": n,
            "mean_accuracy": round(tot_acc / n, 4),
            "mean_completeness": round(tot_comp / n, 4),
            "mean_total_score": round(tot_score / n, 4),
            "diagnosis_match_rate": round(tot_diag / n, 4),
            "department": department or "all",
        }
