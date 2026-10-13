"""
Async MongoDB connection pool using Motor.
One shared client per worker, created in the FastAPI lifespan.
"""
from __future__ import annotations
import os
import time
import logging
from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_db(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
    client_instance: Optional[AsyncIOMotorClient] = None,
) -> AsyncIOMotorDatabase:
    """Initialize the Motor client and return the database handle.
    
    Supports dependency injection of custom Motor clients for testing (e.g. mongomock-motor).
    """
    global _client, _db

    target_uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    target_db_name = db_name or os.getenv("MONGODB_DATABASE", "medvoice")
    max_pool = int(os.getenv("MONGODB_MAX_POOL", "50"))
    min_pool = int(os.getenv("MONGODB_MIN_POOL", "5"))

    if client_instance is not None:
        _client = client_instance
    else:
        _client = AsyncIOMotorClient(
            target_uri,
            maxPoolSize=max_pool,
            minPoolSize=min_pool,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True,
            retryReads=True,
        )

    _db = _client[target_db_name]

    # Ensure indexes idempotently on first connect
    try:
        await _ensure_indexes(_db)
    except Exception as exc:
        logger.warning("[MongoDB] Warning during index creation: %s", exc)

    logger.info("[MongoDB] Connected to %s / %s (pool=%d)", target_uri, target_db_name, max_pool)
    return _db


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    """Create indexes idempotently across all collections, including TTL and compound queries."""
    # Pipeline runs TTL index (default: 90 days = 7776000 seconds)
    ttl_seconds = int(os.getenv("MONGODB_PIPELINE_RUNS_TTL_SECONDS", str(90 * 24 * 3600)))

    # sessions collection indexes
    await db.sessions.create_index("session_id", unique=True, sparse=True)
    await db.sessions.create_index("created_at")
    await db.sessions.create_index("department")
    await db.sessions.create_index("clinician_name")
    await db.sessions.create_index([("department", 1), ("created_at", -1)])

    # pipeline_runs collection indexes + TTL
    await db.pipeline_runs.create_index("session_id")
    await db.pipeline_runs.create_index([("session_id", 1), ("step_name", 1)])
    await db.pipeline_runs.create_index("created_at")
    if ttl_seconds > 0:
        await db.pipeline_runs.create_index("created_at", expireAfterSeconds=ttl_seconds)

    # feedback collection indexes
    await db.feedback.create_index("session_id")
    await db.feedback.create_index("created_at")
    await db.feedback.create_index([("department", 1), ("created_at", -1)])

    # gold_cases collection (evaluation benchmark dataset)
    await db.gold_cases.create_index("case_id", unique=True)
    await db.gold_cases.create_index("department")
    await db.gold_cases.create_index([("department", 1), ("specialty", 1)])

    # eval_runs collection (eval pipeline runs)
    await db.eval_runs.create_index("eval_id", unique=True)
    await db.eval_runs.create_index("case_id")
    await db.eval_runs.create_index([("case_id", 1), ("model_id", 1)])
    await db.eval_runs.create_index("session_id")


async def get_mongo_status() -> Dict[str, Any]:
    """Retrieve connection pool and latency diagnostics for /admin/status and /health."""
    global _client, _db
    if _client is None or _db is None:
        return {"status": "disconnected", "connected": False}

    start = time.perf_counter()
    try:
        # Ping server using admin database command with timeout
        await _db.command("ping")
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": "connected",
            "connected": True,
            "database": _db.name,
            "latency_ms": latency_ms,
            "max_pool_size": int(os.getenv("MONGODB_MAX_POOL", "50")),
            "min_pool_size": int(os.getenv("MONGODB_MIN_POOL", "5")),
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": "error",
            "connected": False,
            "error": str(exc),
            "latency_ms": latency_ms,
        }


async def close_db() -> None:
    """Close active Motor client and reset database handles."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("[MongoDB] Connection closed")


def get_db() -> AsyncIOMotorDatabase:
    """Retrieve initialized database handle or raise RuntimeError."""
    if _db is None:
        raise RuntimeError("MongoDB not initialized — call connect_db() first")
    return _db
