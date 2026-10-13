"""
Database layer package exposing connection management and typed repositories.
"""
from .connection import connect_db, close_db, get_db
from .repositories import (
    SessionRepository,
    FeedbackRepository,
    GoldDatasetRepository,
    EvalRepository,
)
from .models import (
    SessionDoc,
    PipelineRunDoc,
    FeedbackDoc,
    GoldCaseDoc,
    EvalRunDoc,
    GoldCaseModel,
    EvalRunModel,
)

__all__ = [
    "connect_db",
    "close_db",
    "get_db",
    "SessionRepository",
    "FeedbackRepository",
    "GoldDatasetRepository",
    "EvalRepository",
    "SessionDoc",
    "PipelineRunDoc",
    "FeedbackDoc",
    "GoldCaseDoc",
    "EvalRunDoc",
    "GoldCaseModel",
    "EvalRunModel",
]
