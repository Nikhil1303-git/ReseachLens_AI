"""Evaluation package for ResearchLens AI."""

from app.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_mrr,
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_context_relevance,
    compute_faithfulness,
    compute_answer_correctness,
    compute_unsupported_answer_rate,
    compute_citation_metrics,
    compute_verification_distribution,
)
from app.evaluation.experiment_runner import ExperimentRunner
from app.evaluation.dashboard_service import DashboardService

__all__ = [
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_reciprocal_rank",
    "compute_mrr",
    "compute_dcg_at_k",
    "compute_ndcg_at_k",
    "compute_context_relevance",
    "compute_faithfulness",
    "compute_answer_correctness",
    "compute_unsupported_answer_rate",
    "compute_citation_metrics",
    "compute_verification_distribution",
    "ExperimentRunner",
    "DashboardService",
]
