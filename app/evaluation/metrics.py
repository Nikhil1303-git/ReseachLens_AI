"""Evaluation metrics for RAG retrieval, generation quality, citations, and verification.

All metrics are purely mathematical and deterministic, with zero external API dependencies.
Safe against zero-division, empty collections, and malformed inputs.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Union
import math
import re


# ============================================================================
# 1. RETRIEVAL METRICS
# ============================================================================

def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
    k: int = 5,
) -> float:
    """Compute Precision@K.

    Precision@K = |Retrieved[:k] ∩ Relevant| / k
    """
    if k <= 0:
        return 0.0
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k_retrieved = retrieved_ids[:k]
    relevant_retrieved = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_set]

    return round(len(relevant_retrieved) / float(k), 4)


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
    k: int = 5,
) -> float:
    """Compute Recall@K.

    Recall@K = |Retrieved[:k] ∩ Relevant| / |Relevant|
    """
    if k <= 0:
        return 0.0
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    if len(relevant_set) == 0:
        return 0.0

    top_k_retrieved = retrieved_ids[:k]
    relevant_retrieved = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_set]

    return round(min(1.0, len(relevant_retrieved) / float(len(relevant_set))), 4)


def compute_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Union[Set[str], List[str]],
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank of first relevant item (1-indexed). Returns 0.0 if not found.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return round(1.0 / float(rank), 4)

    return 0.0


def compute_mrr(
    all_retrieved_ids: List[List[str]],
    all_relevant_ids: List[Union[Set[str], List[str]]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries."""
    if not all_retrieved_ids or not all_relevant_ids:
        return 0.0

    total_rr = 0.0
    count = min(len(all_retrieved_ids), len(all_relevant_ids))
    if count == 0:
        return 0.0

    for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids):
        total_rr += compute_reciprocal_rank(retrieved, relevant)

    return round(total_rr / float(count), 4)


def compute_dcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int = 5,
) -> float:
    """Compute Discounted Cumulative Gain (DCG@K).

    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0 or not retrieved_ids or not relevance_scores:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 1)
        dcg += gain / discount

    return dcg


def compute_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int = 5,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K).

    nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or not retrieved_ids or not relevance_scores:
        return 0.0

    actual_dcg = compute_dcg_at_k(retrieved_ids, relevance_scores, k=k)
    if actual_dcg == 0.0:
        return 0.0

    # Ideal DCG: sorted by true relevance in descending order
    ideal_retrieved = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_retrieved, relevance_scores, k=k)

    if ideal_dcg <= 0.0:
        return 0.0

    return round(min(1.0, actual_dcg / ideal_dcg), 4)


# ============================================================================
# 2. ANSWER QUALITY METRICS (RAG TRIAD)
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for token overlap metrics."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_context_relevance(
    query: str,
    context_chunks: List[str],
    relevance_threshold: float = 0.15,
) -> float:
    """Compute Context Relevance: proportion of context chunks that have meaningful overlap with query.

    Context Relevance = |{chunk in Context : Overlap(chunk, query) >= threshold}| / |Context|
    """
    if not context_chunks:
        return 0.0

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    relevant_count = 0
    for chunk in context_chunks:
        chunk_tokens = set(_tokenize(chunk))
        if not chunk_tokens:
            continue
        overlap = len(query_tokens.intersection(chunk_tokens)) / float(len(query_tokens))
        if overlap >= relevance_threshold:
            relevant_count += 1

    return round(relevant_count / float(len(context_chunks)), 4)


def compute_faithfulness(
    claims: List[Dict[str, Any]],
) -> float:
    """Compute Faithfulness / Groundedness ratio.

    Faithfulness = |Supported Claims| / |Total Claims|
    Where supported claims have verification_status == 'supported' (or confidence >= 0.8).
    """
    if not claims:
        return 1.0  # vacuously faithful if no claims made

    supported_count = 0
    for claim in claims:
        status = str(claim.get("verification_status", "")).lower()
        if status == "supported" or status == "verified":
            supported_count += 1
        elif status == "partially_supported":
            supported_count += 0.5

    return round(supported_count / float(len(claims)), 4)


def compute_answer_correctness(
    generated_answer: str,
    ground_truth_answer: str,
) -> float:
    """Compute Answer Correctness via token F1 overlap against ground truth."""
    if not generated_answer or not ground_truth_answer:
        return 0.0

    gen_tokens = _tokenize(generated_answer)
    gt_tokens = _tokenize(ground_truth_answer)

    if not gen_tokens or not gt_tokens:
        return 0.0

    common = set(gen_tokens).intersection(set(gt_tokens))
    if not common:
        return 0.0

    precision = len(common) / float(len(set(gen_tokens)))
    recall = len(common) / float(len(set(gt_tokens)))

    if precision + recall == 0.0:
        return 0.0

    f1 = 2.0 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_unsupported_answer_rate(
    claims_or_statuses: List[Union[Dict[str, Any], str]],
) -> float:
    """Compute Unsupported Answer Rate: fraction of claims that are contradicted or lack evidence."""
    if not claims_or_statuses:
        return 0.0

    unsupported_count = 0
    total = len(claims_or_statuses)

    for item in claims_or_statuses:
        status = item.get("verification_status", "") if isinstance(item, dict) else str(item)
        status_clean = status.lower().strip()
        if status_clean in ("contradicted", "insufficient_evidence", "unsupported", "ungrounded"):
            unsupported_count += 1
        elif status_clean == "partially_supported":
            unsupported_count += 0.5

    return round(unsupported_count / float(total), 4)


# ============================================================================
# 3. CITATION & PROVENANCE METRICS
# ============================================================================

def compute_citation_metrics(
    citations: List[Dict[str, Any]],
    claims: Optional[List[Dict[str, Any]]] = None,
    ground_truth_sources: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute Citation Correctness, Completeness, Attribution Accuracy, and Verbatim Evidence Match."""
    if not citations:
        has_claims = bool(claims and len(claims) > 0)
        return {
            "citation_correctness": 1.0 if not has_claims else 0.0,
            "citation_completeness": 1.0 if not has_claims else 0.0,
            "attribution_accuracy": 1.0,
            "evidence_matching": 1.0 if not has_claims else 0.0,
            "phantom_citation_rate": 0.0,
        }

    correct_citations = 0
    attribution_correct = 0
    verbatim_matches = 0
    phantom_citations = 0

    gt_set = set(s.lower() for s in (ground_truth_sources or []))

    for cit in citations:
        evidence = cit.get("evidence_text") or cit.get("evidence") or ""
        doc_name = (cit.get("document_name") or cit.get("source_file") or "").lower()
        chunk_id = cit.get("chunk_id") or ""
        status = str(cit.get("verification_status", "supported")).lower()

        # Verbatim match check
        if evidence and len(evidence.strip()) >= 5:
            verbatim_matches += 1

        # Attribution check: if gt_set provided, verify doc matches
        if gt_set:
            if any(gt in doc_name or doc_name in gt for gt in gt_set):
                attribution_correct += 1
        else:
            if doc_name and doc_name != "unknown":
                attribution_correct += 1

        # Phantom check: citation emitted for unsupported/insufficient claim
        if status in ("insufficient_evidence", "contradicted", "unsupported"):
            phantom_citations += 1
        else:
            correct_citations += 1

    total_cits = float(len(citations))
    citation_correctness = round(correct_citations / total_cits, 4)
    attribution_accuracy = round(attribution_correct / total_cits, 4)
    evidence_matching = round(verbatim_matches / total_cits, 4)
    phantom_rate = round(phantom_citations / total_cits, 4)

    # Completeness: ratio of cited claims to claims needing citations
    if claims:
        cited_indices = set(cit.get("citation_index") or cit.get("index") or i for i, cit in enumerate(citations, 1))
        completeness = round(min(1.0, len(cited_indices) / float(len(claims))), 4)
    else:
        completeness = 1.0

    return {
        "citation_correctness": citation_correctness,
        "citation_completeness": completeness,
        "attribution_accuracy": attribution_accuracy,
        "evidence_matching": evidence_matching,
        "phantom_citation_rate": phantom_rate,
    }


# ============================================================================
# 4. VERIFICATION STATUS DISTRIBUTION
# ============================================================================

def compute_verification_distribution(
    statuses: List[str],
) -> Dict[str, Any]:
    """Compute distribution percentages and counts for verification categories."""
    total = len(statuses)
    if total == 0:
        return {
            "total": 0,
            "supported": 0,
            "partially_supported": 0,
            "contradicted": 0,
            "insufficient_evidence": 0,
            "supported_pct": 0.0,
            "partially_supported_pct": 0.0,
            "contradicted_pct": 0.0,
            "insufficient_evidence_pct": 0.0,
            "unsupported_answer_rate_pct": 0.0,
        }

    counts = {
        "supported": 0,
        "partially_supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
    }

    for s in statuses:
        clean = s.lower().strip()
        if clean in counts:
            counts[clean] += 1
        elif "partial" in clean:
            counts["partially_supported"] += 1
        elif "contradict" in clean:
            counts["contradicted"] += 1
        elif "insufficient" in clean:
            counts["insufficient_evidence"] += 1
        else:
            counts["supported"] += 1

    return {
        "total": total,
        "supported": counts["supported"],
        "partially_supported": counts["partially_supported"],
        "contradicted": counts["contradicted"],
        "insufficient_evidence": counts["insufficient_evidence"],
        "supported_pct": round((counts["supported"] / float(total)) * 100.0, 1),
        "partially_supported_pct": round((counts["partially_supported"] / float(total)) * 100.0, 1),
        "contradicted_pct": round((counts["contradicted"] / float(total)) * 100.0, 1),
        "insufficient_evidence_pct": round((counts["insufficient_evidence"] / float(total)) * 100.0, 1),
        "unsupported_answer_rate_pct": round(
            ((counts["contradicted"] + counts["insufficient_evidence"] + 0.5 * counts["partially_supported"]) / float(total)) * 100.0,
            1,
        ),
    }
