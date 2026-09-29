"""Unit and integration tests for Step 9: Evaluation Dashboard & Research Experiments."""

import pytest
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


# ============================================================================
# 1. RETRIEVAL METRICS TESTS
# ============================================================================

class TestRetrievalMetrics:
    def test_precision_at_k_normal(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}
        p5 = compute_precision_at_k(retrieved, relevant, k=5)
        assert p5 == 0.6  # 3 / 5

    def test_precision_at_k_edge_cases(self):
        assert compute_precision_at_k([], {"doc1"}, k=5) == 0.0
        assert compute_precision_at_k(["doc1"], [], k=5) == 0.0
        assert compute_precision_at_k(["doc1"], {"doc1"}, k=0) == 0.0
        assert compute_precision_at_k(["doc1"], {"doc1"}, k=-1) == 0.0

    def test_recall_at_k_normal(self):
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "docX", "docY"}
        r5 = compute_recall_at_k(retrieved, relevant, k=5)
        assert r5 == 0.5  # 2 / 4

    def test_recall_at_k_edge_cases(self):
        assert compute_recall_at_k([], {"doc1"}, k=5) == 0.0
        assert compute_recall_at_k(["doc1"], set(), k=5) == 0.0
        assert compute_recall_at_k(["doc1"], {"doc1"}, k=0) == 0.0

    def test_reciprocal_rank(self):
        retrieved = ["docA", "docB", "docC"]
        # Match at rank 1 -> RR = 1.0
        assert compute_reciprocal_rank(retrieved, {"docA"}) == 1.0
        # Match at rank 2 -> RR = 0.5
        assert compute_reciprocal_rank(retrieved, {"docB"}) == 0.5
        # Match at rank 3 -> RR = 0.3333
        assert compute_reciprocal_rank(retrieved, {"docC"}) == 0.3333
        # No match -> 0.0
        assert compute_reciprocal_rank(retrieved, {"docZ"}) == 0.0

    def test_mrr(self):
        all_retrieved = [
            ["doc1", "doc2"],  # hit at 1 -> 1.0
            ["docX", "doc1"],  # hit at 2 -> 0.5
            ["docA", "docB"],  # hit at 0 -> 0.0
        ]
        all_relevant = [{"doc1"}, {"doc1"}, {"docZ"}]
        mrr = compute_mrr(all_retrieved, all_relevant)
        assert mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3.0, abs=1e-4)

    def test_ndcg_at_k(self):
        retrieved = ["doc1", "doc2", "doc3"]
        scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        # Perfectly ranked
        assert compute_ndcg_at_k(retrieved, scores, k=3) == 1.0

        # Sub-optimally ranked
        inverted = ["doc3", "doc2", "doc1"]
        ndcg_inv = compute_ndcg_at_k(inverted, scores, k=3)
        assert 0.0 < ndcg_inv < 1.0

        # Edge cases
        assert compute_ndcg_at_k([], scores, k=3) == 0.0
        assert compute_ndcg_at_k(retrieved, {}, k=3) == 0.0
        assert compute_ndcg_at_k(retrieved, scores, k=0) == 0.0


# ============================================================================
# 2. GENERATION QUALITY & RAG TRIAD TESTS
# ============================================================================

class TestGenerationQualityMetrics:
    def test_context_relevance(self):
        query = "educational qualification at IPEC Ghaziabad"
        chunks = [
            "Nikhil attended IPEC Ghaziabad for B.Tech in Artificial Intelligence.",
            "He built a full stack web app using MongoDB and React.",
            "Weather in Ghaziabad is pleasant in November.",
        ]
        relevance = compute_context_relevance(query, chunks, relevance_threshold=0.15)
        assert relevance > 0.0
        assert compute_context_relevance("", chunks) == 0.0
        assert compute_context_relevance(query, []) == 0.0

    def test_faithfulness(self):
        claims = [
            {"claim": "Attended IPEC", "verification_status": "supported"},
            {"claim": "Graduating 2027", "verification_status": "supported"},
            {"claim": "AWS Certified Architect", "verification_status": "partially_supported"},
            {"claim": "Founded Google", "verification_status": "contradicted"},
        ]
        faith = compute_faithfulness(claims)
        # 1 + 1 + 0.5 + 0 = 2.5 / 4 = 0.625
        assert faith == 0.625

    def test_faithfulness_empty(self):
        assert compute_faithfulness([]) == 1.0

    def test_answer_correctness(self):
        gen = "Nikhil graduated from Inderprastha Engineering College with B.Tech."
        gt = "Inderprastha Engineering College (IPEC) B.Tech degree."
        f1 = compute_answer_correctness(gen, gt)
        assert f1 > 0.3

        assert compute_answer_correctness("", gt) == 0.0
        assert compute_answer_correctness(gen, "") == 0.0

    def test_unsupported_answer_rate(self):
        statuses = ["supported", "supported", "contradicted", "insufficient_evidence"]
        rate = compute_unsupported_answer_rate(statuses)
        # 2 / 4 = 0.5
        assert rate == 0.5

        assert compute_unsupported_answer_rate([]) == 0.0


# ============================================================================
# 3. CITATIONS & VERIFICATION DISTRIBUTION TESTS
# ============================================================================

class TestCitationsAndVerificationDistribution:
    def test_citation_metrics(self):
        citations = [
            {
                "claim": "Attended IPEC",
                "document_name": "resume.pdf",
                "chunk_id": "c1",
                "evidence_text": "Inderprastha Engineering College",
                "verification_status": "supported",
            },
            {
                "claim": "Graduating 2027",
                "document_name": "resume.pdf",
                "chunk_id": "c2",
                "evidence_text": "Expected 2027 graduation",
                "verification_status": "supported",
            },
        ]
        m = compute_citation_metrics(citations, claims=[{}, {}], ground_truth_sources=["resume.pdf"])
        assert m["citation_correctness"] == 1.0
        assert m["citation_completeness"] == 1.0
        assert m["attribution_accuracy"] == 1.0
        assert m["evidence_matching"] == 1.0
        assert m["phantom_citation_rate"] == 0.0

    def test_phantom_citation_detected(self):
        citations = [
            {
                "claim": "Fake claim",
                "document_name": "resume.pdf",
                "chunk_id": "c1",
                "evidence_text": "Sample text",
                "verification_status": "contradicted",
            }
        ]
        m = compute_citation_metrics(citations)
        assert m["phantom_citation_rate"] == 1.0
        assert m["citation_correctness"] == 0.0

    def test_verification_distribution(self):
        statuses = ["supported", "supported", "partially_supported", "contradicted", "insufficient_evidence"]
        dist = compute_verification_distribution(statuses)
        assert dist["total"] == 5
        assert dist["supported"] == 2
        assert dist["partially_supported"] == 1
        assert dist["contradicted"] == 1
        assert dist["insufficient_evidence"] == 1
        assert dist["supported_pct"] == 40.0
        assert dist["unsupported_answer_rate_pct"] == 50.0  # (1 + 1 + 0.5) / 5 = 2.5 / 5 = 50%


# ============================================================================
# 4. EXPERIMENT RUNNER & DASHBOARD SERVICE TESTS
# ============================================================================

class TestExperimentRunnerAndService:
    def test_experiment_runner_methods(self):
        runner = ExperimentRunner()
        ret = runner.run_retrieval_comparison()
        assert "vector_only" in ret
        assert "hybrid_rrf" in ret
        assert "hybrid_rerank" in ret
        assert ret["vector_only"]["precision_at_k"] >= 0.0

        top_k = runner.run_top_k_sensitivity()
        assert len(top_k) >= 5
        assert top_k[0]["k"] == 1
        # Top-1 precision >= Top-10 precision
        assert top_k[0]["precision"] >= top_k[-1]["precision"]

        chunks = runner.run_chunk_size_sensitivity()
        assert len(chunks) == 3
        assert any(c["chunk_size"] == 500 for c in chunks)

        verif = runner.run_verification_evaluation()
        assert verif["accuracy_pct"] >= 80.0
        assert verif["supported_pct"] > 0.0

        cite = runner.run_citation_evaluation()
        assert cite["citation_correctness_pct"] >= 90.0

    def test_dashboard_service_payload(self):
        service = DashboardService()
        payload = service.get_dashboard_payload()

        assert payload["status"] == "success"
        assert "kpis" in payload
        assert "pipeline_comparison" in payload
        assert "top_k_sensitivity" in payload
        assert "chunk_size_sensitivity" in payload
        assert "latency_breakdown" in payload

        # Check KPI values
        kpis = payload["kpis"]
        assert kpis["context_relevance"] > 0
        assert kpis["faithfulness"] > 0
        assert kpis["mrr"] > 0
        assert kpis["ndcg_at_5"] > 0

        # Check pipeline comparison rows
        stages = [s["stage_key"] for s in payload["pipeline_comparison"]]
        assert "baseline_vector" in stages
        assert "hybrid_retrieval" in stages
        assert "neural_reranking" in stages
        assert "evidence_verification" in stages

    def test_dashboard_service_live_experiment(self):
        service = DashboardService()
        res = service.run_live_experiment("retrieval")
        assert res["executed_experiment"] == "retrieval"
        assert res["status"] == "success"


# ============================================================================
# 5. WEB SERVER ENDPOINTS TESTS
# ============================================================================

class TestWebEndpoints:
    @pytest.fixture
    def client(self):
        from web_server import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_evaluation_dashboard_endpoint(self, client):
        response = client.get("/api/evaluation/dashboard")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "kpis" in data
        assert "pipeline_comparison" in data
        assert "top_k_sensitivity" in data

    def test_evaluation_run_experiment_endpoint(self, client):
        response = client.post(
            "/api/evaluation/run_experiment",
            json={"experiment": "top_k"},
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["executed_experiment"] == "top_k"
