"""Unit and integration tests for Research Intelligence and Research Gap Detection (Step 8)."""

from unittest.mock import MagicMock
import pytest

from app.intelligence import (
    RESEARCH_GAP_CATEGORIES,
    RESEARCH_INTELLIGENCE_DIMENSIONS,
    ResearchIntelligenceEngine,
)
from app.citations import CitationEngine


@pytest.fixture
def mock_multi_doc_evidence():
    """Mock partitioned evidence for 3 distinct research papers."""
    return {
        "Attention_Is_All_You_Need.pdf": [
            {
                "id": "chunk_transformer_01",
                "chunk_id": "chunk_transformer_01",
                "text": (
                    "The dominant sequence transduction models are based on complex recurrent or "
                    "convolutional neural networks. We propose the Transformer, a model architecture "
                    "relying entirely on an attention mechanism. "
                    "We train on the standard WMT 2014 English-to-German dataset."
                ),
                "metadata": {
                    "document_id": "doc_trans123",
                    "source_file": "Attention_Is_All_You_Need.pdf",
                    "page_number": 1,
                    "section": "Abstract & Introduction",
                    "chunk_id": "chunk_transformer_01",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.952,
            },
            {
                "id": "chunk_transformer_02",
                "chunk_id": "chunk_transformer_02",
                "text": (
                    "On WMT 2014 English-to-German, the model achieves 28.4 BLEU. "
                    "A notable limitation is the quadratic computational complexity with respect to sequence length, "
                    "which limits maximum input window size."
                ),
                "metadata": {
                    "document_id": "doc_trans123",
                    "source_file": "Attention_Is_All_You_Need.pdf",
                    "page_number": 6,
                    "section": "Results & Discussion",
                    "chunk_id": "chunk_transformer_02",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.924,
            },
        ],
        "BERT_Pretraining.pdf": [
            {
                "id": "chunk_bert_01",
                "chunk_id": "chunk_bert_01",
                "text": (
                    "We introduce BERT: Bidirectional Encoder Representations from Transformers. "
                    "Trained on BooksCorpus and English Wikipedia with masked language modeling. "
                    "BERT claims superior fine-tuning performance across NLP benchmarks."
                ),
                "metadata": {
                    "document_id": "doc_bert456",
                    "source_file": "BERT_Pretraining.pdf",
                    "page_number": 1,
                    "section": "Introduction",
                    "chunk_id": "chunk_bert_01",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.941,
            },
            {
                "id": "chunk_bert_02",
                "chunk_id": "chunk_bert_02",
                "text": (
                    "BERT pushes the GLUE score to 80.5%. "
                    "A major bottleneck and limitation is high memory overhead during pre-training and "
                    "inference latency on consumer grade hardware."
                ),
                "metadata": {
                    "document_id": "doc_bert456",
                    "source_file": "BERT_Pretraining.pdf",
                    "page_number": 5,
                    "section": "Evaluation",
                    "chunk_id": "chunk_bert_02",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.898,
            },
        ],
        "ResNet_Deep_Residual.pdf": [
            {
                "id": "chunk_resnet_01",
                "chunk_id": "chunk_resnet_01",
                "text": (
                    "Deeper neural networks are more difficult to train. We present a residual learning "
                    "framework to ease the training of networks that are substantially deeper. "
                    "We evaluate on ImageNet 2012 classification."
                ),
                "metadata": {
                    "document_id": "doc_resnet789",
                    "source_file": "ResNet_Deep_Residual.pdf",
                    "page_number": 1,
                    "section": "Abstract",
                    "chunk_id": "chunk_resnet_01",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.933,
            },
            {
                "id": "chunk_resnet_02",
                "chunk_id": "chunk_resnet_02",
                "text": (
                    "Our ensemble won 1st place on ILSVRC 2015 with 3.57% top-5 error. "
                    "In contrast to language models, vision models exhibit different computational bounds and "
                    "unresolved degradation on extremely shallow layers."
                ),
                "metadata": {
                    "document_id": "doc_resnet789",
                    "source_file": "ResNet_Deep_Residual.pdf",
                    "page_number": 4,
                    "section": "Experiments",
                    "chunk_id": "chunk_resnet_02",
                },
                "retrieval_source": "hybrid",
                "reranker_score": 0.915,
            },
        ],
    }


class TestResearchIntelligenceEngine:
    """Unit tests for ResearchIntelligenceEngine."""

    def test_requires_at_least_two_documents(self):
        engine = ResearchIntelligenceEngine()
        with pytest.raises(ValueError, match="requires at least 2 documents"):
            engine.analyze_research(selected_documents=["single_paper.pdf"])

    def test_intelligence_two_papers(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        {
          "literature_review": {
            "summary": "This review analyzes the Transformer and BERT architectures in modern deep learning.",
            "key_findings": {
              "Attention_Is_All_You_Need.pdf": "Transformers eliminate recurrent networks entirely [chunk_transformer_01].",
              "BERT_Pretraining.pdf": "BERT establishes bidirectional pre-training [chunk_bert_01]."
            },
            "methods": {
              "Attention_Is_All_You_Need.pdf": "Self-attention mechanism",
              "BERT_Pretraining.pdf": "Masked language modeling"
            },
            "datasets": {
              "Attention_Is_All_You_Need.pdf": "WMT 2014 English-German",
              "BERT_Pretraining.pdf": "BooksCorpus and English Wikipedia"
            },
            "results": {
              "Attention_Is_All_You_Need.pdf": "28.4 BLEU score",
              "BERT_Pretraining.pdf": "80.5% GLUE benchmark"
            },
            "common_themes": [
              "Reliance on attention-based sequence representation."
            ],
            "differences": [
              "Transformer focuses on translation while BERT focuses on general representation."
            ],
            "limitations": {
              "Attention_Is_All_You_Need.pdf": "Quadratic computational complexity with sequence length",
              "BERT_Pretraining.pdf": "High memory overhead during pretraining"
            }
          },
          "research_gaps": [
            {
              "gap_id": "GAP-1",
              "category": "limitation",
              "description": "Quadratic attention scaling bounds maximum context window.",
              "evidence": "quadratic computational complexity with respect to sequence length",
              "evidence_chunk_ids": ["chunk_transformer_02"],
              "source_papers": ["Attention_Is_All_You_Need.pdf"],
              "validation_note": "Requires researcher validation."
            }
          ],
          "research_questions": [
            {
              "question_id": "RQ-1",
              "gap_id": "GAP-1",
              "question": "How can sub-quadratic attention approximations preserve translation BLEU scores?",
              "rationale": "Directly tackles the sequence length bottleneck identified in Transformer.",
              "target_papers": ["Attention_Is_All_You_Need.pdf"]
            }
          ]
        }
        """
        cit_engine = CitationEngine(snippet_max_chars=200)
        engine = ResearchIntelligenceEngine(
            llm_client=mock_llm,
            citation_engine=cit_engine,
        )
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        assert "literature_review" in res
        lr = res["literature_review"]
        assert "Transformer" in lr["summary"]
        assert len(lr["common_themes"]) >= 1
        assert len(lr["differences"]) >= 1
        assert len(res["research_gaps"]) >= 1
        assert len(res["research_questions"]) >= 1
        assert res["total_citations"] >= 1

    def test_intelligence_three_plus_papers(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "invalid json for fallback test"
        cit_engine = CitationEngine()
        engine = ResearchIntelligenceEngine(
            llm_client=mock_llm,
            citation_engine=cit_engine,
        )
        engine.retrieve_evidence_for_documents = MagicMock(return_value=mock_multi_doc_evidence)

        res = engine.analyze_research(
            selected_documents=[
                "Attention_Is_All_You_Need.pdf",
                "BERT_Pretraining.pdf",
                "ResNet_Deep_Residual.pdf",
            ]
        )

        assert len(res["documents"]) == 3
        assert "literature_review" in res
        assert len(res["research_gaps"]) >= 1
        assert len(res["research_questions"]) >= 1
        assert res["total_citations"] >= 1

    def test_research_gap_detection(self, mock_multi_doc_evidence):
        engine = ResearchIntelligenceEngine(citation_engine=CitationEngine())
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        gaps = res["research_gaps"]
        assert len(gaps) >= 1
        for gap in gaps:
            assert "gap_id" in gap
            assert gap["category"] in RESEARCH_GAP_CATEGORIES
            assert "description" in gap
            assert len(gap["description"]) > 0
            assert gap["validation_note"] == "Requires researcher validation."
            assert isinstance(gap["source_papers"], list)

    def test_research_question_generation(self, mock_multi_doc_evidence):
        engine = ResearchIntelligenceEngine(citation_engine=CitationEngine())
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        rqs = res["research_questions"]
        gaps = {g["gap_id"] for g in res["research_gaps"]}
        assert len(rqs) >= 1
        for rq in rqs:
            assert "question_id" in rq
            assert rq["gap_id"] in gaps
            assert "?" in rq["question"]
            assert len(rq["rationale"]) > 0

    def test_citation_preservation(self, mock_multi_doc_evidence):
        engine = ResearchIntelligenceEngine(citation_engine=CitationEngine())
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        citations = res["citations"]
        assert len(citations) >= 1
        for cit in citations:
            assert "document_id" in cit
            assert ("document_name" in cit or "source_file" in cit)
            assert "page_number" in cit
            assert "section" in cit
            assert "chunk_id" in cit
            assert "reranker_score" in cit
            assert cit["reranker_score"] is not None

    def test_missing_evidence_handling(self):
        """Test that missing evidence outputs 'Insufficient evidence.' with zero fake citations."""
        sparse_evidence = {
            "Paper_A.pdf": [
                {
                    "id": "chunk_a1",
                    "chunk_id": "chunk_a1",
                    "text": "Paper A introduces an optimization algorithm for convex losses.",
                    "metadata": {
                        "document_id": "doc_a",
                        "source_file": "Paper_A.pdf",
                        "page_number": 1,
                        "section": "Abstract",
                        "chunk_id": "chunk_a1",
                    },
                    "retrieval_source": "hybrid",
                    "reranker_score": 0.85,
                }
            ],
            "Paper_B.pdf": [
                {
                    "id": "chunk_b1",
                    "chunk_id": "chunk_b1",
                    "text": "Paper B introduces another gradient method for online learning.",
                    "metadata": {
                        "document_id": "doc_b",
                        "source_file": "Paper_B.pdf",
                        "page_number": 1,
                        "section": "Abstract",
                        "chunk_id": "chunk_b1",
                    },
                    "retrieval_source": "hybrid",
                    "reranker_score": 0.83,
                }
            ],
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        {
          "literature_review": {
            "summary": "Review of Paper A and Paper B.",
            "key_findings": {
              "Paper_A.pdf": "Convex optimization [chunk_a1]",
              "Paper_B.pdf": "Gradient method [chunk_b1]"
            },
            "methods": {
              "Paper_A.pdf": "Optimization",
              "Paper_B.pdf": "Gradient method"
            },
            "datasets": {
              "Paper_A.pdf": "Insufficient evidence.",
              "Paper_B.pdf": "Insufficient evidence."
            },
            "results": {
              "Paper_A.pdf": "Insufficient evidence.",
              "Paper_B.pdf": "Insufficient evidence."
            },
            "common_themes": ["Optimization"],
            "differences": ["Online vs batch"],
            "limitations": {
              "Paper_A.pdf": "Insufficient evidence.",
              "Paper_B.pdf": "Insufficient evidence."
            }
          },
          "research_gaps": [],
          "research_questions": []
        }
        """

        engine = ResearchIntelligenceEngine(
            llm_client=mock_llm,
            citation_engine=CitationEngine(),
        )
        engine.retrieve_evidence_for_documents = MagicMock(return_value=sparse_evidence)

        res = engine.analyze_research(selected_documents=["Paper_A.pdf", "Paper_B.pdf"])
        lr = res["literature_review"]

        assert lr["limitations"]["Paper_A.pdf"] == "Insufficient evidence."
        assert lr["limitations"]["Paper_B.pdf"] == "Insufficient evidence."
        assert len(res["missing_evidence"]) >= 1

        # Check that insufficient evidence claims have zero citations
        for cit in res["citations"]:
            assert "insufficient evidence" not in cit.get("claim", "").lower()

    def test_conflicting_findings_detection(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
        {
          "literature_review": {
            "summary": "Synthesis showing divergent benchmark findings.",
            "key_findings": {
              "Attention_Is_All_You_Need.pdf": "Transformer claims superior translation BLEU [chunk_transformer_02].",
              "BERT_Pretraining.pdf": "BERT claims superior GLUE generalization [chunk_bert_02]."
            },
            "methods": {"Attention_Is_All_You_Need.pdf": "Transformer", "BERT_Pretraining.pdf": "BERT"},
            "datasets": {"Attention_Is_All_You_Need.pdf": "WMT", "BERT_Pretraining.pdf": "GLUE"},
            "results": {"Attention_Is_All_You_Need.pdf": "28.4 BLEU", "BERT_Pretraining.pdf": "80.5% GLUE"},
            "common_themes": ["Attention"],
            "differences": ["Translation vs Classification"],
            "limitations": {"Attention_Is_All_You_Need.pdf": "Quadratic attention", "BERT_Pretraining.pdf": "Memory"}
          },
          "research_gaps": [
            {
              "gap_id": "GAP-1",
              "category": "conflicting_findings",
              "description": "Conflicting findings on parameter efficiency vs sequence length scaling across the models.",
              "evidence": "In contrast, models report differing memory and latency trade-offs.",
              "evidence_chunk_ids": ["chunk_transformer_02", "chunk_bert_02"],
              "source_papers": ["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"],
              "validation_note": "Requires researcher validation."
            }
          ],
          "research_questions": [
            {
              "question_id": "RQ-1",
              "gap_id": "GAP-1",
              "question": "Which architecture provides greater Pareto-optimal trade-offs under fixed memory?",
              "rationale": "Resolves conflicting empirical claims.",
              "target_papers": ["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
            }
          ]
        }
        """

        engine = ResearchIntelligenceEngine(
            llm_client=mock_llm,
            citation_engine=CitationEngine(),
        )
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        gaps = res["research_gaps"]
        conflict_gaps = [g for g in gaps if g["category"] == "conflicting_findings"]
        assert len(conflict_gaps) >= 1
        assert conflict_gaps[0]["validation_note"] == "Requires researcher validation."

    def test_document_attribution_accuracy(self, mock_multi_doc_evidence):
        engine = ResearchIntelligenceEngine(citation_engine=CitationEngine())
        engine.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        res = engine.analyze_research(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        by_doc = res["citations_by_document"]
        assert "Attention_Is_All_You_Need.pdf" in by_doc
        assert "BERT_Pretraining.pdf" in by_doc

        # Verify no cross-paper attribution
        for cit in by_doc["Attention_Is_All_You_Need.pdf"]:
            assert "bert" not in cit.get("document_id", "").lower()
            assert "bert" not in cit.get("source_file", "").lower()

        for cit in by_doc["BERT_Pretraining.pdf"]:
            assert "trans" not in cit.get("document_id", "").lower()
            assert "attention" not in cit.get("source_file", "").lower()


class TestRAGPipelineIntelligenceIntegration:
    """Integration test with RAGPipeline and regression check."""

    def test_pipeline_analyze_research_and_single_doc_qa(self, mock_multi_doc_evidence):
        mock_pipeline = MagicMock()
        mock_pipeline.retrieval_mode = "hybrid"
        mock_pipeline.retrieve_with_metadata.return_value = (
            mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"]
            + mock_multi_doc_evidence["BERT_Pretraining.pdf"]
        )

        from app.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.retrieval_mode = "hybrid"
        pipeline.retrieve_with_metadata = mock_pipeline.retrieve_with_metadata
        pipeline.hybrid_retriever = None
        pipeline.reranker = None
        pipeline.llm_client = None
        pipeline.verifier = None
        pipeline.citation_engine = CitationEngine()
        pipeline.comparator = None
        pipeline.config = None
        pipeline.research_intelligence = None

        result = pipeline.analyze_research(
            documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        )

        assert "literature_review" in result
        assert "research_gaps" in result
        assert "research_questions" in result
        assert "citations" in result

        # Ensure single-document Q&A remains functional
        mock_generate = MagicMock(return_value="Answer to single doc question.")
        pipeline.generate_response = mock_generate
        pipeline.reranker = None

        # Verify that calling rag_query with verify=False, cite=False doesn't break
        qa_res = pipeline.rag_query("What is self-attention?", n_retrieve=2, verify=False, cite=False)
        assert "response" in qa_res
        assert qa_res["response"] == "Answer to single doc question."
