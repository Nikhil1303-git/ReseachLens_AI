"""Unit and regression tests for DocumentComparator and Multi-Paper Intelligence (Step 7)."""

from unittest.mock import MagicMock
import pytest

from app.comparison import COMPARISON_ASPECTS, DocumentComparator
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
                    "eschewing recurrence and relying entirely on an attention mechanism. "
                    "We train on the standard WMT 2014 English-to-German dataset consisting of about 4.5M sentence pairs."
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
                    "On the WMT 2014 English-to-German translation task, the big transformer model achieves "
                    "a BLEU score of 28.4, outperforming existing models by over 2.0 BLEU. "
                    "A limitation is the quadratic computational complexity with respect to sequence length."
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
                    "BERT is designed to pre-train deep bidirectional representations from unlabeled text "
                    "by jointly conditioning on both left and right context in all layers. "
                    "Trained on BooksCorpus (800M words) and English Wikipedia (2,500M words)."
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
                    "BERT advances the state-of-the-art for eleven NLP tasks, pushing the GLUE score to 80.5%. "
                    "Limitations include high memory consumption during fine-tuning on consumer hardware."
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
                    "We evaluate on the ImageNet 2012 classification dataset with 1.28 million training images."
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
                    "Our ensemble won 1st place on the ILSVRC 2015 classification task with 3.57% top-5 error rate. "
                    "Residual networks of up to 152 layers were successfully trained."
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


class TestDocumentComparator:
    """Unit tests for the DocumentComparator component."""

    def test_requires_at_least_two_documents(self):
        comparator = DocumentComparator()
        with pytest.raises(ValueError, match="requires at least 2 documents"):
            comparator.compare_documents(selected_documents=["single_paper.pdf"])

    def test_two_paper_comparison(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """{
          "matrix": [
            {
              "aspect": "Dataset",
              "values": {
                "Attention_Is_All_You_Need.pdf": "Trained on WMT 2014 English-German (4.5M pairs).",
                "BERT_Pretraining.pdf": "Trained on BooksCorpus and English Wikipedia (3.3B words)."
              },
              "evidence_ids": {
                "Attention_Is_All_You_Need.pdf": ["chunk_transformer_01"],
                "BERT_Pretraining.pdf": ["chunk_bert_01"]
              }
            },
            {
              "aspect": "Results",
              "values": {
                "Attention_Is_All_You_Need.pdf": "Achieved 28.4 BLEU score on WMT 2014 English-to-German.",
                "BERT_Pretraining.pdf": "Pushed GLUE benchmark score to 80.5%."
              },
              "evidence_ids": {
                "Attention_Is_All_You_Need.pdf": ["chunk_transformer_02"],
                "BERT_Pretraining.pdf": ["chunk_bert_02"]
              }
            }
          ],
          "synthesis": "Both papers utilize self-attention; Transformer introduces sequence-to-sequence attention while BERT uses bidirectional pretraining."
        }"""

        citation_engine = CitationEngine()
        comparator = DocumentComparator(
            llm_client=mock_llm,
            citation_engine=citation_engine,
        )
        comparator.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        docs = ["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
        result = comparator.compare_documents(
            selected_documents=docs,
            aspects=["Dataset", "Results"],
        )

        assert len(result["matrix"]) == 2
        assert "Attention_Is_All_You_Need.pdf" in result["markdown_table"]
        assert "BERT_Pretraining.pdf" in result["markdown_table"]
        assert len(result["citations"]) > 0
        assert "synthesis" in result and len(result["synthesis"]) > 0
        assert result["latencies"]["total_latency_ms"] >= 0

    def test_three_plus_paper_comparison(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """{
          "matrix": [
            {
              "aspect": "Dataset",
              "values": {
                "Attention_Is_All_You_Need.pdf": "WMT 2014 English-German (4.5M sentence pairs).",
                "BERT_Pretraining.pdf": "BooksCorpus and English Wikipedia.",
                "ResNet_Deep_Residual.pdf": "ImageNet 2012 classification (1.28M images)."
              },
              "evidence_ids": {
                "Attention_Is_All_You_Need.pdf": ["chunk_transformer_01"],
                "BERT_Pretraining.pdf": ["chunk_bert_01"],
                "ResNet_Deep_Residual.pdf": ["chunk_resnet_01"]
              }
            }
          ],
          "synthesis": "Transformer and BERT focus on language corpora while ResNet evaluates visual benchmarks on ImageNet."
        }"""

        citation_engine = CitationEngine()
        comparator = DocumentComparator(
            llm_client=mock_llm,
            citation_engine=citation_engine,
        )
        comparator.retrieve_evidence_for_documents = MagicMock(return_value=mock_multi_doc_evidence)

        docs = [
            "Attention_Is_All_You_Need.pdf",
            "BERT_Pretraining.pdf",
            "ResNet_Deep_Residual.pdf",
        ]
        result = comparator.compare_documents(
            selected_documents=docs,
            aspects=["Dataset"],
        )

        assert len(result["documents"]) == 3
        # Check all 3 document headers are in markdown table
        assert "| Dimension / Aspect | Attention_Is_All_You_Need.pdf | BERT_Pretraining.pdf | ResNet_Deep_Residual.pdf |" in result["markdown_table"]
        assert len(result["citations"]) == 3

    def test_missing_information_handling(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        # ResNet does not mention NLP translation metrics
        mock_llm.generate.return_value = """{
          "matrix": [
            {
              "aspect": "BLEU Score",
              "values": {
                "Attention_Is_All_You_Need.pdf": "Achieved 28.4 BLEU.",
                "ResNet_Deep_Residual.pdf": "Not found in document."
              },
              "evidence_ids": {
                "Attention_Is_All_You_Need.pdf": ["chunk_transformer_02"],
                "ResNet_Deep_Residual.pdf": []
              }
            }
          ],
          "synthesis": "BLEU score applies to translation in Transformer; not applicable to ResNet."
        }"""

        citation_engine = CitationEngine()
        comparator = DocumentComparator(
            llm_client=mock_llm,
            citation_engine=citation_engine,
        )
        comparator.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "ResNet_Deep_Residual.pdf": mock_multi_doc_evidence["ResNet_Deep_Residual.pdf"],
            }
        )

        result = comparator.compare_documents(
            selected_documents=["Attention_Is_All_You_Need.pdf", "ResNet_Deep_Residual.pdf"],
            aspects=["BLEU Score"],
        )

        assert result["matrix"][0]["values"]["ResNet_Deep_Residual.pdf"] == "Not found in document."
        assert len(result["missing_information"]) == 1
        assert result["missing_information"][0]["document"] == "ResNet_Deep_Residual.pdf"

        # Strictly 0 citations for the missing ResNet cell
        resnet_citations = result["citations_by_document"].get("ResNet_Deep_Residual.pdf", [])
        assert len(resnet_citations) == 0

    def test_citation_attribution_and_metadata_preservation(self, mock_multi_doc_evidence):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """{
          "matrix": [
            {
              "aspect": "Methodology",
              "values": {
                "Attention_Is_All_You_Need.pdf": "Eschews recurrence in favor of multi-head self-attention.",
                "BERT_Pretraining.pdf": "Uses masked language modeling and next sentence prediction."
              },
              "evidence_ids": {
                "Attention_Is_All_You_Need.pdf": ["chunk_transformer_01"],
                "BERT_Pretraining.pdf": ["chunk_bert_01"]
              }
            }
          ],
          "synthesis": "Both rely on attention mechanism."
        }"""

        citation_engine = CitationEngine()
        comparator = DocumentComparator(
            llm_client=mock_llm,
            citation_engine=citation_engine,
        )
        comparator.retrieve_evidence_for_documents = MagicMock(
            return_value={
                "Attention_Is_All_You_Need.pdf": mock_multi_doc_evidence["Attention_Is_All_You_Need.pdf"],
                "BERT_Pretraining.pdf": mock_multi_doc_evidence["BERT_Pretraining.pdf"],
            }
        )

        result = comparator.compare_documents(
            selected_documents=["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"],
            aspects=["Methodology"],
        )

        # Ensure citations point to their respective documents with Step 2 metadata
        for cit in result["citations"]:
            assert cit["document_name"] in ["Attention_Is_All_You_Need.pdf", "BERT_Pretraining.pdf"]
            assert cit["document_id"] in ["doc_trans123", "doc_bert456"]
            assert cit["page_number"] == 1
            assert cit["section"] in ["Abstract & Introduction", "Introduction"]
            assert cit["chunk_id"] in ["chunk_transformer_01", "chunk_bert_01"]
            assert cit["reranker_score"] is not None

    def test_render_markdown_table(self):
        comparator = DocumentComparator()
        matrix = [
            {
                "aspect": "Model",
                "values": {"DocA": "Transformer", "DocB": "BERT"},
            },
            {
                "aspect": "Dataset",
                "values": {"DocA": "WMT14", "DocB": "Not found in document."},
            },
        ]
        table = comparator.render_markdown_table(matrix, ["DocA", "DocB"])
        assert "| Dimension / Aspect | DocA | DocB |" in table
        assert "| Model | Transformer | BERT |" in table
        assert "| Dataset | WMT14 | Not found in document. |" in table


class TestPipelineComparisonIntegration:
    """Integration tests on RAGPipeline with DocumentComparator."""

    def test_pipeline_compare_documents(self):
        from app.rag_pipeline import RAGPipeline

        mock_pdf = MagicMock()
        mock_chunker = MagicMock()
        mock_vector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """{
          "matrix": [
            {
              "aspect": "Results",
              "values": {
                "Paper1.pdf": "Achieved 95% accuracy.",
                "Paper2.pdf": "Achieved 91% accuracy."
              },
              "evidence_ids": {
                "Paper1.pdf": ["c1"],
                "Paper2.pdf": ["c2"]
              }
            }
          ],
          "synthesis": "Paper1 outperforms Paper2 by 4%."
        }"""

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_vector,
            llm_client=mock_llm,
        )

        # Mock retrieve_with_metadata on pipeline
        pipeline.retrieve_with_metadata = MagicMock(return_value=[
            {
                "chunk_id": "c1",
                "id": "c1",
                "text": "Paper 1 achieved 95% accuracy on test set.",
                "metadata": {"document_id": "doc_p1", "source_file": "Paper1.pdf", "page_number": 1, "section": "Results"},
                "retrieval_source": "hybrid",
                "reranker_score": 0.95,
            },
            {
                "chunk_id": "c2",
                "id": "c2",
                "text": "Paper 2 achieved 91% accuracy on test set.",
                "metadata": {"document_id": "doc_p2", "source_file": "Paper2.pdf", "page_number": 1, "section": "Results"},
                "retrieval_source": "hybrid",
                "reranker_score": 0.91,
            },
        ])

        res = pipeline.compare_documents(
            documents=["Paper1.pdf", "Paper2.pdf"],
            aspects=["Results"],
        )

        assert len(res["matrix"]) == 1
        assert len(res["documents"]) == 2
        assert "markdown_table" in res

    def test_single_document_qa_unaffected_regression(self):
        """CRITICAL: Single-document Q&A must continue operating with 100% normal behavior."""
        from app.rag_pipeline import RAGPipeline

        mock_pdf = MagicMock()
        mock_chunker = MagicMock()
        mock_vector = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Normal single document answer."

        pipeline = RAGPipeline(
            pdf_extractor=mock_pdf,
            text_chunker=mock_chunker,
            vector_store=mock_vector,
            llm_client=mock_llm,
        )
        pipeline.retrieve_with_metadata = MagicMock(return_value=[
            {
                "chunk_id": "c_single",
                "id": "c_single",
                "text": "This is test passage for single-document Q&A.",
                "metadata": {"document_id": "doc_single", "source_file": "single.pdf", "page_number": 1, "section": "Intro"},
                "retrieval_source": "hybrid",
                "reranker_score": 0.88,
            }
        ])

        # Execute standard rag_query
        result = pipeline.rag_query("What is the main finding?")

        assert result["response"] == "Normal single document answer."
        assert len(result["retrieved_documents"]) == 1
        assert "citations" in result
        assert result["citations_enabled"] is True
