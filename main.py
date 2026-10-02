#!/usr/bin/env python
"""Command-line interface for RAG application."""

import argparse
import sys
from pathlib import Path
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import load_config, validate_config
from app.utils import setup_logging, get_logger
from app.pdf.text_extractor import PDFTextExtractor
from app.pdf.table_extractor import PDFTableExtractor
from app.text_chunker import TextChunker
from app.embeddings.chroma_store import ChromaVectorStore
from app.llm.client import LLMFactory
from app.rag_pipeline import RAGPipeline


def setup_pipeline():
    """Initialize and return configured RAG pipeline.

    Returns:
        RAGPipeline instance

    Raises:
        ValueError: If configuration is invalid
    """
    config = load_config()
    validate_config(config)

    # Setup logging
    setup_logging(log_level=config.log_level)
    logger = get_logger("main")
    logger.info("Configuration loaded and validated")

    # Initialize components
    pdf_extractor = PDFTextExtractor(method=config.pdf.extraction_method)
    text_chunker = TextChunker(
        chunk_size=config.pdf.chunk_size,
        chunk_overlap=config.pdf.chunk_overlap,
    )
    vector_store = ChromaVectorStore(
        collection_name=config.chroma.collection_name,
        persist_directory=config.chroma.persist_directory,
        embedding_model=config.chroma.embedding_model,
    )
    llm_client = LLMFactory.create(
        provider=config.llm.provider,
        api_key=config.llm.api_key,
        model=config.llm.model,
    )

    pipeline = RAGPipeline(
        pdf_extractor=pdf_extractor,
        text_chunker=text_chunker,
        vector_store=vector_store,
        llm_client=llm_client,
        config=config,
    )

    return pipeline, config


def cmd_ingest(args):
    """Ingest PDF command handler.

    Args:
        args: Command arguments
    """
    logger = get_logger("ingest")

    pipeline, config = setup_pipeline()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        result = pipeline.ingest_pdf(
            pdf_path=pdf_path,
            force_recreate=args.recreate,
        )
        logger.info(f"Ingestion result: {result}")
        print(f"\n✓ Successfully ingested PDF")
        print(f"  Pages: {result['pages_extracted']}")
        print(f"  Chunks: {result['chunks_created']}")
        print(f"  Collection: {result['collection_name']}")
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_query(args):
    """Query command handler.

    Args:
        args: Command arguments
    """
    logger = get_logger("query")

    pipeline, config = setup_pipeline()

    try:
        rerank_flag = False if getattr(args, "no_rerank", False) else None
        verify_flag = False if getattr(args, "no_verify", False) else None
        cite_flag = False if getattr(args, "no_cite", False) else None
        result = pipeline.rag_query(
            query=args.query,
            n_retrieve=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            mode=getattr(args, "mode", None),
            fusion_method=getattr(args, "fusion", None),
            rerank=rerank_flag,
            verify=verify_flag,
            cite=cite_flag,
        )

        print(f"\n{'='*60}")
        print(f"Query: {result['query']}")
        print(f"Retrieval Mode: {result.get('retrieval_mode', 'default')}")
        if "fusion_method" in result:
            print(f"Fusion Method: {result['fusion_method']}")
        if result.get("reranking_enabled"):
            print(f"Neural Reranker: {result.get('reranker_model', 'Enabled')}")
        elif getattr(args, "no_rerank", False):
            print("Neural Reranker: Disabled (--no-rerank)")
        if "retrieval_latency_ms" in result:
            print(f"Retrieval Latency: {result['retrieval_latency_ms']} ms")
        if "generation_latency_ms" in result:
            print(f"Generation Latency: {result['generation_latency_ms']} ms")
        if "verification_latency_ms" in result:
            print(f"Verification Latency: {result['verification_latency_ms']} ms")
        if "citation_latency_ms" in result:
            print(f"Citation Latency: {result['citation_latency_ms']} ms")
        if "total_latency_ms" in result:
            print(f"Total Pipeline Latency: {result['total_latency_ms']} ms")
        print(f"{'='*60}")
        print(f"\nRetrieved {result['n_documents_retrieved']} documents:\n")

        chunks = result.get("retrieved_chunks") or []
        if chunks:
            for i, chunk in enumerate(chunks, 1):
                source_info = chunk.get("retrieval_source", "vector")
                score_val = chunk.get("hybrid_score", chunk.get("dense_score", 0.0))
                rerank_score = chunk.get("reranker_score")
                page_info = chunk.get("metadata", {}).get("page_number", "?")
                sec_info = chunk.get("metadata", {}).get("section", "")
                
                score_str = f"Score: {score_val:.4f}"
                if rerank_score is not None:
                    score_str += f" | Rerank Score: {rerank_score:.4f}"

                header = f"[Document {i}] (Source: {source_info}, {score_str}, Page: {page_info}"
                if sec_info:
                    header += f", Section: {sec_info}"
                header += ")"
                print(header)

                doc_text = chunk.get("text", "")
                print(doc_text[:200] + "..." if len(doc_text) > 200 else doc_text)
                print()
        else:
            for i, doc in enumerate(result["retrieved_documents"], 1):
                print(f"[Document {i}]")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
                print()

        print(f"{'='*60}")
        print("Response:")
        print(f"{'='*60}")
        print(result["response"])

        if result.get("verification_enabled") and "verification" in result:
            v = result["verification"]
            status_label = v.get("status", "unknown").upper()
            status_icons = {
                "SUPPORTED": "[SUPPORTED]",
                "PARTIALLY_SUPPORTED": "[PARTIALLY SUPPORTED]",
                "CONTRADICTED": "[CONTRADICTED]",
                "INSUFFICIENT_EVIDENCE": "[INSUFFICIENT EVIDENCE]",
            }
            icon = status_icons.get(status_label, f"[{status_label}]")
            print(f"\n{'='*60}")
            print(f"Evidence Verification: {icon}")
            print(f"Verification Latency: {v.get('verification_latency_ms', 0)} ms")
            print(f"{'='*60}")
            if v.get("claims"):
                print("Claim-Level Assessment:")
                for c in v["claims"]:
                    c_status = c.get("status", "").upper()
                    c_icon = "(+)" if c_status == "SUPPORTED" else ("(-)" if c_status == "CONTRADICTED" else "(?)")
                    ev_ids = f" (Evidence: {', '.join(c.get('evidence_ids', []))})" if c.get("evidence_ids") else ""
                    print(f"  {c_icon} [{c_status}] {c.get('claim')}{ev_ids}")
        elif getattr(args, "no_verify", False):
            print("\nEvidence Verification: Disabled (--no-verify)")

        # Grounded Citations & Provenance Display (Step 6)
        if result.get("citations_enabled"):
            citations = result.get("citations", [])
            print(f"\n{'='*60}")
            print(f"Grounded Citations & Provenance ({len(citations)} citations):")
            print(f"{'='*60}")
            if citations:
                for cit in citations:
                    c_idx = cit.get("citation_index", 1)
                    c_claim = cit.get("claim", "")
                    c_status = cit.get("claim_status", "supported").upper()
                    doc = cit.get("document_name", "Unknown Document")
                    page = cit.get("page_number", 1)
                    sec = cit.get("section") or "General"
                    cid = cit.get("chunk_id", "chunk_unknown")
                    rerank_sc = cit.get("reranker_score")
                    score_info = (
                        f"Rerank Score: {rerank_sc:.4f}"
                        if rerank_sc is not None
                        else f"Score: {cit.get('confidence_score', 1.0):.4f}"
                    )
                    ev_snippet = cit.get("evidence_text", "")

                    status_sym = "(+)" if c_status == "SUPPORTED" else ("(~)" if c_status == "PARTIALLY_SUPPORTED" else "(-)")
                    print(f"[{c_idx}] {status_sym} [{c_status}] Claim: \"{c_claim}\"")
                    print(f"    Source: [Document: {doc} | Page: {page} | Section: {sec} | Chunk: {cid} | {score_info}]")
                    if ev_snippet:
                        print(f"    Evidence: \"{ev_snippet}\"")
                    print()
            else:
                print("No citations generated (No grounded evidence matched answer claims).")

            unsupported = result.get("unsupported_claims", [])
            if unsupported:
                print(f"Unsupported Claims ({len(unsupported)} ungrounded - 0 fake citations):")
                for un_c in unsupported:
                    print(f"  (?) [INSUFFICIENT EVIDENCE] \"{un_c}\"")
                print()
        elif getattr(args, "no_cite", False):
            print("\nGrounded Citations: Disabled (--no-cite)")


    except Exception as e:
        logger.error(f"Error during query: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_compare(args):
    """Compare documents command handler (Step 7).

    Args:
        args: Command arguments
    """
    logger = get_logger("compare")
    pipeline, config = setup_pipeline()

    if len(args.docs) < 2:
        print("\nError: Multi-document comparison requires at least 2 documents. Please provide at least 2 papers with --docs.")
        sys.exit(1)

    try:
        result = pipeline.compare_documents(
            documents=args.docs,
            query=args.query,
            aspects=args.aspects,
            n_chunks_per_doc=args.top_k,
            temperature=args.temperature,
        )

        docs_str = ", ".join(result["documents"])
        print(f"\n{'='*60}")
        print(f"MULTI-DOCUMENT COMPARISON MATRIX ({len(result['documents'])} Papers)")
        print(f"Documents: {docs_str}")
        print(f"Focus Query: {result['query']}")
        if "latencies" in result:
            lats = result["latencies"]
            print(
                f"Latency: Total {lats.get('total_latency_ms', 0)} ms "
                f"(Ret: {lats.get('retrieval_latency_ms', 0)} ms | "
                f"Gen: {lats.get('generation_latency_ms', 0)} ms | "
                f"Cite: {lats.get('citation_latency_ms', 0)} ms)"
            )
        print(f"{'='*60}\n")

        print(result["markdown_table"])
        print()

        if result.get("synthesis"):
            print(f"{'='*60}")
            print("Comparative Synthesis:")
            print(f"{'='*60}")
            print(result["synthesis"])
            print()

        citations = result.get("citations", [])
        if citations:
            print(f"{'='*60}")
            print(f"Grounded Citations & Provenance ({len(citations)} citations):")
            print(f"{'='*60}")
            for cit in citations:
                c_idx = cit.get("citation_index", 1)
                c_claim = cit.get("claim", "")
                c_status = cit.get("claim_status", "supported").upper()
                doc = cit.get("document_name", "Unknown Document")
                page = cit.get("page_number", 1)
                sec = cit.get("section") or "General"
                cid = cit.get("chunk_id", "chunk_unknown")
                rerank_sc = cit.get("reranker_score")
                score_info = (
                    f"Rerank Score: {rerank_sc:.4f}"
                    if rerank_sc is not None
                    else f"Score: {cit.get('confidence_score', 1.0):.4f}"
                )
                ev_snippet = cit.get("evidence_text", "")

                status_sym = "(+)" if c_status == "SUPPORTED" else ("(~)" if c_status == "PARTIALLY_SUPPORTED" else "(-)")
                print(f"[{c_idx}] {status_sym} [{c_status}] Claim: \"{c_claim}\"")
                print(f"    Source: [Document: {doc} | Page: {page} | Section: {sec} | Chunk: {cid} | {score_info}]")
                if ev_snippet:
                    print(f"    Evidence: \"{ev_snippet}\"")
                print()

        missing = result.get("missing_information", [])
        if missing:
            print(f"{'='*60}")
            print(f"Missing Information ({len(missing)} aspects not found - 0 fake citations):")
            print(f"{'='*60}")
            for item in missing:
                print(f"  (?) {item.get('document')} - {item.get('aspect')}: \"Not found in document.\"")
            print()

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_research(args):
    """Research intelligence and research gap detection command handler (Step 8).

    Args:
        args: Command arguments
    """
    logger = get_logger("research")
    pipeline, config = setup_pipeline()

    if len(args.docs) < 2:
        print("\nError: Research intelligence requires at least 2 documents. Please provide at least 2 papers with --docs.")
        sys.exit(1)

    try:
        result = pipeline.analyze_research(
            documents=args.docs,
            focus_topic=args.topic,
            n_chunks_per_doc=args.top_k,
            temperature=args.temperature,
        )

        docs_str = ", ".join(result["documents"])
        print(f"\n{'='*70}")
        print(f"RESEARCH INTELLIGENCE & GAP DETECTION ({len(result['documents'])} Papers)")
        print(f"Papers: {docs_str}")
        print(f"Topic: {result['topic']}")
        if "latencies" in result:
            lats = result["latencies"]
            print(
                f"Latency: Total {lats.get('total_latency_ms', 0)} ms "
                f"(Ret: {lats.get('retrieval_latency_ms', 0)} ms | "
                f"Gen: {lats.get('generation_latency_ms', 0)} ms | "
                f"Cite: {lats.get('citation_latency_ms', 0)} ms)"
            )
        print(f"{'='*70}\n")

        # 1. Literature Review
        lr = result.get("literature_review", {})
        if lr.get("summary"):
            print("--- LITERATURE REVIEW SUMMARY ---")
            print(lr["summary"])
            print()

        # Key Findings
        if lr.get("key_findings"):
            print("--- KEY FINDINGS ---")
            for doc, finding in lr["key_findings"].items():
                print(f"  * [{doc}]: {finding}")
            print()

        # Common Themes & Differences
        if lr.get("common_themes"):
            print("--- COMMON THEMES ---")
            for theme in lr["common_themes"]:
                print(f"  * {theme}")
            print()

        if lr.get("differences"):
            print("--- KEY DIFFERENCES ---")
            for diff in lr["differences"]:
                print(f"  * {diff}")
            print()

        # Limitations
        if lr.get("limitations"):
            print("--- PAPER LIMITATIONS ---")
            for doc, lim in lr["limitations"].items():
                print(f"  * [{doc}]: {lim}")
            print()

        # 2. Research Gaps
        gaps = result.get("research_gaps", [])
        print(f"{'='*70}")
        print(f"IDENTIFIED RESEARCH GAPS ({len(gaps)} gaps detected)")
        print(f"{'='*70}")
        for gap in gaps:
            gid = gap.get("gap_id", "GAP")
            cat = gap.get("category", "limitation").upper()
            desc = gap.get("description", "")
            ev = gap.get("evidence", "")
            val_note = gap.get("validation_note", "Requires researcher validation.")
            src = ", ".join(gap.get("source_papers", []))

            print(f"[{gid}] [{cat}] Papers: {src}")
            print(f"  Description: {desc}")
            if ev:
                print(f"  Evidence: \"{ev}\"")
            print(f"  Notice: (!) {val_note}")
            print()

        # 3. Research Questions
        rqs = result.get("research_questions", [])
        if rqs:
            print(f"{'='*70}")
            print(f"SUGGESTED RESEARCH QUESTIONS ({len(rqs)} questions)")
            print(f"{'='*70}")
            for rq in rqs:
                qid = rq.get("question_id", "RQ")
                target_gap = rq.get("gap_id", "")
                q_text = rq.get("question", "")
                rat = rq.get("rationale", "")
                print(f"[{qid}] (Addresses: {target_gap})")
                print(f"  Question: {q_text}")
                if rat:
                    print(f"  Rationale: {rat}")
                print()

        # 4. Citations
        citations = result.get("citations", [])
        if citations:
            print(f"{'='*70}")
            print(f"GROUNDED CITATIONS & PROVENANCE ({len(citations)} citations):")
            print(f"{'='*70}")
            for cit in citations:
                c_idx = cit.get("citation_index", 1)
                c_claim = cit.get("claim", "")
                c_status = cit.get("claim_status", "supported").upper()
                doc = cit.get("document_name", "Unknown Document")
                page = cit.get("page_number", 1)
                sec = cit.get("section") or "General"
                cid = cit.get("chunk_id", "chunk_unknown")
                rerank_sc = cit.get("reranker_score")
                score_info = (
                    f"Rerank Score: {rerank_sc:.4f}"
                    if rerank_sc is not None
                    else f"Score: {cit.get('confidence_score', 1.0):.4f}"
                )
                ev_snippet = cit.get("evidence_text", "")

                status_sym = "(+)" if c_status == "SUPPORTED" else ("(~)" if c_status == "PARTIALLY_SUPPORTED" else "(-)")
                print(f"[{c_idx}] {status_sym} [{c_status}] Claim: \"{c_claim}\"")
                print(f"    Source: [Document: {doc} | Page: {page} | Section: {sec} | Chunk: {cid} | {score_info}]")
                if ev_snippet:
                    print(f"    Evidence: \"{ev_snippet}\"")
                print()

        # Missing evidence
        missing = result.get("missing_evidence", [])
        if missing:
            print(f"{'='*70}")
            print(f"MISSING EVIDENCE ({len(missing)} areas lacking evidence - 0 fake citations):")
            print(f"{'='*70}")
            for item in missing:
                print(f"  (?) {item.get('document')} - {item.get('dimension')}: \"Insufficient evidence.\"")
            print()

    except Exception as e:
        logger.error(f"Error during research intelligence analysis: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_extract_tables(args):
    """Extract tables command handler.

    Args:
        args: Command arguments
    """
    logger = get_logger("extract_tables")

    try:
        extractor = PDFTableExtractor(method="pdfplumber")
        pdf_path = Path(args.input)

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)

        output_path = Path(args.output) if args.output else None
        json_result = extractor.extract_tables_to_json(pdf_path, output_path)

        if output_path:
            print(f"\n✓ Tables extracted and saved to: {output_path}")
        else:
            print(f"\n✓ Tables extracted ({len(json_result)} bytes)")
            print(json_result)

    except Exception as e:
        logger.error(f"Error extracting tables: {str(e)}", exc_info=True)
        sys.exit(1)


def cmd_status(args):
    """Status command handler.

    Args:
        args: Command arguments
    """
    logger = get_logger("status")

    try:
        pipeline, config = setup_pipeline()
        status = pipeline.get_status()

        print(f"\n{'='*60}")
        print("RAG Pipeline Status")
        print(f"{'='*60}")
        print(f"LLM Provider: {status['llm_provider']}")
        print(f"LLM Model: {status['llm_model']}")
        print(f"Vector Store: {status['vector_store']}")
        print(f"Collection: {status['collection_info'].get('name', 'N/A')}")
        print(
            f"Documents: {status['collection_info'].get('document_count', 'N/A')}"
        )
        print(f"PDF Extractor: {status['pdf_extractor']}")
        print(f"Chunk Size: {status['chunk_size']}")
        print(f"Chunk Overlap: {status['chunk_overlap']}")

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System with LangChain and ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF
  python main.py ingest --input document.pdf
  
  # Query the ingested documents
  python main.py query --query "What is the main topic?"
  
  # Extract tables from PDF
  python main.py extract-tables --input document.pdf --output tables.json
  
  # Show pipeline status
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest PDF into vector store"
    )
    ingest_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to PDF file",
    )
    ingest_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection (delete existing)",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query documents")
    query_parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Query text",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    query_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens in response (default: 2048)",
    )
    query_parser.add_argument(
        "--mode",
        "-m",
        choices=["hybrid", "vector"],
        default=None,
        help="Retrieval mode: hybrid or vector (default: from config)",
    )
    query_parser.add_argument(
        "--fusion",
        "-f",
        choices=["weighted", "rrf"],
        default=None,
        help="Hybrid fusion method: weighted or rrf (default: from config)",
    )
    query_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable neural cross-encoder reranking (use raw hybrid/vector results)",
    )
    query_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable post-generation evidence verification",
    )
    query_parser.add_argument(
        "--no-cite",
        action="store_true",
        help="Disable grounded citation generation and provenance tracking",
    )
    query_parser.set_defaults(func=cmd_query)


    # Extract tables command
    tables_parser = subparsers.add_parser(
        "extract-tables", help="Extract tables from PDF"
    )
    tables_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to PDF file",
    )
    tables_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path",
    )
    tables_parser.set_defaults(func=cmd_extract_tables)

    # Compare command (Step 7)
    compare_parser = subparsers.add_parser(
        "compare", help="Compare 2 or more research papers across key dimensions"
    )
    compare_parser.add_argument(
        "--docs",
        "-d",
        nargs="+",
        required=True,
        help="List of 2 or more PDF filenames or document IDs to compare",
    )
    compare_parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Optional freeform comparison question or topic focus",
    )
    compare_parser.add_argument(
        "--aspects",
        "-a",
        nargs="+",
        default=None,
        help="Specific comparison aspects (e.g. Dataset Methodology Results)",
    )
    compare_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve per paper (default: 4)",
    )
    compare_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Research Intelligence command (Step 8)
    research_parser = subparsers.add_parser(
        "research", help="Perform literature review and research gap discovery across 2+ papers"
    )
    research_parser.add_argument(
        "--docs",
        "-d",
        nargs="+",
        required=True,
        help="List of 2 or more PDF filenames or document IDs to analyze",
    )
    research_parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default=None,
        help="Optional specific research topic or focus area",
    )
    research_parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve per paper (default: 4)",
    )
    research_parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)",
    )
    research_parser.set_defaults(func=cmd_research)

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show pipeline status"
    )
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
