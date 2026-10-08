"""Web server for the RAG dashboard."""

from pathlib import Path
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from main import setup_pipeline


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
logger = logging.getLogger(__name__)
_pipeline = None
_config = None


def get_pipeline():
    """Create the configured pipeline once, on the first API request."""
    global _pipeline, _config
    if _pipeline is None:
        _pipeline, _config = setup_pipeline()
    return _pipeline, _config


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/status")
def status():
    try:
        pipeline, _ = get_pipeline()
        return jsonify(pipeline.get_status())
    except Exception as error:
        logger.exception("Could not load pipeline status")
        return jsonify({"error": str(error)}), 500


@app.post("/api/ingest")
def ingest():
    """Ingest a PDF file — Step 10 security: size check, magic bytes, sanitized filename."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a PDF file first."}), 400
    if not upload.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        from app.security import sanitize_filename as _sanitize, FileSizeError, FileTypeError, DuplicateDocumentError
    except ImportError:
        _sanitize = None
        FileSizeError = FileTypeError = DuplicateDocumentError = Exception

    try:
        pipeline, config = get_pipeline()

        # Step 10: Sanitize filename
        raw_name = upload.filename or "document.pdf"
        filename = _sanitize(raw_name) if _sanitize else secure_filename(raw_name)
        if not filename:
            return jsonify({"error": "Invalid filename."}), 400

        # Step 10: File-size pre-check (before writing to disk)
        sec_cfg = getattr(config, "security", None)
        max_mb = sec_cfg.max_pdf_size_mb if sec_cfg else 50
        upload.stream.seek(0, 2)
        upload_size = upload.stream.tell()
        upload.stream.seek(0)
        if upload_size > max_mb * 1024 * 1024:
            return jsonify({
                "error": f"File too large ({upload_size / 1024 / 1024:.1f} MB). Maximum allowed: {max_mb} MB."
            }), 413

        pdf_path = config.input_dir / filename
        upload.save(pdf_path)

        result = pipeline.ingest_pdf(
            pdf_path=pdf_path,
            force_recreate=request.form.get("recreate") == "true",
        )
        return jsonify(result)

    except FileTypeError as err:
        logger.warning("File type validation failed: %s", err)
        return jsonify({"error": str(err)}), 415
    except FileSizeError as err:
        logger.warning("File size validation failed: %s", err)
        return jsonify({"error": str(err)}), 413
    except DuplicateDocumentError as err:
        logger.info("Duplicate document rejected: %s", err)
        return jsonify({"error": str(err), "duplicate": True}), 409
    except Exception as error:
        logger.exception("PDF ingestion failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query", "")).strip()
    if not question:
        return jsonify({"error": "Enter a question first."}), 400

    # Step 10: Query sanitization & injection detection
    try:
        from app.security import sanitize_query, detect_prompt_injection
        question = sanitize_query(question)
        is_suspicious, reason = detect_prompt_injection(question)
        if is_suspicious:
            logger.warning("Prompt injection detected in query.")
            return jsonify({
                "error": "Query rejected: potential prompt injection detected.",
                "detail": reason,
            }), 400
    except ImportError:
        pass
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        pipeline, _ = get_pipeline()
        mode = payload.get("retrieval_mode") or payload.get("mode")
        fusion = payload.get("fusion_method") or payload.get("fusion")
        rerank = payload.get("rerank")
        if rerank is None and "no_rerank" in payload:
            rerank = not payload.get("no_rerank")
        verify = payload.get("verify")
        if verify is None and "no_verify" in payload:
            verify = not payload.get("no_verify")
        cite = payload.get("cite")
        if cite is None and "no_cite" in payload:
            cite = not payload.get("no_cite")
        result = pipeline.rag_query(
            query=question,
            n_retrieve=max(1, min(int(payload.get("top_k", 5)), 20)),
            temperature=float(payload.get("temperature", 0.7)),
            max_tokens=max(1, int(payload.get("max_tokens", 2048))),
            mode=mode,
            fusion_method=fusion,
            rerank=rerank,
            verify=verify,
            cite=cite,
        )

        return jsonify(result)
    except Exception as error:
        logger.exception("RAG query failed")
        return jsonify({"error": str(error)}), 500


@app.get("/api/documents")
def list_documents():
    """List all indexed documents in the active vector store."""
    try:
        pipeline, _ = get_pipeline()
        docs = pipeline.list_documents()
        return jsonify({"documents": docs, "count": len(docs)})
    except Exception as error:
        logger.exception("Failed to list indexed documents")
        return jsonify({"error": str(error)}), 500


@app.post("/api/compare")
def compare():
    """Compare 2 or more research papers across key dimensions (Step 7)."""
    payload = request.get_json(silent=True) or {}
    docs = payload.get("documents") or []
    if not isinstance(docs, list) or len(docs) < 2:
        return jsonify({"error": "Please select at least 2 documents to compare."}), 400

    query = str(payload.get("query", "")).strip() or None
    aspects = payload.get("aspects") or None
    top_k = max(1, min(int(payload.get("top_k", 4)), 10))
    temperature = float(payload.get("temperature", 0.2))

    try:
        pipeline, _ = get_pipeline()
        result = pipeline.compare_documents(
            documents=docs,
            query=query,
            aspects=aspects,
            n_chunks_per_doc=top_k,
            temperature=temperature,
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("Document comparison failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/research/analyze")
def research_analyze():
    """Analyze 2 or more research papers for literature review & gaps (Step 8)."""
    payload = request.get_json(silent=True) or {}
    docs = payload.get("documents") or []
    if not isinstance(docs, list) or len(docs) < 2:
        return jsonify({"error": "Please select at least 2 documents to analyze."}), 400

    topic = str(payload.get("topic", "")).strip() or None
    top_k = max(1, min(int(payload.get("top_k", 4)), 10))
    temperature = float(payload.get("temperature", 0.2))

    try:
        pipeline, _ = get_pipeline()
        result = pipeline.analyze_research(
            documents=docs,
            focus_topic=topic,
            n_chunks_per_doc=top_k,
            temperature=temperature,
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("Research intelligence analysis failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/quiz/generate")
def quiz_generate():
    payload = request.get_json(silent=True) or {}
    num_questions = max(1, min(int(payload.get("num_questions", 5)), 15))

    try:
        pipeline, _ = get_pipeline()
        questions = pipeline.generate_quiz(num_questions=num_questions)
        # Includes "context" and "expected_answer" (the grading key) alongside each
        # question — the frontend keeps these in memory and sends them back
        # unchanged when the answer is submitted for grading, but never displays them.
        return jsonify({"questions": questions})
    except Exception as error:
        logger.exception("Quiz generation failed")
        return jsonify({"error": str(error)}), 500


@app.post("/api/quiz/evaluate")
def quiz_evaluate():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    expected_answer = str(payload.get("expected_answer", "")).strip()
    context = str(payload.get("context", "")).strip()
    user_answer = str(payload.get("user_answer", "")).strip()

    if not question or not user_answer:
        return jsonify({"error": "Missing question or answer."}), 400

    try:
        pipeline, _ = get_pipeline()
        result = pipeline.evaluate_answer(
            question=question,
            expected_answer=expected_answer,
            context=context,
            user_answer=user_answer,
        )
        return jsonify(result)
    except Exception as error:
        logger.exception("Answer evaluation failed")
        return jsonify({"error": str(error)}), 500


_dashboard_service = None


def get_dashboard_service():
    """Lazily initialize the DashboardService singleton."""
    global _dashboard_service
    if _dashboard_service is None:
        from app.evaluation.dashboard_service import DashboardService
        pipeline, _ = get_pipeline()
        _dashboard_service = DashboardService(pipeline=pipeline)
    return _dashboard_service


@app.get("/api/evaluation/dashboard")
def evaluation_dashboard():
    """Retrieve full evaluation metrics, comparisons, and parameter sweep results (Step 9)."""
    try:
        service = get_dashboard_service()
        data = service.get_dashboard_payload()
        return jsonify(data)
    except Exception as error:
        logger.exception("Failed to load evaluation dashboard data")
        return jsonify({"error": str(error)}), 500


@app.post("/api/evaluation/run_experiment")
def evaluation_run_experiment():
    """Execute live research experiments on demand (Step 9)."""
    payload = request.get_json(silent=True) or {}
    experiment_type = str(payload.get("experiment", "all")).strip()

    try:
        service = get_dashboard_service()
        result = service.run_live_experiment(experiment_type=experiment_type)
        return jsonify(result)
    except Exception as error:
        logger.exception("Failed to run evaluation experiment")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

