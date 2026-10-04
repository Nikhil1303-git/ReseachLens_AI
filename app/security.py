"""Security module for ResearchLens AI — Step 10.

Provides:
- File type validation (magic bytes, not just extension)
- File size limit enforcement
- Filename sanitization
- Duplicate document detection (SHA-256 fingerprint)
- Query / input sanitization
- Prompt-injection detection heuristics
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Magic bytes for PDF files ─────────────────────────────────────────────────
_PDF_MAGIC = b"%PDF-"

# Default limits
_DEFAULT_MAX_FILE_SIZE_MB: int = 50
_DEFAULT_MAX_QUERY_LEN: int = 2000

# Prompt-injection patterns (heuristic)
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
    re.compile(r"forget\s+everything", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?!an?\s+AI|an?\s+assistant)", re.I),
    re.compile(r"act\s+as\s+(a\s+)?(?!an?\s+AI|an?\s+assistant)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
    re.compile(r"print\s+(your|the)\s+(system\s+)?prompt", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.I),
]


# ── Security exceptions ───────────────────────────────────────────────────────


class SecurityError(Exception):
    """Raised when a security check fails."""


class FileSizeError(SecurityError):
    """Raised when an uploaded file exceeds the size limit."""


class FileTypeError(SecurityError):
    """Raised when an uploaded file is not a valid PDF."""


class DuplicateDocumentError(SecurityError):
    """Raised when a duplicate document is detected."""


class QueryInjectionError(SecurityError):
    """Raised when a query appears to contain a prompt-injection attempt."""


# ── Core validators ───────────────────────────────────────────────────────────


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex-digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex-digest from raw bytes.

    Args:
        data: Raw bytes.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


def validate_file_size(
    path: Path,
    max_mb: Optional[int] = None,
) -> int:
    """Assert that a file does not exceed the maximum allowed size.

    Args:
        path: Path to the file.
        max_mb: Maximum size in megabytes.  Defaults to the
            ``MAX_PDF_SIZE_MB`` environment variable or 50 MB.

    Returns:
        File size in bytes.

    Raises:
        FileSizeError: If the file exceeds the size limit.
    """
    if max_mb is None:
        max_mb = int(os.getenv("MAX_PDF_SIZE_MB", str(_DEFAULT_MAX_FILE_SIZE_MB)))

    size_bytes = path.stat().st_size
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileSizeError(
            f"File '{path.name}' is {size_bytes / 1024 / 1024:.1f} MB which exceeds "
            f"the {max_mb} MB limit."
        )
    logger.debug("File size OK: %s (%.1f MB)", path.name, size_bytes / 1024 / 1024)
    return size_bytes


def validate_pdf_magic(path: Path) -> None:
    """Verify that a file starts with the PDF magic bytes ``%PDF-``.

    Args:
        path: Path to the file.

    Raises:
        FileTypeError: If the file does not begin with the PDF magic bytes.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
    except OSError as exc:
        raise FileTypeError(f"Cannot read file '{path.name}': {exc}") from exc

    if not header.startswith(_PDF_MAGIC):
        raise FileTypeError(
            f"File '{path.name}' is not a valid PDF (invalid magic bytes). "
            "Only genuine PDF files are accepted."
        )
    logger.debug("PDF magic bytes OK: %s", path.name)


def sanitize_filename(filename: str) -> str:
    """Return a safe, filesystem-clean version of a filename.

    - Strips leading/trailing whitespace and dots.
    - Removes path separators and null bytes.
    - Replaces sequences of non-alphanumeric/dot/dash/underscore chars with ``_``.
    - Limits the stem to 200 characters to avoid OS limits.

    Args:
        filename: Original filename string.

    Returns:
        Sanitized filename string.
    """
    # Remove path traversal components
    filename = os.path.basename(filename)
    # Strip null bytes and control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    # Replace unsafe characters
    filename = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)
    # Collapse consecutive underscores
    filename = re.sub(r"_+", "_", filename)
    # Strip leading/trailing dots and underscores
    filename = filename.strip("._")

    if not filename:
        filename = "document.pdf"

    # Ensure .pdf extension
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".pdf":
        filename = stem[:200] + ".pdf"
    else:
        filename = stem[:200] + ext

    logger.debug("Sanitized filename: %s", filename)
    return filename


def check_duplicate(
    sha256: str,
    known_hashes: set[str],
) -> bool:
    """Check whether a document (identified by its SHA-256) already exists.

    Args:
        sha256: SHA-256 hex string of the incoming document.
        known_hashes: Set of SHA-256 hex strings already indexed.

    Returns:
        ``True`` if the document is a duplicate, ``False`` otherwise.
    """
    return sha256 in known_hashes


def sanitize_query(query: str, max_len: Optional[int] = None) -> str:
    """Sanitize a user query string.

    - Strips leading/trailing whitespace.
    - Truncates to ``max_len`` characters (default: ``MAX_QUERY_LEN`` env var or 2000).
    - Removes ASCII control characters (but preserves newlines for multi-line queries).

    Args:
        query: Raw user query.
        max_len: Maximum allowed length.

    Returns:
        Sanitized query string.

    Raises:
        ValueError: If the sanitized query is empty.
    """
    if max_len is None:
        max_len = int(os.getenv("MAX_QUERY_LEN", str(_DEFAULT_MAX_QUERY_LEN)))

    # Strip control chars except newline / tab
    sanitized = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", query)
    sanitized = sanitized.strip()

    if len(sanitized) > max_len:
        logger.warning("Query truncated from %d to %d characters.", len(sanitized), max_len)
        sanitized = sanitized[:max_len]

    if not sanitized:
        raise ValueError("Query is empty after sanitization.")

    return sanitized


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """Heuristically detect prompt-injection attempts in user-supplied text.

    This is a best-effort guard against common jailbreak / injection patterns.
    It is NOT a guarantee of security and should be used alongside other controls.

    Args:
        text: User-supplied text (query or document content snippet).

    Returns:
        A tuple ``(is_suspicious, reason)`` where ``is_suspicious`` is ``True``
        when a pattern matched and ``reason`` is a description.
    """
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            reason = f"Suspicious pattern detected: '{m.group()[:60]}'"
            logger.warning("Prompt injection heuristic triggered: %s", reason)
            return True, reason
    return False, ""


# ── Aggregate validation entry-point ─────────────────────────────────────────


def validate_upload(
    path: Path,
    known_hashes: Optional[set[str]] = None,
    max_mb: Optional[int] = None,
) -> dict:
    """Run all upload security checks on a saved PDF file.

    Performs (in order):
    1. File size check
    2. PDF magic-byte validation
    3. Duplicate detection (if ``known_hashes`` is provided)

    Args:
        path: Path to the uploaded file (already saved to disk).
        known_hashes: Set of SHA-256 hashes of previously indexed documents.
        max_mb: File size limit in megabytes.

    Returns:
        A dict with keys ``sha256``, ``size_bytes``, and ``is_duplicate``.

    Raises:
        FileSizeError: File too large.
        FileTypeError: Not a valid PDF.
        DuplicateDocumentError: Duplicate document.
    """
    size_bytes = validate_file_size(path, max_mb=max_mb)
    validate_pdf_magic(path)

    sha256 = compute_file_sha256(path)
    is_dup = False
    if known_hashes is not None:
        is_dup = check_duplicate(sha256, known_hashes)
        if is_dup:
            raise DuplicateDocumentError(
                f"Document '{path.name}' has already been indexed "
                f"(SHA-256: {sha256[:16]}…). Use force_recreate=True to re-index."
            )

    logger.info(
        "Upload validation passed: %s | SHA-256=%s | size=%.1f KB",
        path.name,
        sha256[:16],
        size_bytes / 1024,
    )
    return {
        "sha256": sha256,
        "size_bytes": size_bytes,
        "is_duplicate": is_dup,
    }
