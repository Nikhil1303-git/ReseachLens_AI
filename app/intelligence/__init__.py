"""Research Intelligence and Research Gap Detection Module (Step 8).

Provides multi-paper literature review synthesis, evidence-grounded research gap
detection, research question formulation, and claim citation resolution.
"""

from app.intelligence.engine import (
    INSUFFICIENT_EVIDENCE_MARKER,
    RESEARCH_GAP_CATEGORIES,
    RESEARCH_INTELLIGENCE_DIMENSIONS,
    ResearchIntelligenceEngine,
    VALIDATION_DISCLAIMER,
)

__all__ = [
    "ResearchIntelligenceEngine",
    "RESEARCH_INTELLIGENCE_DIMENSIONS",
    "RESEARCH_GAP_CATEGORIES",
    "VALIDATION_DISCLAIMER",
    "INSUFFICIENT_EVIDENCE_MARKER",
]
