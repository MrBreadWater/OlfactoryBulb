"""Internal extraction target for reusable computational-neuroscience infrastructure.

This package is intentionally lightweight for now.  It does not yet own the
runtime code paths that still live under ``olfactorybulb`` and ``tools``.
Instead, it provides the first standardized inventory of which subsystems are
candidates for extraction, what their current source locations are, and what
their future package boundaries should look like.
"""

from .inventory import EXTRACTION_CANDIDATES, ExtractionCandidate, repo_specific_areas, target_module_index

__all__ = [
    "EXTRACTION_CANDIDATES",
    "ExtractionCandidate",
    "repo_specific_areas",
    "target_module_index",
]

__version__ = "0.0.0-internal"
