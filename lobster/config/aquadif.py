"""
AQUADIF Tool Taxonomy Configuration.

This module defines the 10-category AQUADIF taxonomy for Lobster AI tools
and the provenance requirements for each category. AQUADIF makes the system
introspectable, enforceable, and teachable to coding agents by declaring
what each tool does (category) and whether it must produce provenance.

For full category definitions and usage examples, see:
    skills/lobster-dev/references/aquadif-contract.md

Categories:
    - IMPORT: Load external data formats into workspace
    - QUALITY: Assess data integrity and calculate QC metrics
    - FILTER: Subset data by removing samples/features
    - PREPROCESS: Transform data representation (normalize, batch correct, scale)
    - ANALYZE: Extract patterns, perform statistical tests, compute embeddings
    - ANNOTATE: Add biological meaning (cell types, gene names, pathway labels)
    - DELEGATE: Hand off work to specialist child agents
    - SYNTHESIZE: Combine or interpret results across analyses
    - UTILITY: Workspace management, status checks, exporting
    - CODE_EXEC: Custom code execution (escape hatch)

Provenance Requirements:
    Required (7 categories): IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE
    Not required (3 categories): DELEGATE, UTILITY, CODE_EXEC (conditional)
"""

import ast
from enum import Enum
from typing import FrozenSet


class AquadifCategory(str, Enum):
    """
    AQUADIF tool taxonomy - 10 categories for classifying Lobster AI tools.

    Using (str, Enum) base allows string comparisons while maintaining type safety:
        AquadifCategory.IMPORT == "IMPORT"  # True
        AquadifCategory.IMPORT.value        # "IMPORT"

    Values match names for simplicity and JSON serialization compatibility.
    """

    IMPORT = "IMPORT"
    QUALITY = "QUALITY"
    FILTER = "FILTER"
    PREPROCESS = "PREPROCESS"
    ANALYZE = "ANALYZE"
    ANNOTATE = "ANNOTATE"
    DELEGATE = "DELEGATE"
    SYNTHESIZE = "SYNTHESIZE"
    UTILITY = "UTILITY"
    CODE_EXEC = "CODE_EXEC"


# Provenance-required categories (7 of 10)
# These categories represent operations that modify scientific data or produce
# analysis results that must be reproducible. Tools in these categories MUST
# call log_tool_usage(ir=ir) to track provenance.
PROVENANCE_REQUIRED: FrozenSet[AquadifCategory] = frozenset(
    {
        AquadifCategory.IMPORT,
        AquadifCategory.QUALITY,
        AquadifCategory.FILTER,
        AquadifCategory.PREPROCESS,
        AquadifCategory.ANALYZE,
        AquadifCategory.ANNOTATE,
        AquadifCategory.SYNTHESIZE,
    }
)


def requires_provenance(primary_category: str) -> bool:
    """
    Check if a category requires provenance tracking.

    Args:
        primary_category: Category string (e.g., "IMPORT", "UTILITY")

    Returns:
        True if the category requires provenance tracking, False otherwise

    Raises:
        ValueError: If the category string is not a valid AquadifCategory

    Example:
        >>> requires_provenance("IMPORT")
        True
        >>> requires_provenance("UTILITY")
        False
        >>> requires_provenance("INVALID")
        ValueError: 'INVALID' is not a valid AquadifCategory
    """
    try:
        category = AquadifCategory(primary_category)
    except ValueError as e:
        raise ValueError(
            f"'{primary_category}' is not a valid AquadifCategory. "
            f"Valid categories: {', '.join(cat.value for cat in AquadifCategory)}"
        ) from e

    return category in PROVENANCE_REQUIRED


def has_provenance_call(tree: ast.AST) -> bool:
    """
    Check if an AST tree contains a log_tool_usage call passing a real ``ir=``.

    Walks the AST looking for method calls to ``log_tool_usage`` that include
    an ``ir=`` keyword argument, indicating proper provenance tracking.

    A literal ``ir=None`` does NOT count. It satisfies the keyword's *name* while
    writing no provenance at all, so accepting it let tools claim
    ``provenance=True`` and pass this check while emitting nothing reproducible.
    Any other value is accepted: a name, call, or attribute cannot be evaluated
    statically, so it is taken at face value rather than guessed at.

    Every ``log_tool_usage`` call is examined, not just the first one carrying an
    ``ir`` keyword — a tool that logs twice, once with real IR and once without,
    still tracks provenance.

    Used by contract test mixin and ``lobster validate-plugin`` CLI command.

    Args:
        tree: Parsed AST tree from tool source code.

    Returns:
        True if a ``log_tool_usage(ir=<not None>)`` call is found, False otherwise.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "log_tool_usage":
                    for keyword in node.keywords:
                        if keyword.arg != "ir":
                            continue
                        if (
                            isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is None
                        ):
                            continue  # ir=None writes no provenance
                        return True
    return False


# ---------------------------------------------------------------------------
# Figure contract: analysis tools never create figures
# ---------------------------------------------------------------------------
# Tools are identified by ``<module of the unwrapped implementation>.<tool.name>``
# (see :func:`figure_tool_key`), never by ``__qualname__``: factory-nested tools carry
# ``<locals>`` in their qualname.

#: Exactly the eight retained transitional figure producers. They keep their
#: existing AQUADIF metadata (ANALYZE, provenance); this set is NOT a category or tag.
#: No wildcard, no pattern: a same-name tool in another module is not a producer. The
#: set and the eight wrappers are removed together in a later release.
TRANSITIONAL_FIGURE_PRODUCERS: FrozenSet[str] = frozenset(
    f"lobster.agents.visualization_expert.{name}"
    for name in (
        "create_umap_plot",
        "create_qc_plots",
        "create_violin_plot",
        "create_feature_plot",
        "create_dot_plot",
        "create_heatmap",
        "create_elbow_plot",
        "create_cluster_composition_plot",
    )
)

#: Trusted figure submitters: they save through the trusted-save path straight to the
#: figure store (not PlotManager). Named exclusions from the figure-contract tool-case
#: harness, covered by core trusted-save tests instead, and reported as "not counted".
#: Not a CODE_EXEC exemption: no other CODE_EXEC tool is listed here.
TRUSTED_FIGURE_SUBMITTERS: FrozenSet[str] = frozenset(
    {"lobster.tools.custom_code_tool.execute_custom_code"}
)

#: Primary categories whose tools must have executed figure-contract cases (bounded
#: coverage): the seven provenance-required categories.
FIGURE_CONTRACT_COVERED: FrozenSet[AquadifCategory] = PROVENANCE_REQUIRED

#: Primary categories explicitly NOT covered by executed figure-contract cases (reported).
FIGURE_CONTRACT_NOT_COVERED: FrozenSet[AquadifCategory] = frozenset(
    {AquadifCategory.DELEGATE, AquadifCategory.UTILITY}
)


def figure_tool_key(tool) -> str:
    """``<module>.<tool name>`` of the callable that actually implements ``tool``.

    The coroutine wins over ``func`` (an async tool's ``func`` is a sync-fallback
    closure), and ``functools.wraps`` chains are unwrapped, matching the provenance
    checks. Raises ``ValueError`` if the tool has no implementation to identify.
    """
    import inspect

    fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    if fn is None:
        raise ValueError(f"tool {getattr(tool, 'name', tool)!r} has no implementation")
    module = getattr(inspect.unwrap(fn), "__module__", None)
    name = getattr(tool, "name", None)
    if not isinstance(module, str) or not isinstance(name, str) or not name:
        raise ValueError("tool implementation has no module or name")
    return f"{module}.{name}"


def is_transitional_figure_producer(key: str) -> bool:
    """True only for one of the exact eight retained producer keys (no pattern matching)."""
    return key in TRANSITIONAL_FIGURE_PRODUCERS
