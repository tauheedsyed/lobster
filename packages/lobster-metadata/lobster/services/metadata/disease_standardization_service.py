"""
Disease Standardization Service

Normalizes disease/condition terminology and filters samples by type (fecal vs tissue).
Designed for IBD microbiome studies (CRC, UC, CD, Healthy controls).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Graceful fallback: Import premium DiseaseOntologyService if available
# Public version (lobster-local) will use hardcoded fallback
try:
    from lobster.services.metadata.disease_ontology_service import (
        DiseaseOntologyService,
    )

    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False
    logger.warning(
        "DiseaseOntologyService not available, using hardcoded disease mappings. "
        "Install lobster-metadata for centralized ontology management."
    )


class DiseaseStandardizationService:
    """
    Service for standardizing disease terminology and filtering by sample type.

    Key capabilities:
    - Maps messy disease labels to standard categories (CRC, UC, CD, Healthy)
    - Conservative matching (exact → whole-phrase contains → unmapped); blank,
      negated and ambiguous values stay unmapped
    - Sample type filtering (fecal, biopsy, tissue, etc.)
    - Provenance tracking for all transformations
    """

    # Exclusion configuration: prevents false positives from fuzzy matching
    # Uses allow-list approach for ambiguous terms (Gemini-recommended architecture)
    # Separates universally-blocked phrases from context-dependent ambiguous words
    DISEASE_EXCLUSION_CONFIG = {
        "healthy": {
            # 1. Universally blocked substrings (always invalid)
            "block_substrings": [
                "negative control",
                "positive control",
                "technical control",
                "disease control",
                "control group",
                "normal tissue",
                "normal flora",
                "normal distribution",
                "normal development",
            ],
            # 2. Ambiguous terms requiring context validation (allow-list approach)
            "ambiguous_terms": {
                "control": {
                    # Block if EXACTLY "control" (no context)
                    "block_exact": ["control"],
                    # Allow only these specific compound phrases
                    "allowed_phrases": ["healthy control"],
                },
                "normal": {
                    # Block if EXACTLY "normal" (no context)
                    "block_exact": ["normal"],
                    # Allow only in this valid compound context
                    "allowed_phrases": [],
                },
            },
        },
    }

    # Standard disease mappings - loaded from DiseaseOntologyService (premium)
    # or fallback to hardcoded mappings (public lobster-local)
    # See lobster/config/disease_ontology.json for the single source of truth
    # NOTE: Generic terms removed to prevent misclassification
    # (e.g., "lung adenocarcinoma" should NOT map to CRC)
    # Trade-off: False negatives < False positives for scientific validity
    # REMOVED: "control" - too generic (matches negative/positive/technical controls)
    # REMOVED: "normal" - too generic (matches "normal tissue" in tumor studies)
    # REMOVED: "ctrl" - abbreviation too ambiguous without context

    # Fallback for public version (when DiseaseOntologyService not available)
    _DISEASE_MAPPINGS_FALLBACK = {
        # Colorectal Cancer (CRC)
        "crc": [
            "crc",
            "colorectal cancer",
            "colon cancer",
            "rectal cancer",
            "colorectal carcinoma",
            "colon carcinoma",
            "rectal carcinoma",
            "colorectal adenocarcinoma",
            "colon adenocarcinoma",
            "rectal adenocarcinoma",
            "colorectal",
            "colon_cancer",
            "rectal_cancer",
        ],
        # Ulcerative Colitis (UC)
        "uc": [
            "uc",
            "ulcerative colitis",
            "colitis ulcerosa",
            "ulcerative_colitis",
            "ulcerative",
            "uc_",
        ],
        # Crohn's Disease (CD)
        "cd": [
            "cd",
            "crohn's disease",
            "crohns disease",
            "crohn disease",
            "crohn's",
            "crohns",
            "crohn",
            "cd_",
            "crohns_disease",
        ],
        # Healthy Controls
        "healthy": [
            "healthy",
            "healthy control",
            "healthy volunteer",
            "healthy donor",
            "healthy subject",
            "healthy controls",
            "healthy volunteers",
            "healthy donors",
            "healthy subjects",
            "non-ibd",
            "non ibd",
            "non_ibd",
            "non-diseased",
            "non diseased",
            "non_diseased",
            "disease-free",
            "disease free",
        ],
    }

    DISEASE_MAPPINGS: Dict[str, List[str]] = {}  # Populated in __init__

    # Sample type mappings
    # NOTE: Generic terms (biopsy, tissue, mucosa) require organ context
    # to prevent false positives (e.g., "liver biopsy" → gut)
    SAMPLE_TYPE_MAPPINGS = {
        "fecal": ["fecal", "feces", "stool", "faecal", "faeces", "fecal sample"],
        "gut": [
            "gut",
            "intestinal",
            "colon",
            "colonic",
            "rectal",
            "ileal",
            # Organ-specific biopsy/tissue terms
            "gut biopsy",
            "colon biopsy",
            "rectal biopsy",
            "intestinal biopsy",
            "gut tissue",
            "colon tissue",
            "intestinal tissue",
            "colonic mucosa",
            "rectal mucosa",
            "intestinal mucosa",
            # REMOVED: "biopsy" - too generic (could be liver, lung, breast)
            # REMOVED: "tissue" - too generic (could be any tissue)
            # REMOVED: "mucosal", "mucosa" - need organ context
        ],
    }

    def __init__(self):
        """Initialize the disease standardization service."""
        # Load disease mappings from centralized ontology service (Phase 1)
        # Graceful fallback: Use premium service if available, otherwise use hardcoded fallback
        if HAS_ONTOLOGY_SERVICE:
            # Premium version: Load from DiseaseOntologyService
            # See lobster/config/disease_ontology.json for the single source of truth
            ontology = DiseaseOntologyService.get_instance()
            self.DISEASE_MAPPINGS = ontology.get_standardization_variants()
            logger.debug("Using DiseaseOntologyService (premium)")
        else:
            # Public version (lobster-local): Use hardcoded fallback
            self.DISEASE_MAPPINGS = self._DISEASE_MAPPINGS_FALLBACK
            logger.debug("Using hardcoded disease mappings (public fallback)")

        # Build reverse lookup for faster matching
        self._reverse_disease_map = {}
        for standard_term, variants in self.DISEASE_MAPPINGS.items():
            for variant in variants:
                self._reverse_disease_map[variant.lower()] = standard_term

        self._reverse_sample_map = {}
        for sample_type, variants in self.SAMPLE_TYPE_MAPPINGS.items():
            for variant in variants:
                self._reverse_sample_map[variant.lower()] = sample_type

    def standardize_disease_terms(
        self, metadata: pd.DataFrame, disease_column: str = "disease"
    ) -> Tuple[pd.DataFrame, Dict[str, Any], AnalysisStep]:
        """
        Standardize disease terminology in metadata.

        Args:
            metadata: DataFrame with disease information
            disease_column: Column name containing disease labels

        Returns:
            Tuple of (standardized_metadata, statistics, provenance_ir)
        """
        if disease_column not in metadata.columns:
            raise ValueError(f"Disease column '{disease_column}' not found in metadata")

        result = metadata.copy()
        original_col = f"{disease_column}_original"
        result[original_col] = result[disease_column]

        # Track standardization statistics
        mapping_stats = {
            "exact_matches": 0,
            "contains_matches": 0,
            "reverse_contains_matches": 0,
            "token_matches": 0,
            "unmapped": 0,
        }
        unique_mappings = {}

        # Standardize each term
        standardized_values = []
        for value in result[disease_column]:
            std_value, match_type = self._fuzzy_match(str(value), self.DISEASE_MAPPINGS)
            standardized_values.append(std_value)

            # Track statistics (unmapped doesn't have _matches suffix)
            key = match_type if match_type == "unmapped" else f"{match_type}_matches"
            mapping_stats[key] += 1
            if str(value).lower() != std_value.lower():
                unique_mappings[str(value)] = std_value

        result[disease_column] = standardized_values

        # Build statistics
        total = len(result)
        standardization_rate = (
            (total - mapping_stats["unmapped"]) / total * 100 if total > 0 else 0.0
        )

        stats = {
            "total_samples": total,
            "standardization_rate": standardization_rate,
            "mapping_stats": mapping_stats,
            "unique_mappings": unique_mappings,
            "disease_distribution": result[disease_column].value_counts().to_dict(),
            "original_column": original_col,
        }

        # Create provenance IR
        ir = self._create_standardization_ir(
            disease_column=disease_column,
            original_column=original_col,
            mapping_stats=mapping_stats,
            unique_mappings=unique_mappings,
        )

        return result, stats, ir

    def filter_by_sample_type(
        self,
        metadata: pd.DataFrame,
        sample_types: List[str],
        sample_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], AnalysisStep]:
        """
        Filter samples by sample type (fecal vs gut tissue).

        Args:
            metadata: DataFrame with sample information
            sample_types: List of sample types to keep (e.g., ["fecal"])
            sample_columns: Columns to check for sample type info
                           (if None, checks common columns)

        Returns:
            Tuple of (filtered_metadata, statistics, provenance_ir)
        """
        # Auto-detect sample columns if not specified
        if sample_columns is None:
            common_cols = [
                "sample_type",
                "tissue",
                "body_site",
                "sample_source",
                "biosample_type",
                "material",
            ]
            sample_columns = [col for col in common_cols if col in metadata.columns]

            if not sample_columns:
                raise ValueError(
                    "No sample type columns found. Specify sample_columns explicitly."
                )

        result = metadata.copy()
        result["sample_type_filter_match"] = False
        result["matched_column"] = ""
        result["matched_value"] = ""

        # Normalize requested sample types
        normalized_types = [t.lower() for t in sample_types]

        # Check each sample against all columns
        for idx, row in result.iterrows():
            for col in sample_columns:
                if pd.isna(row[col]):
                    continue

                # Try to match this value to requested sample types
                matched_type, _ = self._fuzzy_match(
                    str(row[col]),
                    {
                        t: self.SAMPLE_TYPE_MAPPINGS.get(t, [t])
                        for t in normalized_types
                    },
                )

                if matched_type in normalized_types:
                    result.at[idx, "sample_type_filter_match"] = True
                    result.at[idx, "matched_column"] = col
                    result.at[idx, "matched_value"] = str(row[col])
                    break

        # Filter to matching samples
        filtered = result[result["sample_type_filter_match"]].copy()
        filtered = filtered.drop(columns=["sample_type_filter_match"])

        # Build statistics
        stats = {
            "original_samples": len(metadata),
            "filtered_samples": len(filtered),
            "retention_rate": (
                len(filtered) / len(metadata) * 100 if len(metadata) > 0 else 0
            ),
            "sample_types_requested": sample_types,
            "columns_checked": sample_columns,
            "matches_by_column": {
                col: filtered[filtered["matched_column"] == col].shape[0]
                for col in sample_columns
            },
        }

        # Create provenance IR
        ir = self._create_filter_ir(
            sample_types=sample_types,
            sample_columns=sample_columns,
            original_samples=len(metadata),
            filtered_samples=len(filtered),
        )

        return filtered, stats, ir

    def _fuzzy_match(
        self, value: str, mappings: Dict[str, List[str]]
    ) -> Tuple[str, str]:
        """
        Conservative matching strategy with exclusion pattern checking.

        Args:
            value: Input string to match
            mappings: Dictionary of {standard_term: [variants]}

        Returns:
            Tuple of (matched_standard_term, match_type)

        Match types:
            - exact: Direct match in variant list
            - contains: Value contains exactly one category's variant as whole words
            - unmapped: No match, blank, negated or ambiguous (returns original value)
        """
        # A negative/unknown value ("healthy control: no", "crc_status=0") is not
        # a mapping, whatever keyword it contains. In a ';'-separated list only
        # the positive segments are considered.
        raw_lower = " ".join(str(value).lower().split())
        remaining = self._positive_segments(raw_lower)
        if not remaining:
            return value, "unmapped"
        # A negative/unknown segment was dropped: the rest may name a disease but
        # can never establish health ("healthy; ibd: unknown").
        dropped = remaining != raw_lower.strip(" ;|")
        normalized_value = self._normalize_term(remaining)

        # Check exclusion patterns first (prevent false positives)
        # Uses allow-list approach for ambiguous terms (Gemini-recommended)
        for category, config in self.DISEASE_EXCLUSION_CONFIG.items():
            # 1. Check universally blocked substrings first (fast path)
            for phrase in config.get("block_substrings", []):
                normalized_phrase = self._normalize_term(phrase)
                if normalized_phrase in normalized_value:
                    return value, "unmapped"

            # 2. Handle ambiguous terms with allow-list validation
            for term, term_config in config.get("ambiguous_terms", {}).items():
                if term in normalized_value:
                    # Block if exact match to standalone ambiguous term
                    if normalized_value in term_config.get("block_exact", []):
                        return value, "unmapped"

                    # Check if compound phrase is in allow-list
                    is_allowed = False
                    for allowed_phrase in term_config.get("allowed_phrases", []):
                        normalized_allowed = self._normalize_term(allowed_phrase)
                        if normalized_allowed in normalized_value:
                            is_allowed = True
                            break

                    if not is_allowed:
                        # Contains ambiguous term in non-allowed context
                        # (e.g., "IBS control", "normal flora")
                        return value, "unmapped"

        # Level 1: Exact match
        for standard, variants in mappings.items():
            if dropped and standard == "healthy":
                continue
            if normalized_value in [self._normalize_term(v) for v in variants]:
                return standard, "exact"

        # Blank or missing-value text is never a disease or a health status
        if normalized_value in self._MISSING_VALUES:
            return value, "unmapped"

        # Level 2: a variant appears in the value as a whole phrase (not inside a
        # longer word such as "unhealthy" or "CD4") and is not negated.
        # Several different categories in one value is ambiguous -> unmapped.
        # Health status is only taken from a whole-value (exact) match, but
        # health wording next to a disease makes the value ambiguous.
        hits = set()
        for standard, variants in mappings.items():
            for variant in variants:
                if self._contains_phrase(
                    normalized_value, self._normalize_term(variant)
                ):
                    hits.add(standard)
                    break
        if len(hits) == 1 and "healthy" not in hits:
            return hits.pop(), "contains"

        # Partial (reverse-contains) and single-token overlaps are not accepted:
        # "IBD" or an empty string must never resolve to a category.
        return value, "unmapped"

    _NEG_TOKEN = (
        r"(?:0|no|false|n|negative|neg|absent|none|unknown|unk|n/?a|na|"
        r"not applicable|missing|null|nan)"
    )
    _AFTER_SEPARATOR = re.compile(r"[:=]\s*" + _NEG_TOKEN + r"(?![^\W_])")
    _PAREN_NEGATIVE = re.compile(r"\(\s*(?:negative|neg|no|absent|none|false|0)\s*\)")
    _BARE_LEADING = re.compile(r"^(?:negative|neg|no|not|without|absent)\b")

    @classmethod
    def _positive_segments(cls, text_lower: str) -> str:
        """Drop negated/unknown ';'- or '|'-separated segments of a value."""
        kept = []
        for segment in re.split(r"[;|]", text_lower):
            segment = segment.strip()
            if not segment:
                continue
            if (
                cls._BARE_LEADING.search(segment)
                or cls._AFTER_SEPARATOR.search(segment)
                or cls._PAREN_NEGATIVE.search(segment)
            ):
                continue
            kept.append(segment)
        return "; ".join(kept)

    _MISSING_VALUES = frozenset({"", "nan", "none", "null", "na", "n a"})
    _NEGATION_BEFORE = re.compile(
        r"\b(?:no|not|non|without|negative for|free of|absence of|absent)\s*$"
    )
    _NEGATIVE_AFTER = re.compile(r"^\s*(?:0|no|false|n|negative|neg|absent|none)\b")

    def _contains_phrase(self, normalized_value: str, normalized_phrase: str) -> bool:
        """True if the phrase occurs as whole words and is not negated."""
        if not normalized_phrase:
            return False
        pattern = r"(?<![^\W_])" + re.escape(normalized_phrase) + r"(?![^\W_])"
        for found in re.finditer(pattern, normalized_value):
            before = normalized_value[: found.start()]
            after = normalized_value[found.end() :]
            if self._NEGATION_BEFORE.search(before) or self._NEGATIVE_AFTER.search(
                after
            ):
                continue
            return True
        return False

    def _normalize_term(self, term: str) -> str:
        """
        Normalize a term for matching.

        - Lowercase
        - Remove punctuation
        - Collapse whitespace
        """
        term = str(term).lower()
        term = re.sub(r"[^\w\s]", " ", term)  # Replace punctuation with space
        term = re.sub(r"\s+", " ", term).strip()  # Collapse whitespace
        return term

    def _tokenize(self, term: str) -> List[str]:
        """Split term into tokens (words)."""
        return [t for t in term.split() if len(t) > 1]  # Ignore single-char tokens

    def _create_standardization_ir(
        self,
        disease_column: str,
        original_column: str,
        mapping_stats: Dict[str, int],
        unique_mappings: Dict[str, str],
    ) -> AnalysisStep:
        """Create provenance IR for disease standardization."""
        return AnalysisStep(
            operation="disease_standardization",
            tool_name="standardize_disease_terms",
            description=f"Standardize disease terminology in '{disease_column}' column using 5-level fuzzy matching",
            library="custom",
            imports=[],
            code_template="""
# Disease standardization (manual implementation)
metadata['{{ original_column }}'] = metadata['{{ disease_column }}']

# Map disease terms using standardized categories
disease_mappings = {{ disease_mappings }}
for idx, value in metadata['{{ disease_column }}'].items():
    # Apply fuzzy matching logic (5 levels)
    standardized = fuzzy_match(value, disease_mappings)
    metadata.at[idx, '{{ disease_column }}'] = standardized
""",
            parameters={
                "disease_column": disease_column,
                "original_column": original_column,
                "disease_mappings": self.DISEASE_MAPPINGS,
                "mapping_stats": mapping_stats,
                "unique_mappings": unique_mappings,
            },
            parameter_schema={
                "disease_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="disease",
                    required=True,
                    description="Column containing disease labels",
                ),
                "original_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=False,
                    default_value="disease_original",
                    required=False,
                    description="Backup column with original values",
                ),
            },
            input_entities=["metadata"],
            output_entities=["standardized_metadata"],
            execution_context={
                "standardization_rate": (
                    100
                    - (mapping_stats["unmapped"] / sum(mapping_stats.values()) * 100)
                    if sum(mapping_stats.values()) > 0
                    else 0.0
                ),
                "total_mappings": len(unique_mappings),
            },
        )

    def _create_filter_ir(
        self,
        sample_types: List[str],
        sample_columns: List[str],
        original_samples: int,
        filtered_samples: int,
    ) -> AnalysisStep:
        """Create provenance IR for sample type filtering."""
        return AnalysisStep(
            operation="sample_type_filter",
            tool_name="filter_by_sample_type",
            description=f"Filter samples by type: {', '.join(sample_types)}",
            library="custom",
            imports=[],
            code_template="""
# Sample type filtering
sample_columns = {{ sample_columns }}
sample_types = {{ sample_types }}

# Check each column for matching sample types
metadata['keep'] = False
for col in sample_columns:
    if col in metadata.columns:
        metadata['keep'] |= metadata[col].str.lower().str.contains('|'.join(sample_types), na=False)

filtered_metadata = metadata[metadata['keep']].drop(columns=['keep'])
""",
            parameters={
                "sample_types": sample_types,
                "sample_columns": sample_columns,
                "retention_rate": (
                    (filtered_samples / original_samples * 100)
                    if original_samples > 0
                    else 0
                ),
            },
            parameter_schema={
                "sample_types": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=["fecal"],
                    required=True,
                    description="Sample types to retain",
                ),
                "sample_columns": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=[],
                    required=False,
                    description="Columns to check for sample type information",
                ),
            },
            input_entities=["metadata"],
            output_entities=["filtered_metadata"],
            execution_context={
                "original_samples": original_samples,
                "filtered_samples": filtered_samples,
                "retention_rate": (
                    (filtered_samples / original_samples * 100)
                    if original_samples > 0
                    else 0
                ),
            },
        )
