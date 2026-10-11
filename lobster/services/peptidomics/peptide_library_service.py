"""
Peptide library operations: import, filtering, variant generation and
known-activity annotation.

Stateless service returning 3-tuples (AnnData, Dict, AnalysisStep). Each IR
replays the same service call (or, for database lookups, the recorded result)
on a notebook's ``adata``, so an exported notebook reproduces the library.
"""

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import anndata as ad
import pandas as pd

from lobster.core.provenance.analysis_ir import AnalysisStep

logger = logging.getLogger(__name__)

# The 20 standard proteinogenic amino acids.
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

_SEQUENCE_COLUMNS = ("sequence", "seq", "peptide", "peptide_sequence")
_NAME_COLUMNS = ("name", "id", "accession", "peptide_id")
_SERVICE_IMPORT = (
    "from lobster.services.peptidomics.peptide_library_service import (\n"
    "    PeptideLibraryService,\n"
    ")\n\n"
)


def _service_call_ir(
    operation: str,
    tool_name: str,
    description: str,
    call: str,
    parameters: Dict[str, Any],
    output_entities: Optional[List[str]] = None,
    input_entities: Optional[List[str]] = None,
    execution_context: Optional[Dict[str, Any]] = None,
) -> AnalysisStep:
    """IR whose template calls a PeptideLibraryService method with ``parameters``."""
    args = "".join(f"    {name}={{{{ {name} | repr }}}},\n" for name in parameters)
    return AnalysisStep(
        operation=operation,
        tool_name=tool_name,
        description=description,
        library="lobster",
        code_template=_SERVICE_IMPORT + call.replace("{args}", args),
        imports=[],
        parameters=dict(parameters),
        parameter_schema={},
        input_entities=input_entities if input_entities is not None else ["adata"],
        output_entities=output_entities or ["adata"],
        execution_context=execution_context or {},
    )


class PeptideLibraryService:
    """Build, filter and annotate peptide libraries stored as AnnData (peptides in obs)."""

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(path: Path, content: str) -> str:
        """Resolve ``"auto"`` to fasta, tsv or csv from extension and first line."""
        ext = path.suffix.lower()
        if ext in (".fa", ".fasta", ".faa"):
            return "fasta"
        if ext == ".tsv" or "\t" in content.split("\n")[0]:
            return "tsv"
        return "csv"

    def import_sequences(
        self, file_path: str, format: str = "auto"
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """Load peptides from FASTA/CSV/TSV, keeping valid sequences of length 2-100.

        Raises:
            FileNotFoundError: If ``file_path`` does not exist.
            ValueError: If the file holds no sequences.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text()
        resolved_format = (
            self.detect_format(path, content) if format == "auto" else format
        )

        sequences: List[str] = []
        names: List[str] = []
        if resolved_format == "fasta":
            current_name = ""
            current_seq: List[str] = []
            for line in content.strip().split("\n"):
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].strip().split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))
                names.append(current_name)
        else:
            sep = "\t" if resolved_format == "tsv" else ","
            df = pd.read_csv(path, sep=sep)
            seq_col = next(
                (c for c in df.columns if c.lower() in _SEQUENCE_COLUMNS), df.columns[0]
            )
            sequences = df[seq_col].astype(str).tolist()
            name_col = next((c for c in df.columns if c.lower() in _NAME_COLUMNS), None)
            names = (
                df[name_col].astype(str).tolist()
                if name_col
                else [f"pep_{i}" for i in range(len(sequences))]
            )

        if not sequences:
            raise ValueError("No sequences found in file")

        valid_seqs: List[str] = []
        valid_names: List[str] = []
        invalid_count = 0
        for name, seq in zip(names, sequences):
            clean = "".join(c for c in seq.upper() if c.isalpha())
            if (
                clean
                and all(aa in STANDARD_AA for aa in clean)
                and 2 <= len(clean) <= 100
            ):
                valid_seqs.append(clean)
                valid_names.append(name)
            else:
                invalid_count += 1

        obs = pd.DataFrame(
            {
                "sequence": valid_seqs,
                "length": [len(s) for s in valid_seqs],
                "source_file": str(path.name),
            },
            index=valid_names,
        )
        adata = ad.AnnData(obs=obs)

        stats = {
            "format": resolved_format,
            "n_sequences": len(valid_seqs),
            "n_invalid": invalid_count,
            "mean_length": float(obs["length"].mean()) if len(obs) else 0.0,
        }
        ir = _service_call_ir(
            operation="peptidomics.import_sequences",
            tool_name="import_peptide_sequences",
            description=f"Import {len(valid_seqs)} peptide sequences from {path.name}",
            call=(
                "adata, import_stats, _ = PeptideLibraryService().import_sequences(\n"
                "{args})"
            ),
            parameters={"file_path": str(path), "format": resolved_format},
            input_entities=[],
        )
        return adata, stats, ir

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_peptides(
        self,
        adata: ad.AnnData,
        min_length: int = 0,
        max_length: int = 100,
        min_charge: float = -100,
        max_charge: float = 100,
        min_hydrophobicity: float = -10,
        max_hydrophobicity: float = 10,
        min_activity_score: float = 0.0,
        activity_column: str = "",
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """Keep peptides inside the length, charge, GRAVY and activity bounds.

        A bound applies only when its column is present in ``adata.obs``.
        """
        obs = adata.obs
        mask = pd.Series(True, index=obs.index)

        if "length" in obs.columns or "peptide_length" in obs.columns:
            len_col = "peptide_length" if "peptide_length" in obs.columns else "length"
            lengths = obs[len_col].astype(float)
            mask = mask & (lengths >= min_length) & (lengths <= max_length)
        elif "sequence" in obs.columns:
            lengths = obs["sequence"].str.len()
            mask = mask & (lengths >= min_length) & (lengths <= max_length)

        if "peptide_charge" in obs.columns:
            charges = obs["peptide_charge"].astype(float)
            mask = mask & (charges >= min_charge) & (charges <= max_charge)

        if "peptide_gravy" in obs.columns:
            gravy = obs["peptide_gravy"].astype(float)
            mask = mask & (gravy >= min_hydrophobicity) & (gravy <= max_hydrophobicity)

        if activity_column and activity_column in obs.columns:
            scores = obs[activity_column].astype(float)
            mask = mask & (scores >= min_activity_score)

        adata_filtered = adata[mask.values].copy()
        params = {
            "min_length": min_length,
            "max_length": max_length,
            "min_charge": min_charge,
            "max_charge": max_charge,
            "min_hydrophobicity": min_hydrophobicity,
            "max_hydrophobicity": max_hydrophobicity,
            "min_activity_score": min_activity_score,
            "activity_column": activity_column,
        }
        stats = {
            "n_before": int(adata.n_obs),
            "n_after": int(adata_filtered.n_obs),
            "n_removed": int(adata.n_obs - adata_filtered.n_obs),
        }
        ir = _service_call_ir(
            operation="peptidomics.filter_peptides",
            tool_name="filter_peptides",
            description=(
                f"Filter peptides by property bounds: "
                f"{stats['n_before']} -> {stats['n_after']}"
            ),
            call=(
                "adata, filter_stats, _ = PeptideLibraryService().filter_peptides(\n"
                "    adata,\n"
                "{args})"
            ),
            parameters=params,
        )
        return adata_filtered, stats, ir

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def generate_variants(
        self,
        adata: ad.AnnData,
        variant_type: str = "alanine_scan",
        target_peptide: str = "",
        random_seed: Optional[int] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """Generate alanine-scan, point-mutation or scrambled variants of one peptide.

        ``random_seed`` only affects ``"scrambled"``. When it is None a seed is
        drawn and recorded in the IR, so the scrambled set can be regenerated.

        Raises:
            ValueError: For an unknown variant type, a missing ``sequence``
                column, or when no variants result.
        """
        if "sequence" not in adata.obs.columns:
            raise ValueError("No 'sequence' column in modality")

        if target_peptide and target_peptide in adata.obs_names:
            seq = adata.obs.loc[target_peptide, "sequence"]
            parent_name = target_peptide
        else:
            seq = adata.obs["sequence"].iloc[0]
            parent_name = adata.obs_names[0]

        variants: List[Dict[str, Any]] = []
        if variant_type == "alanine_scan":
            for i, aa in enumerate(seq):
                if aa != "A":
                    variants.append(
                        {
                            "sequence": seq[:i] + "A" + seq[i + 1 :],
                            "variant_type": "alanine_scan",
                            "mutation": f"{aa}{i + 1}A",
                            "position": i + 1,
                            "parent": parent_name,
                        }
                    )
        elif variant_type == "point_mutations":
            for i, aa in enumerate(seq):
                for new_aa in sorted(STANDARD_AA):
                    if new_aa != aa:
                        variants.append(
                            {
                                "sequence": seq[:i] + new_aa + seq[i + 1 :],
                                "variant_type": "point_mutation",
                                "mutation": f"{aa}{i + 1}{new_aa}",
                                "position": i + 1,
                                "parent": parent_name,
                            }
                        )
        elif variant_type == "scrambled":
            if random_seed is None:
                random_seed = random.SystemRandom().randrange(2**32)
            rng = random.Random(random_seed)  # nosec B311 - not security relevant
            for j in range(10):
                shuffled = list(seq)
                rng.shuffle(shuffled)
                variants.append(
                    {
                        "sequence": "".join(shuffled),
                        "variant_type": "scrambled",
                        "mutation": f"scramble_{j + 1}",
                        "position": 0,
                        "parent": parent_name,
                    }
                )
        else:
            raise ValueError(
                f"Unknown variant_type '{variant_type}'. "
                "Use: alanine_scan, point_mutations, scrambled"
            )

        if not variants:
            raise ValueError("No variants generated")

        var_df = pd.DataFrame(variants)
        var_df["length"] = var_df["sequence"].str.len()
        var_df.index = [f"var_{i}" for i in range(len(var_df))]
        adata_out = ad.AnnData(obs=var_df)

        stats = {
            "variant_type": variant_type,
            "parent": str(parent_name),
            "parent_sequence": str(seq),
            "n_variants": len(variants),
            "random_seed": random_seed,
        }
        params: Dict[str, Any] = {
            "variant_type": variant_type,
            "target_peptide": str(parent_name),
        }
        if variant_type == "scrambled":
            params["random_seed"] = random_seed
        ir = _service_call_ir(
            operation="peptidomics.generate_variants",
            tool_name="generate_peptide_variants",
            description=f"{variant_type}: {len(variants)} variants of {parent_name}",
            call=(
                "adata, variant_stats, _ = PeptideLibraryService().generate_variants(\n"
                "    adata,\n"
                "{args})"
            ),
            parameters=params,
        )
        return adata_out, stats, ir

    # ------------------------------------------------------------------
    # Known-activity annotation
    # ------------------------------------------------------------------

    def annotate_known_activity(
        self,
        adata: ad.AnnData,
        providers: Iterable[Callable[[], Any]],
        database: str,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """Look each sequence up in peptide databases and record known activities.

        Live database results can change over time, so the IR does not repeat
        the lookups: it records the sequence -> activities hits and re-applies
        them, which reproduces this exact annotation offline.
        """
        if "sequence" not in adata.obs.columns:
            raise ValueError("No 'sequence' column found in modality")

        provider_classes = list(providers)
        sequences = adata.obs["sequence"].astype(str).tolist()
        annotations: List[str] = []
        for seq in sequences:
            found_activities: List[str] = []
            for provider_cls in provider_classes:
                try:
                    provider = provider_cls()
                    for r in provider.search_sequences(seq[:20], max_results=3):
                        if r.sequence == seq and r.activities:
                            found_activities.extend(r.activities)
                except Exception as e:
                    logger.warning(f"Annotation lookup failed: {e}")
            annotations.append(
                ", ".join(sorted(set(found_activities)))
                if found_activities
                else "unknown"
            )

        adata_out = adata.copy()
        adata_out.obs["known_activities"] = annotations

        known = {
            seq: label
            for seq, label in zip(sequences, annotations)
            if label != "unknown"
        }
        stats = {
            "database": database,
            "n_queried": len(sequences),
            "n_annotated": sum(1 for a in annotations if a != "unknown"),
        }
        ir = AnalysisStep(
            operation="peptidomics.annotate_known_activity",
            tool_name="annotate_peptide_activity",
            description=(
                f"Known activities from {database}: "
                f"{stats['n_annotated']}/{stats['n_queried']} peptides matched"
            ),
            library="pandas",
            code_template=(
                "# Activities found in the {{ database }} lookup (sequence -> activities).\n"
                "# Live databases change, so the recorded hits are re-applied here.\n"
                "known_activities = {{ known_activities | repr }}\n"
                "adata.obs['known_activities'] = (\n"
                "    adata.obs['sequence'].astype(str).map(known_activities).fillna('unknown')\n"
                ")"
            ),
            imports=[],
            parameters={"database": database, "known_activities": known},
            parameter_schema={},
            input_entities=["adata"],
            output_entities=["adata"],
        )
        return adata_out, stats, ir
