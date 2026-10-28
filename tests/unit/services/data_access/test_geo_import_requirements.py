"""Opt-in GEO requirements and structured candidate evidence use synthetic input."""

from dataclasses import FrozenInstanceError

import pytest

from lobster.services.data_access.geo.constants import GEOServiceError


@pytest.mark.parametrize("value", [None, ""])
def test_empty_requirement_is_inactive(value):
    from lobster.services.data_access.geo.import_requirements import ImportRequirement

    requirement = ImportRequirement.parse(value)
    assert requirement.representation == "any"
    assert requirement.level == "any"
    assert requirement.active is False


def test_gene_counts_requirement_is_explicit_and_immutable():
    from lobster.services.data_access.geo.import_requirements import ImportRequirement

    requirement = ImportRequirement.parse("gene_counts")
    assert requirement.representation == "counts"
    assert requirement.level == "gene"
    assert requirement.active is True
    with pytest.raises(FrozenInstanceError):
        requirement.level = "any"


@pytest.mark.parametrize(
    "value", ["counts", "GENE_COUNTS", "gene_counts ", 0, False, []]
)
def test_unknown_requirements_fail_fast(value):
    from lobster.services.data_access.geo.import_requirements import ImportRequirement

    with pytest.raises(ValueError, match="Unknown data_requirement"):
        ImportRequirement.parse(value)


def test_evidence_distinguishes_unchecked_values_from_negative_findings():
    from lobster.services.data_access.geo.import_requirements import CandidateEvidence

    evidence = CandidateEvidence(file_name="counts.gene.tsv", level="gene")
    assert evidence.verdict == "not_checked"
    assert evidence.complete_parse is None
    assert evidence.sample_rule is None
    assert evidence.whole_numbers is None
    assert evidence.equal_sample_totals is None
    evidence.complete_parse = False
    evidence.verdict = "rejected"
    evidence.reason = "incomplete parse"
    assert evidence.complete_parse is False


@pytest.mark.parametrize(
    "problem",
    ["no_candidate", "ambiguous", "identity", "too_many_candidates", "value_fidelity"],
)
def test_structured_refusal_keeps_candidate_facts_and_per_file_reasons(problem):
    from lobster.services.data_access.geo.import_requirements import (
        CandidateEvidence,
        ImportRequirement,
        ImportRequirementNotMet,
    )

    requirement = ImportRequirement.parse("gene_counts")
    candidates = [
        CandidateEvidence(
            file_name="scaled.gene.tsv",
            level="gene",
            verdict="rejected",
            reason="values are not whole-number counts",
        ),
        CandidateEvidence(
            file_name="counts.transcript.tsv",
            level="transcript",
            verdict="rejected",
            reason="transcript-level file",
        ),
    ]
    error = ImportRequirementNotMet(
        "synthetic_series", requirement, candidates, problem=problem
    )
    assert isinstance(error, GEOServiceError)
    assert error.geo_id == "synthetic_series"
    assert error.requirement == requirement
    assert error.candidates == candidates
    assert error.problem == problem
    message = error.to_message()
    assert str(error) == message
    assert f"Requirement not met for synthetic_series ({problem})" in message
    assert "gene-level counts" in message
    assert "scaled.gene.tsv: rejected, values are not whole-number counts" in message
    assert "counts.transcript.tsv: rejected, transcript-level file" in message
    assert "Do not retry this entry unchanged" in message


def test_ambiguity_message_lists_passing_files_without_picking_one():
    from lobster.services.data_access.geo.import_requirements import (
        CandidateEvidence,
        ImportRequirement,
        ImportRequirementNotMet,
    )

    candidates = [
        CandidateEvidence(
            file_name=name,
            level="gene",
            verdict="accepted",
            reason="whole-number counts (numeric check)",
        )
        for name in ["counts_a.gene.tsv", "counts_b.gene.tsv"]
    ]
    error = ImportRequirementNotMet(
        "synthetic_series",
        ImportRequirement.parse("gene_counts"),
        candidates,
        problem="ambiguous",
    )
    assert "counts_a.gene.tsv: accepted" in str(error)
    assert "counts_b.gene.tsv: accepted" in str(error)
    assert "numeric check" in str(error)
    assert "raw counts verified" not in str(error)
