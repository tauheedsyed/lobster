"""Safety contracts: unknown, ambiguous, blank or negative values never become healthy.

These tests call the real export, metadata-helper, keyword and standardization
functions with synthetic strings; only the agent's LLM construction is replaced.
They cover every reported false-healthy input, partial-hit confidence, and the
valid healthy/control mappings that must keep working.
"""

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

from lobster.agents.metadata_assistant import config as enrichment_config
from lobster.agents.metadata_assistant.metadata_assistant import metadata_assistant
from lobster.core.schemas.export_schemas import harmonize_disease_field
from lobster.services.metadata.disease_ontology_service import DiseaseOntologyService
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

UNRESOLVED_FLAGS = [
    "control",
    False,
    "No",
    "false",
    0,
    None,
    np.nan,
    "unknown",
    "not applicable",
    "",
    " ",
    "present",
    "unhealthy",
    "abnormal",
]
KNOWN_IDS = {"crc", "cd", "uc", "healthy"}


@pytest.fixture
def ontology(monkeypatch):
    monkeypatch.setattr(DiseaseOntologyService, "_instance", None)
    service = DiseaseOntologyService.get_instance()
    assert service._vector_service is None, "This is an offline keyword-path test"
    return service


@pytest.fixture(params=[True, False], ids=["premium", "public-fallback"])
def standardizer(request, ontology, monkeypatch):
    module = importlib.import_module(
        "lobster.services.metadata.disease_standardization_service"
    )
    monkeypatch.setattr(module, "HAS_ONTOLOGY_SERVICE", request.param)
    return module.DiseaseStandardizationService()


@pytest.fixture
def metadata_extractor(tmp_path, ontology):
    dm = RecordingDataManager(workspace_path=tmp_path)
    tools = factory_tools(
        metadata_assistant,
        "lobster.agents.metadata_assistant.metadata_assistant",
        dm,
    )
    # Access the live closure used by the actual filtering tool. No AST-extracted
    # reimplementation: this is exactly the factory-created helper under test.
    closure = inspect.getclosurevars(tools["filter_samples_by"].func)
    return closure.nonlocals["_extract_disease_from_raw_fields"]


@pytest.mark.parametrize("value", UNRESOLVED_FLAGS)
@pytest.mark.parametrize("path", ["export", "metadata-helper"])
def test_negative_or_unresolved_flags_do_not_prove_healthy(
    value, path, metadata_extractor
):
    sample = {"crohns_disease": value, "colitis_disease": value}
    if path == "export":
        result, _ = harmonize_disease_field([sample])
        disease = result[0].get("disease")
    else:
        frame = pd.DataFrame([sample])
        metadata_extractor(frame)
        disease = frame.loc[0, "disease"]
    assert disease != "healthy", f"{path}: {value!r} became healthy"


@pytest.mark.parametrize("reverse_columns", [False, True])
@pytest.mark.parametrize("path", ["export", "metadata-helper"])
def test_positive_disease_flag_is_not_overridden_by_control(
    reverse_columns, path, metadata_extractor
):
    pairs = [("crohns_disease", "Yes"), ("colitis_disease", "control")]
    sample = dict(reversed(pairs) if reverse_columns else pairs)
    if path == "export":
        result, _ = harmonize_disease_field([sample])
        disease = result[0].get("disease")
    else:
        frame = pd.DataFrame([sample])
        metadata_extractor(frame)
        disease = frame.loc[0, "disease"]
    assert disease != "healthy", "A competing control flag is not proof of health"


@pytest.mark.parametrize("path", ["export", "metadata-helper"])
def test_known_positive_flag_control(path, metadata_extractor):
    sample = {"crohns_disease": "Yes", "colitis_disease": "No"}
    if path == "export":
        result, _ = harmonize_disease_field([sample])
        disease = result[0]["disease"]
    else:
        frame = pd.DataFrame([sample])
        metadata_extractor(frame)
        disease = frame.loc[0, "disease"]
    assert disease == "cd"


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
@pytest.mark.parametrize(
    "term",
    [
        "unhealthy",
        "negative control",
        "control",
        "unknown",
        "present",
        "abnormal",
        "IBD",
        "",
        " ",
        None,
    ],
)
def test_column_rescan_does_not_invent_healthy(term, premium, ontology, monkeypatch):
    # Exercise the supported premium/fallback branch, not a fake matcher.
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR001", "condition": term}]
    enrichment_config.phase1_column_rescan(samples)
    assert samples[0].get("disease") != "healthy", f"{term!r} became healthy"


@pytest.mark.parametrize(
    "term",
    ["unhealthy", "abnormal", "present", "IBD", "", " ", "unknown", "control"],
)
def test_keyword_matching_unknown_is_not_healthy(term, ontology):
    matches = ontology.match_disease(term)
    assert all(match.disease_id != "healthy" for match in matches)


@pytest.mark.parametrize("term", ["CD45+ T cells", "glucose", "colon mucosa"])
def test_keyword_matching_unrelated_text_is_not_a_disease(term, ontology):
    assert not ontology.match_disease(
        term
    ), f"Unrelated term {term!r} resolved to a disease"


@pytest.mark.parametrize(
    "term",
    [
        "IBD",
        "",
        " ",
        None,
        np.nan,
        "unknown",
        "present",
        "unhealthy",
        "abnormal",
        "control",
        "negative control",
        "completely unknown disease",
    ],
)
def test_standardization_ambiguous_or_missing_is_not_resolved(term, standardizer):
    frame = pd.DataFrame({"disease": [term]})
    standardized, _, _ = standardizer.standardize_disease_terms(frame)
    disease = standardized.loc[0, "disease"]
    assert disease not in KNOWN_IDS, f"{term!r} silently resolved to {disease!r}"


@pytest.mark.parametrize(
    "term,expected",
    [
        ("colorectal cancer", "crc"),
        ("Crohn's disease", "cd"),
        ("ulcerative colitis", "uc"),
        ("healthy", "healthy"),
    ],
)
def test_keyword_and_standardization_positive_controls(
    term, expected, ontology, standardizer
):
    matches = ontology.match_disease(term)
    assert {match.disease_id for match in matches} == {expected}
    standardized, _, _ = standardizer.standardize_disease_terms(
        pd.DataFrame({"disease": [term]})
    )
    assert standardized.loc[0, "disease"] == expected


@pytest.mark.parametrize(
    "term",
    [
        "Crohn's disease and healthy controls",
        "colorectal cancer and healthy volunteers",
    ],
)
def test_mixed_cohort_text_is_not_a_healthy_sample(term, ontology):
    matches = ontology.match_disease(term)
    assert all(
        match.disease_id != "healthy" for match in matches
    ), "Mixed/ambiguous cohort text must not label a sample healthy"


@pytest.mark.parametrize("path", ["export", "metadata-helper"])
def test_explicit_healthy_label_is_preserved(path, metadata_extractor):
    sample = {"condition": "healthy"}
    if path == "export":
        result, _ = harmonize_disease_field([sample])
        disease = result[0]["disease"]
    else:
        frame = pd.DataFrame([sample])
        metadata_extractor(frame)
        disease = frame.loc[0, "disease"]
    assert disease == "healthy"


# --- Whole-token matching: substrings of other words are not diseases ---------

TRAPS = [
    ("CD4+ T cells", None),
    ("CD4", None),
    ("pancreatic ductal adenocarcinoma", None),
    ("ductal", None),
    ("crc=0", None),
    ("CRC: no", None),
    ("healthy control: no", None),
    ("healthy_volunteer: false", None),
    ("healthy: unknown", None),
    ("adjacent healthy tissue", None),
    ("disease-free survival", None),
    ("disease-free survival 24 months", None),
    ("crc_status=0", None),
    ("crc_flag: 0", None),
    ("negative crc", None),
    ("crc (negative)", None),
    ("normal control", None),
    ("normal", None),
    ("crc - negative", None),
    ("no crohn's disease", None),
    ("not healthy", None),
    ("non-healthy", None),
]


@pytest.mark.parametrize("term,_expected", TRAPS)
def test_keyword_matching_ignores_substrings_and_negated_keywords(
    term, _expected, ontology
):
    assert ontology.match_disease(term) == []


@pytest.mark.parametrize("term,_expected", TRAPS)
@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
def test_column_rescan_ignores_substrings_and_negated_keywords(
    term, _expected, premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR002", "condition": term}]
    enrichment_config.phase1_column_rescan(samples)
    assert not samples[0].get("disease"), f"{term!r} -> {samples[0].get('disease')!r}"


@pytest.mark.parametrize("term,_expected", TRAPS)
def test_standardization_ignores_substrings_and_negated_keywords(
    term, _expected, standardizer
):
    standardized, _, _ = standardizer.standardize_disease_terms(
        pd.DataFrame({"disease": [term]})
    )
    assert standardized.loc[0, "disease"] not in KNOWN_IDS


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
def test_column_rescan_text_naming_several_diseases_is_left_unresolved(
    premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [
        {
            "run_accession": "SRR003",
            "condition": "Crohn's disease and healthy controls",
        }
    ]
    enrichment_config.phase1_column_rescan(samples)
    assert not samples[0].get("disease")


# --- Confidence: only a whole-value match is 1.0 ------------------------------


def test_partial_keyword_hit_is_not_full_confidence(ontology):
    exact = ontology.match_disease("colorectal cancer")
    partial = ontology.match_disease("biopsy from colorectal cancer patient")
    assert [m.disease_id for m in exact] == ["crc"] and exact[0].confidence == 1.0
    assert [m.disease_id for m in partial] == ["crc"]
    assert partial[0].confidence < 1.0
    assert partial[0].match_type != "exact_keyword"


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
def test_column_rescan_partial_hit_is_a_candidate_not_applied(
    premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR004", "tissue": "colorectal tissue"}]
    enriched, _ = enrichment_config.phase1_column_rescan(samples)
    assert enriched == 0
    assert "disease" not in samples[0]
    assert samples[0]["disease_candidate"] == "crc"
    assert samples[0]["disease_candidate_confidence"] < 1.0


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
@pytest.mark.parametrize(
    "term", ["adjacent healthy tissue", "disease-free survival", "healthy control: no"]
)
def test_column_rescan_never_applies_or_proposes_health_from_text(
    term, premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR006", "condition": term}]
    enrichment_config.phase1_column_rescan(samples)
    assert "disease" not in samples[0]
    assert samples[0].get("disease_candidate") != "healthy"


def test_multi_flag_text_keeps_only_the_positive_segment(ontology):
    matches = ontology.match_disease("crc=1;uc=0;cd=0")
    assert [m.disease_id for m in matches] == ["crc"]
    assert matches[0].confidence < 1.0


def test_whole_value_match_is_applied_by_column_rescan_premium(ontology, monkeypatch):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", True)
    samples = [{"run_accession": "SRR007", "dx": "Crohn's disease"}]
    enrichment_config.phase1_column_rescan(samples)
    assert samples[0]["disease"] == "cd"
    assert samples[0]["disease_confidence"] == 1.0


# --- Valid healthy / control mappings keep working -----------------------------

HEALTHY_PHRASES = [
    "healthy",
    "healthy control",
    "healthy volunteer",
    "healthy donor",
    "healthy subject",
    "non-ibd",
    "non-diseased",
    "disease-free",
]


@pytest.mark.parametrize("term", HEALTHY_PHRASES)
def test_explicit_healthy_phrases_still_match_keywords(term, ontology):
    matches = ontology.match_disease(term)
    assert [m.disease_id for m in matches] == ["healthy"]
    assert matches[0].confidence == 1.0


@pytest.mark.parametrize("term", HEALTHY_PHRASES)
def test_explicit_healthy_phrases_still_standardize(term, standardizer):
    standardized, _, _ = standardizer.standardize_disease_terms(
        pd.DataFrame({"disease": [term]})
    )
    assert standardized.loc[0, "disease"] == "healthy"


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
@pytest.mark.parametrize("term", ["healthy", "healthy control", "non-ibd"])
def test_explicit_healthy_phrases_still_enrich_in_column_rescan(
    term, premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR005", "condition": term}]
    enrichment_config.phase1_column_rescan(samples)
    assert samples[0]["disease"] == "healthy"
    assert samples[0]["disease_confidence"] == 1.0


@pytest.mark.parametrize("path", ["export", "metadata-helper"])
def test_positive_flag_with_negative_siblings_is_the_disease(path, metadata_extractor):
    sample = {
        "crohns_disease": "Yes",
        "colitis_disease": "No",
        "colorectal_disease": "false",
    }
    if path == "export":
        result, _ = harmonize_disease_field([sample])
        disease = result[0]["disease"]
    else:
        frame = pd.DataFrame([sample])
        metadata_extractor(frame)
        disease = frame.loc[0, "disease"]
    assert disease == "cd"


# --- Re-running the rescan must not promote its own candidates ----------------


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
def test_column_rescan_run_twice_does_not_promote_candidate(
    premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR008", "tissue": "colorectal tissue"}]
    enrichment_config.phase1_column_rescan(samples)
    assert samples[0]["disease_candidate"] == "crc" and "disease" not in samples[0]
    first = dict(samples[0])

    enriched, _ = enrichment_config.phase1_column_rescan(samples)

    assert enriched == 0
    assert "disease" not in samples[0]
    assert samples[0] == first  # idempotent: no field changed or was added


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
@pytest.mark.parametrize(
    "field",
    [
        "disease_candidate",
        "disease_original",
        "disease_source",
        "disease_match_type",
        "disease_evidence",
    ],
)
def test_column_rescan_ignores_its_own_output_fields(
    field, premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR009", field: "crc"}]
    enriched, _ = enrichment_config.phase1_column_rescan(samples)
    assert enriched == 0
    assert "disease" not in samples[0]


@pytest.mark.parametrize("premium", [True, False], ids=["premium", "public-fallback"])
def test_exported_and_reimported_candidate_is_not_promoted(
    premium, ontology, monkeypatch
):
    monkeypatch.setattr(enrichment_config, "HAS_ONTOLOGY_SERVICE", premium)
    samples = [{"run_accession": "SRR010", "tissue": "colorectal tissue"}]
    enrichment_config.phase1_column_rescan(samples)
    assert samples[0]["disease_candidate"] == "crc"

    exported, _ = harmonize_disease_field(samples)
    reimported = [
        {k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
        for row in exported
    ]
    for row in reimported:
        row.pop("disease", None)  # the export left disease unresolved
    enriched, _ = enrichment_config.phase1_column_rescan(reimported)

    assert enriched == 0
    assert not reimported[0].get("disease")
    assert reimported[0]["disease_candidate"] == "crc"
