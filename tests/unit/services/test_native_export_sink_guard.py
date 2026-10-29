"""Native writers never write into the managed figure folder (P1-2 lane B).

Each sink is driven through its real tool (or real service, where the service is the
writer) with the real classifier and a revision saved through the real store. External
boundaries only are faked: HTTP (``requests``), ``subprocess`` in the PyMOL service
module, the docling import, and the injected chemistry services. A refusal returns the
fixed message (tools) or raises the typed error (services), and leaves the whole
workspace, the saved revision, the modalities, the metadata store and the tool log
unchanged. Ordinary outputs keep their names, files and replies.
"""

from __future__ import annotations

import os
import subprocess
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_guard import MANAGED_EXPORT_MESSAGE
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools
from tests.unit.export_guard_fixtures import alias, saved_revision, workspace_tree


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A DataManager workspace with one saved revision; tool-log calls recorded."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    dm = DataManagerV2(workspace_path=tmp_path / "ws")
    rev = saved_revision(dm.workspace_path)
    logged = []
    monkeypatch.setattr(dm, "log_tool_usage", lambda *a, **k: logged.append(k))
    return types.SimpleNamespace(
        dm=dm, ws=dm.workspace_path, rev=rev, logged=logged, cwd=cwd, tmp=tmp_path
    )


def _into_revision(s, name: str) -> str:
    """A spelling, relative to the workspace, of ``name`` inside the saved revision."""
    return f"plots/{s.rev.directory.parent.name}/{s.rev.directory.name}/{name}"


def _assert_nothing_changed(s, before):
    assert workspace_tree(s.ws) == before
    assert s.rev.unchanged()
    assert s.logged == []
    assert list(s.cwd.iterdir()) == []


# ======================================================================= pathway (F1)


class _IndraResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "go": [
                {
                    "curie": "GO:0000001",
                    "name": "a process",
                    "p_value": 0.001,
                    "q_value": 0.01,
                    "gene_overlap": ["g1"],
                    "pathway_size": 10,
                }
            ]
        }


@pytest.fixture
def indra(monkeypatch):
    import requests

    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _IndraResponse()

    monkeypatch.setattr(requests, "post", post)
    return calls


def _selected_adata():
    adata = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["s1", "s2"]),
        var=pd.DataFrame({"stability_selected": [True, False]}, index=["g1", "g2"]),
    )
    return adata


def _pathway_tool(dm):
    from lobster.agents.machine_learning.shared_tools import (
        create_feature_selection_tools,
    )

    tools = create_feature_selection_tools(dm, agent_name="feature_selection_expert")
    return {t.name: t for t in tools}["enrich_pathways_for_selected_features"]


def test_pathway_csv_lands_in_the_session_workspace(session, indra, monkeypatch):
    s = session
    ambient = s.tmp / "ambient"
    monkeypatch.setenv("LOBSTER_WORKSPACE", str(ambient))
    s.dm.modalities["m"] = _selected_adata()
    response = _pathway_tool(s.dm).invoke({"modality_name": "m"})
    assert "Pathway enrichment complete!" in response
    assert (s.ws / "m_pathway_enrichment.csv").is_file()
    assert not ambient.exists()  # the ambient workspace is neither used nor created
    assert s.dm.modalities["m"].uns["pathway_enrichment"]["csv_file"] == str(
        s.ws / "m_pathway_enrichment.csv"
    )


def test_pathway_refuses_a_managed_csv_before_the_request(session, indra):
    s = session
    name = _into_revision(s, "m")
    adata = _selected_adata()
    s.dm.modalities[name] = adata
    before = workspace_tree(s.ws)
    response = _pathway_tool(s.dm).invoke({"modality_name": name})
    assert response == MANAGED_EXPORT_MESSAGE
    assert indra == []  # no request was sent
    assert "pathway_enrichment" not in adata.uns
    assert s.dm.modalities[name] is adata
    _assert_nothing_changed(s, before)


def test_new_api_pathway_service_refuses_before_the_request(session, indra):
    """New-API check: ``workspace_path`` is the keyword added by the fix."""
    from lobster.services.analysis.pathway_enrichment_bridge_service import (
        PathwayEnrichmentBridgeService,
    )

    s = session
    adata = _selected_adata()
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        PathwayEnrichmentBridgeService(timeout=1).enrich_selected_features(
            adata, _into_revision(s, "m"), workspace_path=s.ws
        )
    assert indra == [] and "pathway_enrichment" not in adata.uns
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_regression_pathway_timeout_text_is_unchanged(session, monkeypatch):
    import requests

    s = session

    def post(url, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", post)
    s.dm.modalities["m"] = _selected_adata()
    response = _pathway_tool(s.dm).invoke({"modality_name": "m"})
    assert response == (
        "Error: INDRA API timeout - try again later or reduce gene list size"
    )
    assert not (s.ws / "m_pathway_enrichment.csv").exists()


# ===================================================================== PyMOL (P1-3b)

PYMOL_MODULE = "lobster.agents.protein_structure_visualization_expert"
PDB = "1ABC"


class _FakeSubprocess:
    """``subprocess`` as seen by the PyMOL service module: every call recorded."""

    TimeoutExpired = subprocess.TimeoutExpired
    CalledProcessError = subprocess.CalledProcessError
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.calls = []
        self.on_probe = None

    def run(self, args, **kwargs):
        self.calls.append(list(args))
        if args[:3] == ["pymol", "-c", "-Q"] and self.on_probe is not None:
            self.on_probe()
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def Popen(self, args, **kwargs):
        self.calls.append(list(args))
        return types.SimpleNamespace(pid=4242, poll=lambda: None)


@pytest.fixture
def pymol(session, monkeypatch):
    from lobster.services.visualization import pymol_visualization_service as svc

    fake = _FakeSubprocess()
    monkeypatch.setattr(svc, "subprocess", fake)
    structure = session.ws / "structures" / f"{PDB}.cif"
    structure.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text("data_1ABC\n")
    session.dm.modalities[f"structure_{PDB}"] = {"file_path": str(structure)}
    return fake


def _pymol_tool(dm):
    from lobster.agents.protein_structure_visualization_expert import (
        protein_structure_visualization_expert,
    )

    tools = factory_tools(protein_structure_visualization_expert, PYMOL_MODULE, dm)
    return tools["visualize_with_pymol"]


def _launches(fake):
    """Calls other than the installation probe ``pymol -c -Q``."""
    return [c for c in fake.calls if c[:3] != ["pymol", "-c", "-Q"]]


def _viz(dm):
    return [k for k in dm.modalities if str(k).startswith("viz_")]


def _visualizations(s) -> Path:
    return s.ws / "structures" / "visualizations"


def _pymol_case(s, case):
    """Arguments (and set-up) for one managed PyMOL destination."""
    up = "../../../" + _into_revision(s, "")
    if case == "style":
        return {"style": "a/" + up + "s"}
    if case == "color_by":
        return {"color_by": "c/" + up + "d"}
    if case == "visualizations-alias":
        alias(_visualizations(s), s.rev.directory)
        return {}
    _visualizations(s).mkdir(parents=True)
    name = f"{PDB}_cartoon_chain" + ("_commands.pml" if case == "script" else ".png")
    alias(_visualizations(s) / name, s.rev.directory / sorted(s.rev.files)[0])
    return {}


@pytest.mark.parametrize(
    "case", ["style", "color_by", "visualizations-alias", "script", "image"]
)
def test_pymol_refuses_a_managed_destination_before_anything(session, pymol, case):
    s = session
    args = {"pdb_id": PDB, "mode": "batch", **_pymol_case(s, case)}
    before = workspace_tree(s.ws)
    response = _pymol_tool(s.dm).invoke(args)
    assert response == MANAGED_EXPORT_MESSAGE
    assert pymol.calls == []
    assert _viz(s.dm) == []
    _assert_nothing_changed(s, before)


def test_pymol_refuses_an_image_retargeted_before_launch(session, pymol):
    s = session
    image = _visualizations(s) / f"{PDB}_cartoon_chain.png"
    pymol.on_probe = lambda: os.path.lexists(image) or alias(
        image, s.rev.directory / "new.png"
    )
    response = _pymol_tool(s.dm).invoke({"pdb_id": PDB, "mode": "batch"})
    assert response == MANAGED_EXPORT_MESSAGE
    assert pymol.calls and _launches(pymol) == []  # probed, never launched
    assert not (s.rev.directory / "new.png").exists()
    assert s.rev.unchanged() and _viz(s.dm) == [] and s.logged == []


def test_pymol_ordinary_names_and_script_are_unchanged(session, pymol):
    s = session
    response = _pymol_tool(s.dm).invoke({"pdb_id": PDB, "execute": False})
    script = _visualizations(s) / f"{PDB}_cartoon_chain_commands.pml"
    assert script.is_file()
    assert str(script) in response
    assert _viz(s.dm) == [f"viz_{PDB}_cartoon_chain"]
    assert [k["tool_name"] for k in s.logged] == ["visualize_with_pymol"]
    assert _launches(pymol) == []  # the reply only probes the installation
    assert s.rev.unchanged()


def test_pymol_ordinary_batch_run_launches_once(session, pymol):
    s = session
    response = _pymol_tool(s.dm).invoke({"pdb_id": PDB, "mode": "batch"})
    script = _visualizations(s) / f"{PDB}_cartoon_chain_commands.pml"
    assert script.is_file() and str(script) in response
    assert _launches(pymol) == [["pymol", "-c", str(script)]]


def test_regression_pymol_traversal_outside_the_managed_folder_is_not_refused(
    session, pymol
):
    s = session
    _visualizations(s).mkdir(parents=True)
    (_visualizations(s) / f"{PDB}_a").mkdir()
    response = _pymol_tool(s.dm).invoke(
        {"pdb_id": PDB, "style": "a/../../elsewhere", "execute": False}
    )
    assert response != MANAGED_EXPORT_MESSAGE
    assert (s.ws / "structures" / "elsewhere_chain_commands.pml").is_file()
    assert s.rev.unchanged()


# ============================================================ workspace content (F4)

ENTRY = "entry_x"
SRA_SAMPLES = [
    {
        "run_accession": "SRR1",
        "biosample": "SAMN1",
        "bioproject": "PRJNA1",
        "library_strategy": "AMPLICON",
        "time_point": "d1",
        "crohns_disease": "Yes",
    },
    {
        "run_accession": "SRR2",
        "biosample": "SAMN2",
        "bioproject": "PRJNA1",
        "library_strategy": "AMPLICON",
        "timepoint": "d2",
        "crohns_disease": "No",
    },
]
CONTENT_DIRS = ("literature", "metadata", "exports")


@pytest.fixture
def content(session):
    """The session with no content folder yet (removed after DataManager set-up)."""
    import shutil

    for name in CONTENT_DIRS:
        shutil.rmtree(session.ws / name, ignore_errors=True)
        assert not (session.ws / name).exists()
    return session


def _write(s, **args):
    import asyncio

    from lobster.tools.workspace_tool import create_write_to_workspace_tool

    args = {"identifier": ENTRY, "add_timestamp": False, **args}
    return asyncio.run(create_write_to_workspace_tool(s.dm).ainvoke(args))


def _revision_file(s) -> Path:
    return s.rev.directory / sorted(s.rev.files)[0]


def _assert_write_refused(s, before, store_before, response):
    assert response == MANAGED_EXPORT_MESSAGE
    assert s.dm.metadata_store[ENTRY] == store_before
    _assert_nothing_changed(s, before)


def test_workspace_json_into_an_aliased_metadata_folder_is_refused(content):
    s = content
    s.dm.metadata_store[ENTRY] = {"value": 1}
    alias(s.ws / "metadata", s.rev.directory)
    before, store_before = workspace_tree(s.ws), dict(s.dm.metadata_store[ENTRY])
    response = _write(s, workspace="metadata")
    _assert_write_refused(s, before, store_before, response)
    assert not (s.ws / "literature").exists() and not (s.ws / "exports").exists()


@pytest.mark.parametrize("which", ["simple", "rich", "strict", "provenance"])
def test_workspace_csv_refuses_every_sibling_before_the_first_write(content, which):
    s = content
    s.dm.metadata_store[ENTRY] = {"samples": [dict(x) for x in SRA_SAMPLES]}
    name = {
        "simple": f"{ENTRY}.csv",
        "rich": f"{ENTRY}_rich.csv",
        "strict": f"{ENTRY}_strict.csv",
        "provenance": f"{ENTRY}_harmonization_log.tsv",
    }[which]
    alias(s.ws / "exports" / name, _revision_file(s))
    before = workspace_tree(s.ws)
    store_before = dict(s.dm.metadata_store[ENTRY])
    response = _write(
        s,
        workspace="exports",
        output_format="csv",
        export_mode="simple" if which == "simple" else "rich",
    )
    _assert_write_refused(s, before, store_before, response)
    assert sorted(p.name for p in (s.ws / "exports").iterdir()) == [name]
    assert not (s.ws / "literature").exists() and not (s.ws / "metadata").exists()


def test_workspace_rich_into_an_aliased_exports_folder_is_refused(content):
    s = content
    s.dm.metadata_store[ENTRY] = {"samples": [dict(x) for x in SRA_SAMPLES]}
    alias(s.ws / "exports", s.rev.directory)
    before = workspace_tree(s.ws)
    store_before = dict(s.dm.metadata_store[ENTRY])
    response = _write(s, workspace="exports", output_format="csv", export_mode="rich")
    _assert_write_refused(s, before, store_before, response)
    assert not (s.ws / "literature").exists() and not (s.ws / "metadata").exists()


def test_write_content_raises_the_typed_refusal_without_a_path(content):
    from lobster.services.data_access.workspace_content_service import (
        ContentType,
        MetadataContent,
        WorkspaceContentService,
    )

    s = content
    alias(s.ws / "metadata", s.rev.directory)
    service = WorkspaceContentService(s.dm)
    before = workspace_tree(s.ws)
    item = MetadataContent(
        identifier=ENTRY,
        content_type="unknown",
        data={"value": 1},
        description="d",
        source="test",
        cached_at="2026-01-01T00:00:00",
    )
    with pytest.raises(ManagedPathError) as refused:
        service.write_content(item, ContentType.METADATA)
    assert str(s.ws) not in str(refused.value) and ENTRY not in str(refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_workspace_ordinary_json_is_unchanged(content):
    s = content
    s.dm.metadata_store[ENTRY] = {"value": 1}
    response = _write(s, workspace="metadata")
    target = s.ws / "metadata" / f"{ENTRY}.json"
    assert target.is_file(), response
    assert f"**Location**: {target}\n" in response
    for name in ("literature", "data", "metadata"):  # the same folders as at base
        assert (s.ws / name).is_dir()
    assert not (s.ws / "exports").exists()


@pytest.mark.parametrize("mode", ["simple", "rich"])
def test_workspace_ordinary_csv_is_unchanged(content, mode):
    s = content
    s.dm.metadata_store[ENTRY] = {"samples": [dict(x) for x in SRA_SAMPLES]}
    response = _write(s, workspace="exports", output_format="csv", export_mode=mode)
    exports = s.ws / "exports"
    expected = (
        [f"{ENTRY}.csv"]
        if mode == "simple"
        else [
            f"{ENTRY}_harmonization_log.tsv",
            f"{ENTRY}_rich.csv",
            f"{ENTRY}_strict.csv",
        ]
    )
    assert sorted(p.name for p in exports.iterdir()) == expected, response
    if mode == "simple":
        assert "Content Cached Successfully" in response
    else:
        assert f"2. **Strict Export**: {ENTRY}_strict.csv" in response
    primary = f"{ENTRY}.csv" if mode == "simple" else f"{ENTRY}_rich.csv"
    assert s.dm.metadata_store[ENTRY]["csv_export_path"] == str(exports / primary)
    for name in ("literature", "data", "metadata", "exports"):
        assert (s.ws / name).is_dir()
    assert s.rev.unchanged()


# ===================================================================== docling (F4)

DOC_SOURCE = "https://example.org/paper.pdf"
DOCLING_LOGGER = "lobster.services.data_access.docling_service"


@pytest.fixture
def fake_docling(monkeypatch):
    """``docling`` at the import boundary: exactly the names the service imports."""
    import sys

    class _Options:
        pass

    class _Converter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    names = [
        "docling",
        "docling.datamodel",
        "docling.datamodel.base_models",
        "docling.datamodel.pipeline_options",
        "docling.document_converter",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    modules["docling.datamodel.base_models"].InputFormat = types.SimpleNamespace(
        PDF="pdf", HTML="html"
    )
    modules["docling.datamodel.pipeline_options"].PdfPipelineOptions = _Options
    converter = modules["docling.document_converter"]
    converter.ConversionStatus = types.SimpleNamespace(SUCCESS="success")
    converter.DocumentConverter = _Converter
    converter.PdfFormatOption = lambda **kwargs: kwargs
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class _Doc:
    def model_dump(self):
        return {"name": "parsed"}


def _docling(dm):
    from lobster.services.data_access.docling_service import DoclingService

    return DoclingService(data_manager=dm)


def test_docling_cache_folder_in_a_revision_disables_only_the_cache(
    session, fake_docling
):
    s = session
    alias(s.dm.cache_dir / "literature_cache", s.rev.directory)
    before = workspace_tree(s.ws)
    service = _docling(s.dm)
    assert service.is_available()  # the converter is unaffected
    assert service._get_cached_document(DOC_SOURCE) is None
    assert service.purge_failed_cache(DOC_SOURCE) is False
    service._cache_document(DOC_SOURCE, _Doc())
    service._cache_failure(DOC_SOURCE, "HTTP_403", "denied")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


@pytest.mark.parametrize("writer", ["document", "failure"])
def test_docling_cache_file_alias_is_not_written(session, fake_docling, writer, caplog):
    import logging

    s = session
    service = _docling(s.dm)
    alias(service._cache_path_for(DOC_SOURCE), _revision_file(s))
    before = workspace_tree(s.ws)
    with caplog.at_level(logging.DEBUG, logger=DOCLING_LOGGER):
        if writer == "document":
            service._cache_document(DOC_SOURCE, _Doc())
        else:
            service._cache_failure(DOC_SOURCE, "HTTP_403", "denied")
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    skipped = [
        r for r in caplog.records if r.getMessage().startswith("Document cache write")
    ]
    assert len(skipped) == 1 and str(s.ws) not in skipped[0].getMessage()


def test_new_api_docling_cache_flag_reports_the_disabled_cache(session, fake_docling):
    """New-API check: ``_cache_enabled`` is the attribute added by the fix."""
    s = session
    alias(s.dm.cache_dir / "literature_cache", s.rev.directory)
    assert _docling(s.dm)._cache_enabled is False
    assert _docling(DataManagerV2(workspace_path=s.tmp / "other"))._cache_enabled


def test_docling_ordinary_cache_writes_are_unchanged(session, fake_docling):
    import json

    s = session
    service = _docling(s.dm)
    service._cache_document(DOC_SOURCE, _Doc())
    assert json.loads(service._cache_path_for(DOC_SOURCE).read_text()) == {
        "name": "parsed"
    }
    other = "https://example.org/other.pdf"
    service._cache_failure(other, "HTTP_403", "denied")
    failure = json.loads(service._cache_path_for(other).read_text())
    assert failure["cache_type"] == "failure" and failure["error"] == "HTTP_403"


def test_regression_docling_unrelated_cache_errors_keep_their_text(
    session, fake_docling, caplog
):
    import logging

    s = session
    service = _docling(s.dm)
    if os.geteuid() == 0:
        pytest.skip("privileged: a 0500 folder stays writable")
    mode = service.cache_dir.stat().st_mode
    service.cache_dir.chmod(0o500)
    try:
        with caplog.at_level(logging.DEBUG, logger=DOCLING_LOGGER):
            service._cache_document(DOC_SOURCE, _Doc())
            service._cache_failure(DOC_SOURCE, "HTTP_403", "denied")
    finally:
        service.cache_dir.chmod(mode)
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Failed to cache document (non-fatal): ") for m in messages)
    assert any(m.startswith("Failed to cache failure (non-fatal): ") for m in messages)


# ===================================================================== peptide (F4)


def _peptide_tool(dm):
    from lobster.agents.proteomics.peptide_tools import create_peptide_tools

    return {t.name: t for t in create_peptide_tools(dm)}["export_peptide_report"]


def _peptides():
    return ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame({"sequence": ["ACD", "KLM"]}, index=["p1", "p2"]),
        var=pd.DataFrame(index=["f1"]),
    )


def test_peptide_report_alias_is_refused(session):
    s = session
    s.dm.modalities["pep"] = _peptides()
    alias(s.dm.exports_dir / "pep_report.csv", _revision_file(s))
    before = workspace_tree(s.ws)
    response = _peptide_tool(s.dm).invoke(
        {"modality_name": "pep", "output_format": "csv"}
    )
    assert response == MANAGED_EXPORT_MESSAGE
    _assert_nothing_changed(s, before)


def test_peptide_ordinary_report_is_unchanged(session):
    s = session
    s.dm.modalities["pep"] = _peptides()
    response = _peptide_tool(s.dm).invoke(
        {"modality_name": "pep", "output_format": "csv"}
    )
    target = s.dm.exports_dir / "pep_report.csv"
    assert response == f"STATUS: complete\nformat=csv\nfile={target}\nrows=2"
    assert target.is_file()
    assert [k["tool_name"] for k in s.logged] == ["export_peptide_report"]


# ============================================================ cheminformatics (F4)

SMILES = "CCO"
MOL_BLOCK = "mol block\nM  END\n"


class _Chemistry:
    """The injected computation services (rdkit is not installed)."""

    def prepare_molecule_3d(self, smiles, n_conformers=1):
        stats = {
            "mol_block": MOL_BLOCK,
            "num_atoms": 9,
            "num_heavy_atoms": 3,
            "energy_kcal_mol": 1.5,
            "n_conformers_generated": 1,
        }
        return None, stats, None

    def fingerprint_similarity(self, smiles, fingerprint="morgan", radius=2):
        n = len(smiles)
        stats = {
            "similarity_matrix": [[1.0] * n for _ in range(n)],
            "compound_ids": [f"c{i}" for i in range(n)],
            "n_compounds": n,
            "mean_similarity": 1.0,
            "max_similarity": 1.0,
            "min_similarity": 1.0,
        }
        return None, stats, None


def _chem_tools(dm):
    from lobster.agents.drug_discovery.cheminformatics_tools import (
        create_cheminformatics_tools,
    )

    chemistry = _Chemistry()
    tools = create_cheminformatics_tools(dm, chemistry, None, None, chemistry)
    return {t.name: t for t in tools}


def _sdf(s) -> Path:
    import hashlib

    digest = hashlib.md5(SMILES.encode(), usedforsecurity=False).hexdigest()[:8]
    return s.ws / "data" / f"molecule_{digest}.sdf"


def test_chem_sdf_alias_is_skipped_with_the_fixed_note(session):
    from lobster.core.figures.export_guard import MANAGED_EXPORT_REASON

    s = session
    alias(_sdf(s), _revision_file(s))
    before = workspace_tree(s.ws)
    response = _chem_tools(s.dm)["prepare_molecule_3d"].invoke({"smiles": SMILES})
    note = f"\n  Structure file: not written ({MANAGED_EXPORT_REASON})"
    assert response.endswith(note) and "3D structure prepared" in response
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_chem_ordinary_sdf_is_unchanged(session):
    s = session
    response = _chem_tools(s.dm)["prepare_molecule_3d"].invoke({"smiles": SMILES})
    assert response.endswith(f"\n  Structure file: data/{_sdf(s).name}")
    assert _sdf(s).read_text() == MOL_BLOCK


def test_chem_similarity_into_an_aliased_metadata_folder_is_refused(session):
    s = session
    alias(s.ws / "metadata", s.rev.directory)
    before = workspace_tree(s.ws)
    response = _chem_tools(s.dm)["fingerprint_similarity"].invoke(
        {"smiles_list": "CCO,CCN"}
    )
    assert response == MANAGED_EXPORT_MESSAGE
    # Folders are created only once the result file is allowed: no effect at all.
    _assert_nothing_changed(s, before)


def test_chem_ordinary_similarity_is_unchanged(session):
    s = session
    response = _chem_tools(s.dm)["fingerprint_similarity"].invoke(
        {"smiles_list": "CCO,CCN"}
    )
    assert "Result: metadata/fingerprint_similarity_" in response
    assert len(list((s.ws / "metadata").glob("fingerprint_similarity_*.json"))) == 1
    assert len(s.logged) == 1


# ============================================== B2: recursive folder creation (SEC-03)


def test_b2_recursive_mkdir_premise_creates_every_lexical_prefix(tmp_path):
    """The OS behaviour the folder check exists for (plain folders, no guard)."""
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "NEW" / ".." / ".." / "c").mkdir(parents=True)
    assert (tmp_path / "a" / "b" / "NEW").is_dir() and (tmp_path / "a" / "c").is_dir()


def _through_revision(s, *tail):
    """A spelling that passes through a NEW folder in the revision, then leaves it."""
    return s.rev.directory.joinpath("NEW", "..", "..", "..", "..", *tail)


def test_b2_export_dirs_refuses_a_managed_intermediate_folder(session):
    from lobster.core.figures.export_dirs import assert_dirs_allowed

    s = session
    hostile = _through_revision(s, "exports", "out")
    assert os.path.normpath(hostile) == str(s.ws / "exports" / "out")
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        assert_dirs_allowed(s.ws, hostile)
    assert refused.value.details == {} and str(s.ws) not in str(refused.value)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_b2_export_dirs_allows_ordinary_and_existing_folders(session):
    from lobster.core.figures.export_dirs import assert_dirs_allowed

    s = session
    before = workspace_tree(s.ws)
    assert_dirs_allowed(s.ws, s.ws / "exports" / "a" / "b")  # new nested folder
    assert_dirs_allowed(s.ws, s.ws / "exports")  # existing final folder
    assert_dirs_allowed(s.ws, s.tmp / "outside" / "x")  # outside the workspace
    assert workspace_tree(s.ws) == before  # the check creates nothing
    alias(s.ws / "exports" / "lnk", s.rev.directory)
    with pytest.raises(ManagedPathError):  # an existing final folder that is a link
        assert_dirs_allowed(s.ws, s.ws / "exports" / "lnk")


def test_b2_export_dirs_checks_a_folder_as_a_container_not_as_a_flat_file(session):
    from lobster.core.figures.export_dirs import assert_dirs_allowed
    from lobster.core.figures.export_guard import assert_export_allowed

    s = session
    new = s.ws / "plots" / "new_folder"
    assert_export_allowed(s.ws, new)  # allowed as a legacy flat FILE name...
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):  # ...but not as a new folder for files
        assert_dirs_allowed(s.ws, new)
    assert_dirs_allowed(s.ws, s.ws)  # files written at the workspace root: allowed
    assert_dirs_allowed(s.ws, s.ws / "plots")  # legacy flat files: as the file guard
    with pytest.raises(ManagedPathError):  # an existing figure folder
        assert_dirs_allowed(s.ws, s.rev.directory.parent)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


# ===================================================== B2: PyMOL command script (F2)


def _injections(s):
    rogue = s.rev.directory / "rogue.png"
    command = f"\npng {rogue}\n#"
    return {
        "highlight_color": {
            "highlight_residues": "15",
            "highlight_color": "red" + command,
        },
        "highlight_style": {
            "highlight_residues": "15",
            "highlight_style": "sticks" + command,
        },
        "highlight_residues": {"highlight_residues": "15" + command},
        "highlight_groups": {"highlight_groups": f"15|red{command}|sticks"},
        "argument_boundary": {
            "highlight_residues": "15",
            "highlight_color": "red, all",
        },
        "separator": {"highlight_residues": "15; png x.png"},
    }


@pytest.mark.parametrize(
    "field",
    [
        "highlight_color",
        "highlight_style",
        "highlight_residues",
        "highlight_groups",
        "argument_boundary",
        "separator",
    ],
)
def test_b2_pymol_refuses_an_injected_command_before_anything(session, pymol, field):
    s = session
    before = workspace_tree(s.ws)
    response = _pymol_tool(s.dm).invoke(
        {"pdb_id": PDB, "mode": "batch", **_injections(s)[field]}
    )
    assert response.startswith("Error creating visualization: "), response
    assert str(s.rev.directory) not in response  # the value is not echoed
    assert not (s.rev.directory / "rogue.png").exists()
    assert pymol.calls == []  # not probed, never launched
    assert _viz(s.dm) == []
    _assert_nothing_changed(s, before)


@pytest.mark.parametrize("case", ["background", "width", "structure-file"])
def test_b2_pymol_service_refuses_injection_in_other_fields(session, pymol, case):
    from lobster.services.visualization.pymol_visualization_service import (
        PyMOLVisualizationError,
        PyMOLVisualizationService,
    )

    s = session
    structure = s.ws / "structures" / f"{PDB}.cif"
    args = {}
    if case == "background":
        args["background"] = f"white\npng {s.rev.directory / 'rogue.png'}\n#"
    elif case == "width":
        args["width"] = f"1\npng {s.rev.directory / 'rogue.png'}\n#"
    else:
        structure = structure.parent / "x\nsave rogue.pdb\n#.cif"
        structure.write_text("data_x\n")
    before = workspace_tree(s.ws)
    with pytest.raises(PyMOLVisualizationError):
        PyMOLVisualizationService().visualize_structure(
            structure_file=structure, mode="batch", workspace_path=s.ws, **args
        )
    assert pymol.calls == []
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_b2_pymol_ordinary_highlight_script_lines_are_unchanged(session, pymol):
    s = session
    response = _pymol_tool(s.dm).invoke(
        {
            "pdb_id": PDB,
            "execute": False,
            "highlight_groups": "15,42|red|sticks;A:100-120|0xff0000|surface",
        }
    )
    script = _visualizations(s) / f"{PDB}_cartoon_chain_commands.pml"
    lines = script.read_text().splitlines()
    for line in [
        f"load {s.ws / 'structures' / (PDB + '.cif')}, protein1",
        "bg_color white",
        "select highlight_group_1, resi 15+42",
        "show sticks, highlight_group_1",
        "color red, highlight_group_1",
        "select highlight_group_2, (chain A and resi 100-120)",
        "show surface, highlight_group_2",
        "color 0xff0000, highlight_group_2",
    ]:
        assert line in lines, response
    assert s.rev.unchanged()


def test_b2_pymol_refuses_an_output_folder_passing_through_a_revision(session, pymol):
    from lobster.services.visualization.pymol_visualization_service import (
        PyMOLVisualizationService,
    )

    s = session
    image = _through_revision(s, "exports", "img.png")
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        PyMOLVisualizationService().visualize_structure(
            structure_file=s.ws / "structures" / f"{PDB}.cif",
            mode="batch",
            output_image=image,
            execute_commands=False,
            workspace_path=s.ws,
        )
    assert not (s.rev.directory / "NEW").exists()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_b2_pymol_refusal_logs_no_caller_path(session, pymol, caplog):
    import logging

    s = session
    up = "a/../../../" + _into_revision(s, "") + "s"
    with caplog.at_level(logging.DEBUG):
        response = _pymol_tool(s.dm).invoke(
            {"pdb_id": PDB, "mode": "batch", "style": up}
        )
    assert response == MANAGED_EXPORT_MESSAGE
    assert not [r for r in caplog.records if "plots" in r.getMessage()]


# ======================================================= B2: docling cache folder (SEC-03/06)


def test_b2_docling_cache_folder_through_a_revision_disables_only_the_cache(
    session, fake_docling, caplog
):
    import logging

    from lobster.services.data_access.docling_service import DoclingService

    s = session
    folder = _through_revision(s, "docling_cache")
    before = workspace_tree(s.ws)
    with caplog.at_level(logging.DEBUG, logger=DOCLING_LOGGER):
        service = DoclingService(data_manager=s.dm, cache_dir=folder)
    assert service.is_available()
    service._cache_document(DOC_SOURCE, _Doc())
    assert not (s.rev.directory / "NEW").exists()
    assert workspace_tree(s.ws) == before and s.rev.unchanged()
    assert not [r for r in caplog.records if "docling_cache" in r.getMessage()]


# ================================================ B2: legacy queue deletion lock (F3)


def test_b2_queue_deletion_refuses_a_lock_file_alias_into_a_revision(session):
    import json

    from lobster.core.schemas.publication_queue import PublicationQueueEntry
    from lobster.tools.workspace_tool import create_delete_from_workspace_tool

    s = session
    entry_id = "pub_queue_legacy"
    s.dm.publication_queue.add_entry(
        PublicationQueueEntry(entry_id=entry_id, pmid="12345678", title="t")
    )
    legacy = s.ws / "publication_queue.jsonl"
    legacy.write_text(json.dumps({"entry_id": entry_id}) + "\n")
    lock = s.ws / ".publication_queue.jsonl.lock"
    alias(lock, s.rev.directory / "new.lock")  # dangling, inside the revision
    tool = create_delete_from_workspace_tool(s.dm)
    args = {"identifier": entry_id, "workspace": "publication_queue"}
    tool.invoke({**args, "confirmation": False})  # the preview first (its set-up)
    s.logged.clear()
    before, text = workspace_tree(s.ws), legacy.read_text()
    response = tool.invoke({**args, "confirmation": True})
    assert response == MANAGED_EXPORT_MESSAGE
    assert not os.path.exists(s.rev.directory / "new.lock")
    assert legacy.read_text() == text
    _assert_nothing_changed(s, before)


def test_b2_ordinary_legacy_queue_deletion_is_unchanged(session):
    import json

    from lobster.core.schemas.publication_queue import PublicationQueueEntry
    from lobster.tools.workspace_tool import create_delete_from_workspace_tool

    s = session
    entry_id = "pub_queue_legacy"
    s.dm.publication_queue.add_entry(
        PublicationQueueEntry(entry_id=entry_id, pmid="12345678", title="t")
    )
    legacy = s.ws / "publication_queue.jsonl"
    legacy.write_text(json.dumps({"entry_id": entry_id}) + "\n")
    response = create_delete_from_workspace_tool(s.dm).invoke(
        {"identifier": entry_id, "workspace": "publication_queue", "confirmation": True}
    )
    assert f"✓ Deleted queue entry: {entry_id}" in response
    assert legacy.read_text() == ""
    assert (s.ws / ".publication_queue.jsonl.lock").is_file()


# ============================================== B2: content CSV write boundary (F5)


def test_b2_content_csv_writer_checks_again_at_the_write(content):
    from lobster.services.data_access.workspace_content_service import (
        MetadataContent,
        WorkspaceContentService,
    )

    s = content
    service = WorkspaceContentService(s.dm)
    target = s.ws / "metadata" / "late.csv"
    alias(target, _revision_file(s))  # after the caller's own check (L1-shaped)
    item = MetadataContent(
        identifier="late",
        content_type="t",
        data={"a": [1, 2]},
        description="d",
        source="s",
        cached_at="2026-10-28T00:00:00",
    )
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError):
        service._write_csv(item, item.model_dump(), target)
    assert workspace_tree(s.ws) == before and s.rev.unchanged()


def test_b2_ordinary_content_csv_is_unchanged(content):
    from lobster.services.data_access.workspace_content_service import (
        ContentType,
        MetadataContent,
        WorkspaceContentService,
    )

    s = content
    item = MetadataContent(
        identifier="plain",
        content_type="t",
        data={"a": [1, 2]},
        description="d",
        source="s",
        cached_at="2026-10-28T00:00:00",
    )
    path = WorkspaceContentService(s.dm).write_content(
        item, ContentType.METADATA, output_format="csv"
    )
    assert Path(path) == s.ws / "metadata" / "plain.csv"
    assert pd.read_csv(path)["a"].tolist() == [1, 2]


# ============================================ B2: ML export late refusal (SEC-07)

ML_MODULE = "lobster.agents.machine_learning.machine_learning_expert"


def _hooked_matrix(hook):
    import scipy.sparse as sp

    class _Hooked(sp.csr_matrix):
        """Runs ``hook`` once when the export densifies the matrix."""

        def toarray(self, *args, **kwargs):
            nonlocal hook
            if hook is not None:
                run, hook = hook, None
                run()
            return super().toarray(*args, **kwargs)

    return _Hooked(np.ones((3, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "fmt,name", [("tensorflow", "tf_data.npz"), ("numpy", "features.npy")]
)
def test_b2_ml_export_late_refusal_crosses_both_handlers(session, fmt, name, caplog):
    import logging

    from lobster.agents.machine_learning.machine_learning_expert import (
        machine_learning_expert,
    )

    s = session
    export_dir = s.dm.exports_dir / "ml"
    target = export_dir / f"m_{name}"
    s.dm.store_modality(
        "m",
        ad.AnnData(
            X=_hooked_matrix(lambda: alias(target, _revision_file(s))),
            obs=pd.DataFrame(index=["a", "b", "c"]),
            var=pd.DataFrame(index=["g1", "g2"]),
        ),
    )
    tools = factory_tools(machine_learning_expert, ML_MODULE, s.dm)
    with caplog.at_level(logging.ERROR, logger=ML_MODULE):
        response = tools["export_for_ml_framework"].invoke(
            {
                "modality_name": "m",
                "format": fmt,
                "include_labels": False,
                "output_dir": "ml",
            }
        )
    assert os.path.islink(target)  # the retarget really happened mid-export
    assert response == MANAGED_EXPORT_MESSAGE
    assert s.rev.unchanged() and s.logged == []
    assert caplog.records == []


# =========================================================== B3 (lane B2 delta reviews)


def test_b3_export_dirs_refuses_a_folder_whose_check_name_points_out(session):
    """A link at the classified name inside a managed folder cannot redirect the check."""
    from lobster.core.figures.export_dirs import assert_dirs_allowed

    s = session
    (s.ws / "queue-safe").mkdir()
    alias(s.ws / "exports" / "alias", s.rev.directory)
    alias(s.rev.directory / ".export-dir-check", s.ws / "queue-safe" / "probe")
    before = workspace_tree(s.ws)
    with pytest.raises(ManagedPathError) as refused:
        assert_dirs_allowed(s.ws, s.ws / "exports" / "alias")
    assert refused.value.details == {} and str(s.ws) not in str(refused.value)
    assert workspace_tree(s.ws) == before


def test_b3_export_dirs_refuses_any_existing_entry_at_the_check_name(session):
    from lobster.core.figures.export_dirs import assert_dirs_allowed

    s = session
    folder = s.ws / "exports" / "plain"
    folder.mkdir(parents=True)
    assert_dirs_allowed(s.ws, folder)  # absent: an ordinary folder is allowed
    (folder / ".export-dir-check").write_text("x")
    with pytest.raises(ManagedPathError):  # present (even a plain file): refused
        assert_dirs_allowed(s.ws, folder)


def _no_record_names(caplog, *needles):
    leaked = [
        r
        for r in caplog.records
        if any(n in f"{r.getMessage()} {r.args!r}" for n in needles)
    ]
    assert leaked == [], [r.getMessage() for r in leaked]


@pytest.mark.parametrize("field", ["highlight_residues", "highlight_groups"])
def test_b3_pymol_rejected_highlight_value_is_not_logged(session, pymol, field, caplog):
    import logging

    s = session
    fig, rev = s.rev.directory.parent.name, s.rev.directory.name
    spelled = f"../plots/{fig}/{rev}"
    if field == "highlight_residues":
        args = {"highlight_residues": f"A:15,{spelled}"}
    else:
        args = {"highlight_groups": f"15|red|sticks;{spelled}"}
    with caplog.at_level(logging.DEBUG):
        response = _pymol_tool(s.dm).invoke({"pdb_id": PDB, "execute": False, **args})
    _no_record_names(caplog, fig, rev)
    assert fig not in response and rev not in response
    if field == "highlight_residues":  # refused before anything (the fixed text)
        assert response.startswith("Error creating visualization: "), response
        assert "Invalid highlight residues" in response
        assert pymol.calls == []
    else:  # the malformed group is skipped with a fixed warning, as before
        assert "Skipped a highlight group" in caplog.text
    assert s.rev.unchanged()


def test_b3_chem_refusal_logs_no_workspace_path(session, caplog):
    import logging

    s = session
    alias(s.ws / "metadata", s.rev.directory)
    tools = _chem_tools(s.dm)  # factory set-up first
    with caplog.at_level(logging.DEBUG):
        response = tools["fingerprint_similarity"].invoke({"smiles_list": "CCO,CCN"})
    assert response == MANAGED_EXPORT_MESSAGE
    _no_record_names(caplog, str(s.ws), os.path.realpath(s.ws))
