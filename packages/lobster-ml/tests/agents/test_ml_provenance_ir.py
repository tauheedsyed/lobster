"""ML preparation tools log IR that reproduces the feature matrix and the splits."""

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lobster.agents.machine_learning.machine_learning_expert import (
    machine_learning_expert,
)
from lobster.core.notebooks.exporter import NotebookExporter
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import (
    RecordingDataManager,
    factory_tools,
    replay_ir,
)

MODULE = "lobster.agents.machine_learning.machine_learning_expert"


@pytest.fixture
def samples() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(4, size=(60, 200)).astype(np.float32)
    X[:, 150:] = 0  # sparse tail exercises zero handling
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"condition": pd.Categorical(["a", "b", "c"] * 20)},
            index=[f"s{i}" for i in range(60)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(200)]),
    )


def _tools(adata):
    dm = RecordingDataManager({"data": adata})
    return dm, factory_tools(machine_learning_expert, MODULE, dm)


@pytest.mark.parametrize(
    "feature_selection,handle_zeros",
    [
        ("highly_variable", "keep"),
        ("pca", "keep"),
        ("all", "remove"),
        ("all", "impute"),
    ],
)
def test_prepare_ml_features_replays(samples, feature_selection, handle_zeros):
    original = samples.copy()
    dm, tools = _tools(samples)
    result = tools["prepare_ml_features"].invoke(
        {
            "modality_name": "data",
            "feature_selection": feature_selection,
            "n_features": 40,
            "handle_zeros": handle_zeros,
            "save_result": False,
        }
    )
    assert "Successfully prepared" in result, result
    ir = dm.last_ir("prepare_ml_features")
    assert isinstance(ir, AnalysisStep) and ir.replayable
    produced = dm.modalities["data_ml_features"]
    replayed = replay_ir(ir, adata=original.copy())["adata"]
    assert list(replayed.var_names) == list(produced.var_names)
    np.testing.assert_allclose(np.asarray(replayed.X), np.asarray(produced.X))
    assert replayed.uns["ml_preprocessing"] == produced.uns["ml_preprocessing"]


@pytest.mark.parametrize(
    "validation_size,stratify_by", [(0.2, "condition"), (0.0, None)]
)
def test_create_ml_splits_replays(samples, validation_size, stratify_by):
    original = samples.copy()
    dm, tools = _tools(samples)
    result = tools["create_ml_splits"].invoke(
        {
            "modality_name": "data",
            "test_size": 0.25,
            "validation_size": validation_size,
            "stratify_by": stratify_by,
            "random_state": 7,
            "save_result": False,
        }
    )
    assert "Successfully created" in result, result
    ir = dm.last_ir("create_ml_splits")
    assert isinstance(ir, AnalysisStep) and ir.replayable
    scope = replay_ir(ir, adata=original.copy())
    pd.testing.assert_series_equal(
        scope["adata"].obs["ml_split"], dm.modalities["data"].obs["ml_split"]
    )
    assert list(scope["adata_train"].obs_names) == list(
        dm.modalities["data_train"].obs_names
    )
    assert list(scope["adata_test"].obs_names) == list(
        dm.modalities["data_test"].obs_names
    )
    if validation_size:
        assert list(scope["adata_validation"].obs_names) == list(
            dm.modalities["data_validation"].obs_names
        )
    else:
        assert scope["adata_validation"] is None


def _split(tools, validation_size=0.2, random_state=7):
    tools["create_ml_splits"].invoke(
        {
            "modality_name": "data",
            "test_size": 0.25,
            "validation_size": validation_size,
            "random_state": random_state,
            "save_result": False,
        }
    )


def _prepare(tools, name, **kwargs):
    args = {
        "modality_name": name,
        "feature_selection": "all",
        "scale": False,
        "save_result": False,
    }
    args.update(kwargs)
    result = tools["prepare_ml_features"].invoke(args)
    assert "Successfully prepared" in result, result


def _replay_all(dm, original):
    """Replay every logged step in order, like an exported notebook."""
    scope = {"adata": original.copy()}
    for call in dm.calls:
        scope = replay_ir(call.ir, **scope)
    return scope


def _assert_same(replayed, produced):
    assert list(replayed.obs_names) == list(produced.obs_names)
    assert list(replayed.var_names) == list(produced.var_names)
    np.testing.assert_allclose(np.asarray(replayed.X), np.asarray(produced.X))


@pytest.mark.parametrize(
    "steps",
    [
        ["data_train"],
        ["data_train", "data_test"],
        ["data_test", "data_train"],
        ["data_train", "data_validation"],
        ["data_train", "data_train_ml_features"],
        ["data_train", "data_test", "data_train_ml_features"],
    ],
    ids="->".join,
)
def test_sequential_feature_steps_replay_on_their_own_input(samples, steps):
    """Each feature step must replay on the modality it used, never on held-out
    samples or on the previous step's output."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    for name in steps:
        _prepare(tools, name)
    scope = _replay_all(dm, original)
    for name in steps:
        _assert_same(
            scope["lobster_modalities"][f"{name}_ml_features"],
            dm.modalities[f"{name}_ml_features"],
        )
    _assert_same(scope["adata"], dm.modalities[f"{steps[-1]}_ml_features"])


def test_same_split_prepared_twice_uses_the_split_both_times(samples):
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train", feature_selection="pca", n_features=10)
    _prepare(tools, "data_train")  # again from the split, not from the PCA output
    scope = _replay_all(dm, original)
    _assert_same(scope["adata"], dm.modalities["data_train_ml_features"])
    assert scope["adata"].n_vars == 200


def test_split_retained_from_an_earlier_split_replays_on_that_split(samples):
    """Re-splitting without validation keeps the old validation modality; its
    feature step must replay on those samples, not on the whole dataset."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools, validation_size=0.2, random_state=7)
    _split(tools, validation_size=0.0, random_state=42)
    _prepare(tools, "data_validation")
    produced = dm.modalities["data_validation_ml_features"]
    assert produced.n_obs == 9
    _assert_same(_replay_all(dm, original)["adata"], produced)


def test_resplit_after_a_feature_step_splits_the_parent(samples):
    """The second split cell must split the parent, not the previous step's features."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools, random_state=7)
    _prepare(tools, "data_train", feature_selection="pca", n_features=10)
    _split(tools, random_state=42)
    _prepare(tools, "data_train")
    _assert_same(
        _replay_all(dm, original)["adata"], dm.modalities["data_train_ml_features"]
    )


@pytest.mark.parametrize("second_step", ["prepare_original", "split_original"])
def test_original_is_still_available_after_its_features(samples, second_step):
    """Preparing features must not lose the original: a later step on it
    replays on the original, not on the previous step's output."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _prepare(tools, "data", feature_selection="pca", n_features=10)
    if second_step == "prepare_original":
        _prepare(tools, "data")
        expected = dm.modalities["data_ml_features"]
    else:
        _split(tools)
        _prepare(tools, "data_train")
        expected = dm.modalities["data_train_ml_features"]
    replayed = _replay_all(dm, original)["adata"]
    _assert_same(replayed, expected)
    assert replayed.n_vars == 200


def test_replay_refuses_a_name_a_non_ml_step_may_have_replaced(samples):
    """After a non-ML cell rebinds adata, a registered name is ambiguous."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    scope = _replay_all(dm, original)
    scope["adata"] = original[:12, :3].copy()  # e.g. a re-import under the same name
    _prepare(tools, "data")
    with pytest.raises(ValueError, match="Ambiguous input 'data'"):
        replay_ir(dm.last_ir("prepare_ml_features"), **scope)


def test_ambiguity_persists_across_later_ml_steps(samples):
    """An unrelated ML step after a rebind must not make old names trusted again."""
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    scope = _replay_all(dm, original)
    scope["adata"] = original[:12, :3].copy()  # a non-ML cell rebinds adata
    _prepare(tools, "data_train")
    _prepare(tools, "data")
    with pytest.raises(ValueError, match="Ambiguous input"):
        for call in dm.calls[1:]:
            scope = replay_ir(call.ir, **scope)


@pytest.mark.parametrize(
    "first",
    [
        {"feature_selection": "highly_variable", "n_features": 10},
        {"feature_selection": "all", "scale": True},
    ],
)
def test_loaded_split_stays_the_input_for_later_steps(samples, first):
    """A split loaded as notebook input is registered, so a second step on it does
    not pick up the first step's output (which no longer carries the split tag)."""
    dm, tools = _tools(samples)
    _split(tools)
    loaded = dm.modalities["data_train"].copy()
    dm.calls.clear()
    _prepare(tools, "data_train", **first)
    assert "ml_split_source" not in dm.modalities["data_train_ml_features"].uns
    _prepare(tools, "data_train")
    scope = {"adata": loaded}
    for call in dm.calls:
        scope = replay_ir(call.ir, **scope)
    _assert_same(scope["adata"], dm.modalities["data_train_ml_features"])


def test_split_loaded_as_input_replays_without_its_split_step(samples):
    """A saved split loaded in a new session carries its tag and replays alone."""
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train")
    loaded = dm.modalities["data_train"].copy()
    replayed = replay_ir(dm.last_ir("prepare_ml_features"), adata=loaded)["adata"]
    _assert_same(replayed, dm.modalities["data_train_ml_features"])


def test_split_feature_step_fails_closed_without_its_split(samples):
    original = samples.copy()
    dm, tools = _tools(samples)
    _split(tools)
    _prepare(tools, "data_train")
    with pytest.raises(ValueError, match="create_ml_splits"):
        replay_ir(dm.last_ir("prepare_ml_features"), adata=original.copy())


@pytest.mark.parametrize("with_parent", [False, True])
def test_unrelated_train_named_modality_gets_no_split_selection(samples, with_parent):
    """A modality merely named *_train (even with ml_split == 'train') is not a
    split unless create_ml_splits tagged it."""
    dm, tools = _tools(samples)
    if with_parent:
        _split(tools)
    cohort = samples[:8].copy()
    cohort.obs["ml_split"] = "train"
    cohort.uns.pop("ml_split_source", None)
    dm.modalities["cohort_train"] = cohort
    _prepare(tools, "cohort_train")
    code = dm.last_ir("prepare_ml_features").render()
    assert "split_id" not in code
    if with_parent:
        with pytest.raises(ValueError, match="Unregistered ML input 'cohort_train'"):
            _replay_all(dm, samples)
    else:
        # An untagged train-suffixed name is a valid first loaded input, not a
        # request to infer a split from its suffix.
        _assert_same(
            replay_ir(dm.last_ir("prepare_ml_features"), adata=cohort)["adata"],
            dm.modalities["cohort_train_ml_features"],
        )


def _real_session(samples, tmp_path, input_name="data"):
    """Real load, storage, saving and provenance; only the agent LLM is mocked."""
    input_path = tmp_path / "input.h5ad"
    samples.write_h5ad(input_path)
    dm = DataManagerV2(workspace_path=tmp_path / "workspace", auto_scan=False)
    dm.load_modality(
        input_name, input_path, adapter="transcriptomics_bulk", validate=False
    )
    return dm, factory_tools(machine_learning_expert, MODULE, dm), input_path


def _export_real_notebook(dm, monkeypatch, strategy="successful"):
    # Environment/Git metadata is irrelevant to numerical replay and must not
    # run subprocesses. The notebook itself is the real exported artifact.
    monkeypatch.setattr(NotebookExporter, "_get_system_info", lambda self: {})
    monkeypatch.setattr(NotebookExporter, "_snapshot_dependencies", lambda self: {})
    path = dm.export_notebook("ml_replay", filter_strategy=strategy)
    return nbformat.read(path, as_version=4)


def _run_all(notebook, input_path, prefix, scope):
    """Execute every code cell, including load and save, in one kernel namespace.

    Inject parameters immediately after the parameters cell just as Papermill
    does. Passing the same dictionary again models a same-kernel Run All.
    """
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(cell.source, f"<notebook-cell:{i}>", "exec"), scope)
        if "parameters" in cell.metadata.get("tags", []):
            scope.update(input_data=str(input_path), output_prefix=str(prefix))
    return scope


def _saved_matches(path, expected):
    assert path.is_file(), path
    actual = ad.read_h5ad(path)
    assert actual.obs_names.tolist() == expected.obs_names.tolist()
    assert actual.var_names.tolist() == expected.var_names.tolist()

    def to_dense(x):
        return x.toarray() if sparse.issparse(x) else np.asarray(x)

    np.testing.assert_array_equal(to_dense(actual.X), to_dense(expected.X))
    if "ml_split" in expected.obs:
        assert actual.obs["ml_split"].tolist() == expected.obs["ml_split"].tolist()
    return actual


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize("validation_size", [0.1, 0.0])
@pytest.mark.parametrize("sparse_input", [False, True], ids=["dense", "csr"])
def test_default_saved_split_notebook_writes_exact_cohorts(
    samples, tmp_path, monkeypatch, strategy, validation_size, sparse_input
):
    if sparse_input:
        samples.X = sparse.csr_matrix(samples.X)
    dm, tools, input_path = _real_session(samples, tmp_path)
    # Intentionally omit save_result: the default-saving public API is the bug.
    result = tools["create_ml_splits"].invoke(
        {"modality_name": "data", "validation_size": validation_size}
    )
    assert "Successfully created" in result, result
    notebook = _export_real_notebook(dm, monkeypatch, strategy)
    prefix = tmp_path / "results"
    scope = _run_all(notebook, input_path, prefix, {})
    splits = ["train", "test"] + (["validation"] if validation_size else [])
    for split in splits:
        name = f"data_{split}"
        expected = dm.get_modality(name)
        # Both runtime save and complete exported notebook persist the same
        # cohort, IDs, order, feature names, matrix and split labels.
        _saved_matches(dm.data_dir / f"{name}.h5ad", expected)
        saved = _saved_matches(tmp_path / f"results_{name}.h5ad", expected)
        assert set(saved.obs["ml_split"]) == {split}
    assert {p.name for p in tmp_path.glob("results_*.h5ad")} == {
        *(f"results_data_{split}.h5ad" for split in splits),
        "results_processed.h5ad",
    }
    _saved_matches(tmp_path / "results_processed.h5ad", dm.get_modality("data"))
    assert scope["adata"].n_obs == samples.n_obs  # parent retained for later steps


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize("workflow", ["feature", "mixed_known_inputs"])
def test_full_notebook_run_all_twice_resets_only_at_load(
    samples, tmp_path, monkeypatch, strategy, workflow
):
    dm, tools, input_path = _real_session(samples, tmp_path)
    if workflow == "feature":
        result = tools["prepare_ml_features"].invoke(
            {"modality_name": "data", "feature_selection": "all", "scale": False}
        )
        assert "Successfully prepared" in result, result
        expected = dm.get_modality("data_ml_features")
        final_path = tmp_path / "results_data_ml_features.h5ad"
    else:
        # A multi-output saving split followed by known branches: final saver
        # must not overwrite validation with the later train features or parent.
        result = tools["create_ml_splits"].invoke({"modality_name": "data"})
        assert "Successfully created" in result, result
        for name, kwargs in [
            ("data_train", {"feature_selection": "pca", "n_features": 5}),
            ("data_test", {}),
            ("data_validation", {}),
            ("data_train", {}),
            ("data", {}),
        ]:
            _prepare(tools, name, **kwargs)
        expected = dm.get_modality("data_ml_features")
        final_path = tmp_path / "results_processed.h5ad"
    notebook = _export_real_notebook(dm, monkeypatch, strategy)
    scope = {}
    first_registry = None
    for run in range(2):
        _run_all(notebook, input_path, tmp_path / "results", scope)
        _assert_same(scope["adata"], expected)
        _saved_matches(final_path, expected)
        if workflow == "mixed_known_inputs":
            for split in ("train", "test", "validation"):
                _saved_matches(
                    tmp_path / f"results_data_{split}.h5ad",
                    dm.get_modality(f"data_{split}"),
                )
        if run == 0:
            first_registry = scope["lobster_modalities"]
        else:
            assert scope["lobster_modalities"] is not first_registry
        assert scope["lobster_epoch"] == 0

    # An actual within-run replacement must still fail closed, even after two
    # successful full reruns. This exercises the same exported analysis cell.
    scope["adata"] = samples[:12, :3].copy()
    feature_cells = [
        c
        for c in notebook.cells
        if c.cell_type == "code"
        and "MLPreparationService().select_ml_features(" in c.source
    ]
    with pytest.raises(ValueError, match="Ambiguous input"):
        exec(feature_cells[0].source, scope)


@pytest.mark.parametrize("tagged_split", [False, True])
def test_full_notebook_accepts_first_loaded_input_and_reuses_it(
    samples, tmp_path, monkeypatch, tagged_split
):
    if tagged_split:
        recorded, tools = _tools(samples)
        _split(tools)
        loaded = recorded.modalities["data_train"].copy()
        name = "data_train"
    else:
        loaded = samples[:8].copy()
        name = "cohort_train"  # legitimate name, no inferred split
    dm, tools, input_path = _real_session(loaded, tmp_path, input_name=name)
    _prepare(tools, name, feature_selection="pca", n_features=5)
    _prepare(tools, name)  # first input, not its PCA result, remains known
    notebook = _export_real_notebook(dm, monkeypatch)
    scope = {}
    for _ in range(2):
        _run_all(notebook, input_path, tmp_path / "results", scope)
        expected = dm.get_modality(f"{name}_ml_features")
        _assert_same(scope["adata"], expected)
        _saved_matches(tmp_path / "results_processed.h5ad", expected)
        assert scope["adata"].shape == loaded.shape


@pytest.mark.parametrize("strategy", ["successful", "replay"])
@pytest.mark.parametrize("known_splits_first", [False, True])
def test_full_notebook_rejects_unloaded_cohort_after_known_ml_inputs(
    samples, tmp_path, monkeypatch, strategy, known_splits_first
):
    dm, tools, input_path = _real_session(samples, tmp_path)
    cohort = samples[:8].copy()
    # No accidental tag/suffix inference: this is an independent resident input.
    dm.store_modality("cohort_train", cohort)
    if known_splits_first:
        _split(tools)
        _prepare(tools, "data_train", feature_selection="pca", n_features=5)
        _prepare(tools, "data_test")
        _prepare(tools, "data_validation")
    else:
        _prepare(tools, "data", feature_selection="pca", n_features=5)
    _prepare(tools, "cohort_train")  # real runtime operation is valid
    assert dm.get_modality("cohort_train_ml_features").shape == (8, 200)
    notebook = _export_real_notebook(dm, monkeypatch, strategy)
    scope = {}
    with pytest.raises(ValueError, match="Unregistered ML input 'cohort_train'"):
        _run_all(notebook, input_path, tmp_path / "results", scope)
    assert "cohort_train" not in scope["lobster_modalities"]
    assert "cohort_train_ml_features" not in scope["lobster_modalities"]
    assert not list(tmp_path.glob("results*.h5ad"))  # rejection precedes final save


@pytest.mark.parametrize("strategy", ["successful", "replay"])
def test_self_loading_peptide_split_run_all_twice_starts_fresh(
    tmp_path, monkeypatch, strategy
):
    """The real initial peptide import has no dedicated H5AD load cell."""
    from lobster.agents.proteomics.peptide_tools import create_peptide_tools

    fasta = tmp_path / "cohort.fasta"
    fasta.write_text("".join(f">peptide_{i}\nACDEFGHIK\n" for i in range(60)))
    dm = DataManagerV2(workspace_path=tmp_path / "workspace", auto_scan=False)
    peptide_tools = {
        tool.name: tool for tool in create_peptide_tools(dm, workspace_path=tmp_path)
    }
    result = peptide_tools["import_peptide_sequences"].invoke(
        {"file_path": str(fasta)}
    )
    assert "STATUS: complete" in result, result
    assert dm.get_modality("peptides").obs_names.tolist() == [
        f"peptide_{i}" for i in range(60)
    ]
    tools = factory_tools(machine_learning_expert, MODULE, dm)
    # Use all public defaults, including saving each train/test/validation split.
    result = tools["create_ml_splits"].invoke({"modality_name": "peptides"})
    assert "Successfully created" in result, result
    notebook = _export_real_notebook(dm, monkeypatch, strategy)
    code = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert not any("ad.read_h5ad(input_data)" in source for source in code)
    import_source = next(s for s in code if ".import_sequences(" in s)
    split_source = next(s for s in code if "adata = lobster_input('peptides')" in s)
    scope = {}
    first_registries = None
    for run in range(2):
        # One namespace, all cells, no manual registry seeding or clearing.
        prefix = tmp_path / f"run_{run}"
        _run_all(notebook, fasta, prefix, scope)
        for split in ("train", "test", "validation"):
            name = f"peptides_{split}"
            expected = dm.get_modality(name)
            _saved_matches(dm.data_dir / f"{name}.h5ad", expected)
            saved = _saved_matches(tmp_path / f"run_{run}_{name}.h5ad", expected)
            assert set(saved.obs["ml_split"]) == {split}
            pd.testing.assert_frame_equal(
                saved.obs, expected.obs, check_dtype=False, check_categorical=False
            )
        _saved_matches(tmp_path / f"run_{run}_processed.h5ad", dm.get_modality("peptides"))
        assert set(scope["lobster_modalities"]) == {
            "peptides", "peptides_train", "peptides_test", "peptides_validation"
        }
        assert set(scope["lobster_registered"]) == set(scope["lobster_modalities"])
        assert set(scope["lobster_registered"].values()) == {0}
        assert scope["lobster_epoch"] == 0
        if run == 0:
            first_registries = (
                scope["lobster_modalities"], scope["lobster_registered"]
            )
        else:
            assert scope["lobster_modalities"] is not first_registries[0]
            assert scope["lobster_registered"] is not first_registries[1]

    # Re-executing an import within this run is NOT a run boundary. It must
    # retain the ambiguity guard and reject before writing any new split files.
    exec(import_source, scope)
    scope["output_prefix"] = str(tmp_path / "rejected")
    with pytest.raises(ValueError, match="Ambiguous input 'peptides'"):
        exec(split_source, scope)
    assert not list(tmp_path.glob("rejected*.h5ad"))
