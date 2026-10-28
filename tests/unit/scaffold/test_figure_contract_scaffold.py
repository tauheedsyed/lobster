"""Scaffold: generated tests execute the figure contract; validate-plugin checks wiring."""

import importlib.util

import pytest

from lobster.scaffold import scaffold_agent
from lobster.scaffold.validators import validate_plugin


def _pkg(tmp_path, children=None):
    return scaffold_agent(
        name="epigenomics_expert",
        display_name="Epigenomics Expert",
        description="Epigenomics analysis",
        children=children or [],
        output_dir=tmp_path / "out",
    )


def _d8(results):
    return [r for r in results if r.check == "Figure contract (D8)"]


def test_the_generated_d8_test_runs_and_holds_on_the_generated_agent(
    tmp_path, monkeypatch
):
    pkg = _pkg(tmp_path)
    monkeypatch.syspath_prepend(str(pkg))
    spec = importlib.util.spec_from_file_location(
        "generated_contract", pkg / "tests" / "test_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = module.TestEpigenomicsExpertFigureContract()
    work = tmp_path / "work"
    work.mkdir()
    props = []
    with pytest.warns(UserWarning, match="list_epigenomics_modalities \\(UTILITY\\)"):
        contract.test_figure_contract_coverage(
            work, lambda name, value: props.append((name, value))
        )
    assert props and props[0][0] == "d8_coverage"
    contract.test_figure_tool_cases_hold(work)


def test_a_scaffolded_package_passes_the_d8_wiring_check(tmp_path):
    results = _d8(validate_plugin(_pkg(tmp_path)))
    assert [r.message for r in results] == [
        "analyze_epigenomics: has a case",
        "assess_epigenomics_quality: has a case",
        "import_epigenomics_data: has a case",
    ]
    assert all(r.passed for r in results)


def test_tests_without_the_mixin_fail_the_check(tmp_path):
    pkg = _pkg(tmp_path)
    test_file = pkg / "tests" / "test_contract.py"
    text = test_file.read_text()
    test_file.write_text(text.replace("(FigureContractTestMixin)", "(object)"))
    [result] = _d8(validate_plugin(pkg))
    assert not result.passed and "do not use FigureContractTestMixin" in result.message


def test_a_covered_tool_without_a_case_fails_and_a_skip_is_a_warning(tmp_path):
    pkg = _pkg(tmp_path)
    test_file = pkg / "tests" / "test_contract.py"
    text = test_file.read_text()
    text = text.replace('"analyze_epigenomics",', '"renamed_tool",')
    test_file.write_text(text)
    by_name = {r.message.split(":")[0]: r for r in _d8(validate_plugin(pkg))}
    assert not by_name["analyze_epigenomics"].passed

    test_file.write_text(
        text.replace(
            "figure_contract_skips = {}",
            'figure_contract_skips = {"analyze_epigenomics": "needs a GPU fixture"}',
        )
    )
    by_name = {r.message.split(":")[0]: r for r in _d8(validate_plugin(pkg))}
    skipped = by_name["analyze_epigenomics"]
    assert skipped.passed and skipped.severity == "warning"


def test_utility_tools_are_not_required_to_have_cases(tmp_path):
    names = [r.message for r in _d8(validate_plugin(_pkg(tmp_path)))]
    assert not any("list_epigenomics_modalities" in n for n in names)


def test_a_package_with_children_still_passes(tmp_path):
    results = _d8(validate_plugin(_pkg(tmp_path, children=["methylation_expert"])))
    assert results and all(r.passed for r in results)


def test_non_literal_case_names_are_reported_not_evaluated(tmp_path):
    pkg = _pkg(tmp_path)
    test_file = pkg / "tests" / "test_contract.py"
    text = test_file.read_text()
    text = text.replace('"analyze_epigenomics",', "TOOL_NAME,", 1)
    test_file.write_text('TOOL_NAME = "analyze_epigenomics"\n' + text)
    results = _d8(validate_plugin(pkg))
    dynamic = [r for r in results if "not checked" in r.message]
    assert len(dynamic) == 1 and dynamic[0].severity == "warning"
    by_name = {r.message.split(":")[0]: r for r in results}
    assert not by_name["analyze_epigenomics"].passed  # presence-only: no evaluation
