"""
Unit tests for AQUADIF category enum and provenance configuration.

These tests validate the AQUADIF taxonomy structure that all contract tests
depend on. They ensure the enum has exactly 10 categories with correct names,
values, and provenance requirements.
"""

import ast

import pytest

from lobster.config.aquadif import (
    PROVENANCE_REQUIRED,
    AquadifCategory,
    has_provenance_call,
    requires_provenance,
)


class TestAquadifCategory:
    """Test AQUADIF category enum structure and behavior."""

    def test_has_exactly_ten_categories(self):
        """Verify the enum has exactly 10 category members."""
        assert (
            len(AquadifCategory) == 10
        ), f"AQUADIF must have exactly 10 categories, found {len(AquadifCategory)}"

    def test_category_names(self):
        """Verify all 10 expected category names are present."""
        expected_names = {
            "IMPORT",
            "QUALITY",
            "FILTER",
            "PREPROCESS",
            "ANALYZE",
            "ANNOTATE",
            "DELEGATE",
            "SYNTHESIZE",
            "UTILITY",
            "CODE_EXEC",
        }
        actual_names = {cat.name for cat in AquadifCategory}

        assert actual_names == expected_names, (
            f"Category names mismatch.\n"
            f"Expected: {sorted(expected_names)}\n"
            f"Actual: {sorted(actual_names)}\n"
            f"Missing: {sorted(expected_names - actual_names)}\n"
            f"Extra: {sorted(actual_names - expected_names)}"
        )

    def test_category_values_match_names(self):
        """Verify each category's value equals its name (e.g., IMPORT.value == 'IMPORT')."""
        for category in AquadifCategory:
            assert category.value == category.name, (
                f"Category {category.name} has mismatched value '{category.value}'. "
                f"Values should match names for consistency."
            )

    def test_provenance_required_has_seven_members(self):
        """Verify PROVENANCE_REQUIRED has exactly 7 category members."""
        assert (
            len(PROVENANCE_REQUIRED) == 7
        ), f"PROVENANCE_REQUIRED must have exactly 7 categories, found {len(PROVENANCE_REQUIRED)}"

    def test_provenance_required_categories(self):
        """Verify exact membership of PROVENANCE_REQUIRED set."""
        expected = {
            AquadifCategory.IMPORT,
            AquadifCategory.QUALITY,
            AquadifCategory.FILTER,
            AquadifCategory.PREPROCESS,
            AquadifCategory.ANALYZE,
            AquadifCategory.ANNOTATE,
            AquadifCategory.SYNTHESIZE,
        }
        assert PROVENANCE_REQUIRED == expected, (
            f"PROVENANCE_REQUIRED membership mismatch.\n"
            f"Expected: {sorted(c.value for c in expected)}\n"
            f"Actual: {sorted(c.value for c in PROVENANCE_REQUIRED)}\n"
            f"Missing: {sorted(c.value for c in (expected - PROVENANCE_REQUIRED))}\n"
            f"Extra: {sorted(c.value for c in (PROVENANCE_REQUIRED - expected))}"
        )

    def test_non_provenance_categories(self):
        """Verify DELEGATE, UTILITY, CODE_EXEC are NOT in PROVENANCE_REQUIRED."""
        non_provenance = {
            AquadifCategory.DELEGATE,
            AquadifCategory.UTILITY,
            AquadifCategory.CODE_EXEC,
        }

        for category in non_provenance:
            assert (
                category not in PROVENANCE_REQUIRED
            ), f"{category.value} should NOT require provenance but is in PROVENANCE_REQUIRED"

    def test_requires_provenance_true(self):
        """Verify requires_provenance() returns True for IMPORT category."""
        assert (
            requires_provenance("IMPORT") is True
        ), "IMPORT category should require provenance"

    def test_requires_provenance_false(self):
        """Verify requires_provenance() returns False for UTILITY category."""
        assert (
            requires_provenance("UTILITY") is False
        ), "UTILITY category should NOT require provenance"

    def test_requires_provenance_invalid(self):
        """Verify requires_provenance() raises ValueError for invalid category string."""
        with pytest.raises(ValueError) as exc_info:
            requires_provenance("INVALID")

        error_message = str(exc_info.value)
        assert "INVALID" in error_message, "Error should mention the invalid category"
        assert (
            "not a valid AquadifCategory" in error_message
        ), "Error should explain it's not a valid category"

    def test_string_comparison(self):
        """Verify string enum behavior: AquadifCategory.IMPORT == 'IMPORT' is True."""
        # This validates the (str, Enum) base class works correctly
        assert (
            AquadifCategory.IMPORT == "IMPORT"
        ), "String enum should allow direct string comparison"
        assert AquadifCategory.QUALITY == "QUALITY"
        assert AquadifCategory.UTILITY == "UTILITY"

        # Also verify value access
        assert AquadifCategory.IMPORT.value == "IMPORT"


class TestHasProvenanceCall:
    """Test has_provenance_call AST helper function."""

    def test_detects_ir_keyword(self):
        """Should detect log_tool_usage with ir= keyword argument."""
        import ast

        tree = ast.parse('data_manager.log_tool_usage("test", params, stats, ir=ir)')
        assert has_provenance_call(tree) is True

    def test_rejects_without_ir(self):
        """Should return False when log_tool_usage has no ir= keyword."""
        import ast

        tree = ast.parse('data_manager.log_tool_usage("test", params, stats)')
        assert has_provenance_call(tree) is False

    def test_detects_ir_in_multiline(self):
        """Should detect ir= in multi-line function body."""
        import ast

        source = """
def my_tool(data):
    result, stats, ir = service.analyze(adata)
    data_manager.log_tool_usage("my_tool", {"data": data}, stats, ir=ir)
    return "done"
"""
        tree = ast.parse(source)
        assert has_provenance_call(tree) is True

    def test_rejects_unrelated_call(self):
        """Should not match unrelated method names."""
        import ast

        tree = ast.parse('data_manager.save_results("test", ir=ir)')
        assert has_provenance_call(tree) is False

    def test_empty_tree(self):
        """Should return False for empty/trivial AST."""
        import ast

        tree = ast.parse("pass")
        assert has_provenance_call(tree) is False

    def test_rejects_literal_ir_none(self):
        """A literal ir=None writes no provenance and must not satisfy the check.

        This is the case that let 34 tools declare provenance=True and pass: the
        check matched the keyword's *name* and never looked at its value.
        """
        import ast

        tree = ast.parse('data_manager.log_tool_usage("test", params, stats, ir=None)')
        assert has_provenance_call(tree) is False

    def test_accepts_ir_expression_that_cannot_be_evaluated_statically(self):
        """A call or attribute is taken at face value — only literal None is rejected."""
        import ast

        for expr in ("build_ir()", "result.ir", "ir or fallback_ir"):
            tree = ast.parse(f'data_manager.log_tool_usage("t", p, s, ir={expr})')
            assert has_provenance_call(tree) is True, expr

    def test_one_real_ir_among_several_calls_is_enough(self):
        """A tool logging twice, once with real IR, still tracks provenance."""
        import ast

        source = """
data_manager.log_tool_usage("audit", {}, {}, ir=None)
data_manager.log_tool_usage("analyze", {}, stats, ir=ir)
"""
        assert has_provenance_call(ast.parse(source)) is True

    def test_all_calls_passing_none_is_a_violation(self):
        """Every log_tool_usage call is examined, not just the first."""
        import ast

        source = """
data_manager.log_tool_usage("a", {}, {}, ir=None)
data_manager.log_tool_usage("b", {}, {}, ir=None)
"""
        assert has_provenance_call(ast.parse(source)) is False


class TestProvenanceGateInspection:
    """Guard the two inspection helpers the AQUADIF provenance gate relies on.

    These live in lobster.testing.contract_mixins but are tested here because
    tests/unit/config is one of the four paths Basic CI actually selects — a
    guard CI never runs is not a guard.

    Both helpers exist because the gate previously mis-read the code it was
    validating and reported 1 violation where there were 31.
    """

    def test_parses_indented_source_containing_a_column_zero_string(self):
        """dedent alone cannot handle a code_template that starts at column 0.

        Tools defined inside a factory are indented, but many embed a notebook
        ``code_template`` whose lines deliberately begin at column 0. That makes
        dedent's common prefix empty, so it strips nothing and ast.parse raises
        "unexpected indent" — silently exempting the tools with the richest
        provenance IR from validation.
        """
        import textwrap

        from lobster.testing.contract_mixins import AgentContractTestMixin

        source = '''    @tool
    def load_thing(path: str) -> str:
        """Docstring."""
        ir = AnalysisStep(
            code_template="""# Load it
import anndata as ad
adata = ad.read_h5ad(path)""",
        )
        data_manager.log_tool_usage("load_thing", {}, {}, ir=ir)
        return "done"
'''
        # Establish the precondition: the old approach genuinely fails here.
        with pytest.raises(SyntaxError):
            ast.parse(textwrap.dedent(source))

        tree = AgentContractTestMixin._parse_tool_source(source)
        assert tree is not None, "indented source with a column-0 template must parse"
        assert has_provenance_call(tree) is True

    def test_parse_returns_none_for_genuinely_broken_source(self):
        """Unparseable source must report failure, not be silently accepted."""
        from lobster.testing.contract_mixins import AgentContractTestMixin

        assert AgentContractTestMixin._parse_tool_source("def broken(:\n") is None

    def test_implementations_include_the_coroutine_behind_a_sync_wrapper(self):
        """An async tool's body lives in .coroutine, not .func.

        enable_sync_fallback assigns .func a wrapper that merely awaits the
        coroutine, so inspecting .func alone finds no log_tool_usage call and
        every async tool looks like a violation regardless of what it does.
        """
        from langchain_core.tools import tool as lc_tool

        from lobster.testing.contract_mixins import AgentContractTestMixin
        from lobster.tools._async_compat import enable_sync_fallback

        @lc_tool
        async def async_thing(x: str) -> str:
            """An async tool."""
            return x

        assert async_thing.func is None
        enable_sync_fallback(async_thing)
        assert async_thing.func is not None  # now a wrapper

        impls = AgentContractTestMixin._tool_implementations(async_thing)
        names = [getattr(f, "__name__", None) for f in impls]
        assert (
            "async_thing" in names
        ), f"the real implementation must be inspected, got {names}"

    def test_implementations_of_a_sync_tool_is_its_function(self):
        """A plain sync tool yields exactly its own function, once."""
        from langchain_core.tools import tool as lc_tool

        from lobster.testing.contract_mixins import AgentContractTestMixin

        @lc_tool
        def sync_thing(x: str) -> str:
            """A sync tool."""
            return x

        assert sync_thing.coroutine is None
        impls = AgentContractTestMixin._tool_implementations(sync_thing)
        assert [getattr(f, "__name__", None) for f in impls] == ["sync_thing"]
