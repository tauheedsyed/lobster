"""
Contract test mixins for validating plugin API compliance.

This module provides test mixins that agent package authors can inherit
to validate their packages follow the plugin contract established in Phase 2.

Usage:
    from lobster.testing import AgentContractTestMixin

    class TestMyAgent(AgentContractTestMixin):
        agent_module = 'my_package.agents.my_agent'
        factory_name = 'my_agent'

    # Then run with pytest
"""

import ast
import importlib
import inspect
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Set, Tuple
from unittest.mock import MagicMock


def _extract_tools(graph) -> list:
    """Tool objects of a compiled agent graph (LangGraph wraps ToolNode in PregelNode)."""
    if hasattr(graph, "nodes") and "tools" in graph.nodes:
        tools_node = graph.nodes["tools"]
        if hasattr(tools_node, "tools_by_name"):
            return list(tools_node.tools_by_name.values())
        if hasattr(tools_node, "bound") and hasattr(tools_node.bound, "tools_by_name"):
            return list(tools_node.bound.tools_by_name.values())
    return []


class AgentContractTestMixin:
    """
    Test mixin for validating agent package plugin API compliance.

    Subclasses must define:
        agent_module: str - Module path containing AGENT_CONFIG (entry point module)
        factory_name: str - Name of the factory function

    Optional:
        factory_module: str - Module containing the factory (defaults to agent_module)
        expected_tier: str - Expected tier requirement (default: None, skips check)
        tools_required: bool - If True (default), empty tools = FAIL. If False, empty tools = skip.

    Example (same module for config and factory):
        class TestAnnotationExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics.annotation_expert'
            factory_name = 'annotation_expert'

    Example (separate modules - package __init__ has config, nested module has factory):
        class TestTranscriptomicsExpert(AgentContractTestMixin):
            agent_module = 'lobster_transcriptomics'  # AGENT_CONFIG in __init__.py
            factory_module = 'lobster_transcriptomics.transcriptomics_expert'  # factory here
            factory_name = 'transcriptomics_expert'
    """

    # Required attributes - subclasses must define these
    agent_module: ClassVar[str]
    factory_name: ClassVar[str]

    # Optional attributes
    factory_module: ClassVar[Optional[str]] = (
        None  # Defaults to agent_module if not set
    )
    expected_tier: ClassVar[Optional[str]] = None
    is_parent_agent: ClassVar[bool] = (
        False  # Set True for parent agents with child agents
    )
    tools_required: ClassVar[bool] = True  # Set False for agents still being scaffolded

    # Standard factory parameters as defined in Phase 2 plugin contract
    STANDARD_PARAMS: ClassVar[Set[str]] = {
        "data_manager",
        "callback_handler",
        "agent_name",
        "delegation_tools",
        "workspace_path",
    }

    # Deprecated parameters that should not be used
    DEPRECATED_PARAMS: ClassVar[Set[str]] = {
        "handoff_tools",  # Renamed to delegation_tools in Phase 2
    }

    # Required AGENT_CONFIG fields
    REQUIRED_CONFIG_FIELDS: ClassVar[Set[str]] = {
        "name",
        "tier_requirement",
    }

    # Class-level cache for tool extraction (populated once per test class)
    _tools_cache: ClassVar[dict] = {}

    def _get_module(self) -> Any:
        """Import and return the config module (contains AGENT_CONFIG)."""
        return importlib.import_module(self.agent_module)

    def _get_factory_module(self) -> Any:
        """Import and return the factory module (contains factory function)."""
        module_path = self.factory_module if self.factory_module else self.agent_module
        return importlib.import_module(module_path)

    def _get_factory(self) -> Any:
        """Get the factory function from the factory module."""
        module = self._get_factory_module()
        return getattr(module, self.factory_name)

    def _get_factory_params(self) -> Set[str]:
        """Get the set of parameter names from the factory signature."""
        factory = self._get_factory()
        sig = inspect.signature(factory)
        return set(sig.parameters.keys())

    def _get_agent_config(self) -> Any:
        """Get the AGENT_CONFIG from the module."""
        module = self._get_module()
        return getattr(module, "AGENT_CONFIG", None)

    def _get_tools_from_factory(self) -> list:
        """
        Get list of tool objects from agent factory.

        Calls the factory with MagicMock dependencies and extracts
        tool objects. Raises on unexpected failures (fail-by-default).

        Known-benign failures (ImportError for optional deps like torch/scvi)
        raise ImportError which callers handle via pytest.skip().

        Results are cached per class to avoid redundant graph compilation.

        Returns:
            List of tool objects from the factory's compiled graph

        Raises:
            ImportError: When factory needs optional dependencies not installed
            Exception: All other errors propagate (fail-by-default)
        """
        import pytest

        cache_key = f"{self.agent_module}:{self.factory_name}"
        if cache_key in self.__class__._tools_cache:
            return self.__class__._tools_cache[cache_key]

        try:
            from unittest.mock import patch

            factory = self._get_factory()

            # Create mock dependencies for factory call
            mock_data_manager = MagicMock()
            mock_callback_handler = MagicMock()
            mock_delegation_tools = []

            # Patch LLM creation — factories need LLMs but contract tests
            # don't execute tools, they only inspect metadata.
            # Patch at both the definition site and the factory's module to
            # handle both `from X import create_llm` and `X.create_llm()`.
            factory_module_path = (
                self.factory_module if self.factory_module else self.agent_module
            )
            with (
                patch(
                    "lobster.config.llm_factory.create_llm",
                    return_value=MagicMock(),
                ),
                patch(
                    f"{factory_module_path}.create_llm",
                    return_value=MagicMock(),
                ),
            ):
                # Call factory to get CompiledStateGraph
                graph = factory(
                    data_manager=mock_data_manager,
                    callback_handler=mock_callback_handler,
                    agent_name="test",
                    delegation_tools=mock_delegation_tools,
                    workspace_path=None,
                )

            tools = _extract_tools(graph)

            self.__class__._tools_cache[cache_key] = tools
            return tools

        except (ImportError, ModuleNotFoundError) as e:
            # Known-benign: optional heavy deps (torch, scvi-tools) not installed
            pytest.skip(
                f"Factory '{self.factory_name}' requires unavailable dependency: {e}"
            )

    def _has_log_tool_usage_with_ir(self, tree: ast.AST) -> bool:
        """
        Check if AST tree contains log_tool_usage call with ir= keyword argument.

        Delegates to the standalone ``has_provenance_call`` function in
        ``lobster.config.aquadif`` for reusability by ``lobster validate-plugin``.

        Args:
            tree: AST tree from parsed tool source code

        Returns:
            True if log_tool_usage(ir=ir) call found, False otherwise
        """
        from lobster.config.aquadif import has_provenance_call

        return has_provenance_call(tree)

    @staticmethod
    def _tool_implementations(tool) -> list:
        """Return the callable that actually implements ``tool``.

        Inspecting ``tool.func`` alone is not enough. For an ``async def`` tool
        LangChain leaves ``func=None`` and puts the body in ``coroutine``, and
        ``enable_sync_fallback`` then assigns ``func`` a ``_sync_fallback``
        closure that merely awaits the coroutine. Reading only ``func`` therefore
        inspects a wrapper whose body contains no ``log_tool_usage`` call at all.

        The coroutine, when present, is the implementation and must carry the
        provenance call itself: a compliant sync callable cannot vouch for a
        non-compliant coroutine. Without a coroutine, ``func`` is inspected.
        """
        fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
        if fn is None:
            return []
        return [inspect.unwrap(fn)]  # strip functools.wraps chains

    @staticmethod
    def _parse_tool_source(source: str) -> ast.AST | None:
        """Parse tool source that may be indented, returning None if impossible.

        ``textwrap.dedent`` is not sufficient on its own. Tools defined inside a
        factory are indented, but many also embed a notebook ``code_template``
        as a triple-quoted string whose lines deliberately start at column 0.
        That makes dedent's common prefix empty, so it strips nothing and
        ``ast.parse`` raises "unexpected indent" on line 1 — which silently
        exempted the tools building the *richest* provenance IR from validation.

        Wrapping in a dummy block makes the leading indentation legal; the
        column-0 lines are inside a string literal, so the parser does not care
        about their position.
        """
        candidates = (
            textwrap.dedent(source),
            source,
            "if True:\n" + source,
        )
        for candidate in candidates:
            try:
                return ast.parse(candidate)
            except SyntaxError:
                continue
        return None

    def _require_tools(self) -> list:
        """Get tools, failing or skipping based on tools_required setting."""
        import pytest

        tools = self._get_tools_from_factory()

        if not tools:
            if self.tools_required:
                raise AssertionError(
                    f"Factory '{self.factory_name}' returned no tools — "
                    f"check graph construction in '{self.agent_module}'. "
                    f"Set tools_required = False if this agent has no tools yet."
                )
            else:
                pytest.skip(
                    f"No tools from factory '{self.factory_name}' (tools_required=False)."
                )

        return tools

    def test_factory_has_standard_params(self) -> None:
        """
        Verify factory function has the standard parameters.

        The plugin contract requires these parameters:
        - data_manager: DataManagerV2 instance
        - callback_handler: Optional callback handler for LLM interactions
        - agent_name: Name identifier for the agent instance
        - delegation_tools: List of tools for delegating to child agents
        - workspace_path: Optional path to workspace directory
        """
        params = self._get_factory_params()
        missing = self.STANDARD_PARAMS - params

        assert (
            not missing
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"is missing required parameters: {sorted(missing)}. "
            f"Factory must accept: {sorted(self.STANDARD_PARAMS)}"
        )

    def test_no_deprecated_handoff_tools(self) -> None:
        """
        Verify factory does NOT use deprecated handoff_tools parameter.

        The handoff_tools parameter was renamed to delegation_tools in Phase 2
        for semantic clarity. Factories should use delegation_tools instead.
        """
        params = self._get_factory_params()
        deprecated_used = self.DEPRECATED_PARAMS & params

        assert (
            not deprecated_used
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Factory '{self.factory_name}' in '{self.agent_module}' "
            f"uses deprecated parameters: {sorted(deprecated_used)}. "
            f"Use 'delegation_tools' instead of 'handoff_tools'."
        )

    def test_agent_config_exists(self) -> None:
        """
        Verify AGENT_CONFIG exists at module top level.

        AGENT_CONFIG must be defined at the top of the module (before heavy imports)
        to enable fast entry point discovery (<50ms target).
        """
        config = self._get_agent_config()

        assert (
            config is not None
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Module '{self.agent_module}' does not have AGENT_CONFIG defined. "
            f"Add: AGENT_CONFIG = AgentRegistryConfig(...) at module top, "
            f"before heavy imports."
        )

    def test_agent_config_has_name(self) -> None:
        """
        Verify AGENT_CONFIG has required 'name' field.

        The name field uniquely identifies the agent in the registry
        and is used for routing, configuration, and logging.
        """
        config = self._get_agent_config()

        assert (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            config is not None
        ), f"Cannot check name field: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            config, "name"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'name' field."

        assert (
            config.name
        ), f"AGENT_CONFIG.name in '{self.agent_module}' is empty."  # nosec B101 # Contract-test assertion, not runtime security enforcement.

    def test_agent_config_has_tier_requirement(self) -> None:
        """
        Verify AGENT_CONFIG has required 'tier_requirement' field.

        The tier_requirement field controls runtime access based on
        user's subscription level. Valid values: "free", "premium", "enterprise".
        """
        config = self._get_agent_config()

        assert (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            config is not None
        ), f"Cannot check tier_requirement: AGENT_CONFIG not found in '{self.agent_module}'"

        assert hasattr(  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            config, "tier_requirement"
        ), f"AGENT_CONFIG in '{self.agent_module}' is missing 'tier_requirement' field."

        valid_tiers = {"free", "premium", "enterprise"}
        tier = config.tier_requirement

        assert (
            tier in valid_tiers
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"AGENT_CONFIG.tier_requirement in '{self.agent_module}' has invalid value '{tier}'. "
            f"Must be one of: {sorted(valid_tiers)}"
        )

        # Optional: check expected tier if specified
        if self.expected_tier is not None:
            assert (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
                tier == self.expected_tier
            ), f"AGENT_CONFIG.tier_requirement is '{tier}', expected '{self.expected_tier}'"

    # =========================================================================
    # AQUADIF Contract Tests (Phase 2)
    # =========================================================================

    def test_tools_have_aquadif_metadata(self) -> None:
        """
        Verify all tools have AQUADIF metadata (categories and provenance).

        Each tool must declare:
        - .metadata dict with "categories" and "provenance" keys
        - categories: non-empty list of AQUADIF category strings
        - provenance: boolean indicating if provenance tracking is required

        This validates TEST-02: Metadata presence.
        """
        tools = self._require_tools()

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))

            # Check metadata dict exists
            assert (
                hasattr(tool, "metadata") and tool.metadata is not None
            ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
                f"Tool '{tool_name}' in '{self.agent_module}' is missing .metadata. "
                f"Add: tool.metadata = {{'categories': [...], 'provenance': True/False}}"
            )

            # Check categories key exists
            assert (
                "categories" in tool.metadata
            ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
                f"Tool '{tool_name}' in '{self.agent_module}' has .metadata but missing 'categories' key. "
                f"Add: 'categories': ['PRIMARY_CATEGORY']"
            )

            # Check categories is non-empty list
            categories = tool.metadata["categories"]
            assert (
                isinstance(categories, list) and len(categories) > 0
            ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
                f"Tool '{tool_name}' in '{self.agent_module}' has empty or invalid categories. "
                f"Must be non-empty list: ['CATEGORY']"
            )

            # Check provenance key exists
            assert (
                "provenance" in tool.metadata
            ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
                f"Tool '{tool_name}' in '{self.agent_module}' has .metadata but missing 'provenance' key. "
                f"Add: 'provenance': True or False"
            )

    def test_categories_are_valid(self) -> None:
        """
        Verify all tool categories are valid AQUADIF categories.

        Categories must be from the 10-category AQUADIF taxonomy:
        IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE,
        DELEGATE, SYNTHESIZE, UTILITY, CODE_EXEC

        This validates TEST-03: Category validity.
        """
        from lobster.config.aquadif import AquadifCategory

        tools = self._require_tools()

        invalid_categories = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            for category in categories:
                try:
                    # Validate by attempting to construct enum member
                    AquadifCategory(category)
                except ValueError:
                    invalid_categories.append((tool_name, category))

        assert (
            not invalid_categories
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Invalid AQUADIF categories found in '{self.agent_module}':\n"
            + "\n".join(
                f"  - Tool '{tool}': invalid category '{cat}'"
                for tool, cat in invalid_categories
            )
            + f"\n\nValid categories: {', '.join(c.value for c in AquadifCategory)}"
        )

    def test_categories_capped_at_three(self) -> None:
        """
        Verify tools have at most 3 categories.

        Most tools need only 1 category. Multi-category is uncommon and should
        only be used when a tool has substantial functionality in multiple areas.

        This validates TEST-04: Category cap.
        """
        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if len(categories) > 3:
                violations.append((tool_name, len(categories)))

        assert (
            not violations
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' exceed 3-category limit:\n"
            + "\n".join(
                f"  - Tool '{tool}': {count} categories (max is 3)"
                for tool, count in violations
            )
            + "\n\nGuidance: Most tools need 1 category; multi-category is uncommon."
        )

    def test_provenance_tools_have_flag(self) -> None:
        """
        Verify tools with provenance-required categories declare provenance: True.

        Categories requiring provenance (7 of 10):
        IMPORT, QUALITY, FILTER, PREPROCESS, ANALYZE, ANNOTATE, SYNTHESIZE

        Categories NOT requiring provenance (3 of 10):
        DELEGATE, UTILITY, CODE_EXEC

        The primary category (first in list) determines the provenance requirement.

        This validates TEST-05: Provenance flag compliance.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if not categories:
                continue  # Already caught by metadata presence test

            # Get primary category (first in list)
            primary_category_str = categories[0]
            try:
                primary_category = AquadifCategory(primary_category_str)
            except ValueError:
                continue  # Already caught by category validity test

            # Check if primary category requires provenance
            requires_prov = primary_category in PROVENANCE_REQUIRED
            declared_prov = tool.metadata.get("provenance", False)

            if requires_prov and not declared_prov:
                violations.append((tool_name, primary_category.value))

        assert (
            not violations
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' have provenance-required categories but provenance=False:\n"
            + "\n".join(
                f"  - Tool '{tool}': primary category '{cat}' requires provenance=True"
                for tool, cat in violations
            )
        )

    def test_metadata_objects_are_unique(self) -> None:
        """
        Verify each tool has its own metadata dict (no shared objects).

        Tools must not share metadata dict references. Common cause:
        metadata dict created outside the tool creation loop in the factory.

        This validates TEST-07: Metadata uniqueness.
        """
        tools = self._require_tools()

        metadata_ids = [
            id(tool.metadata) for tool in tools if hasattr(tool, "metadata")
        ]

        # Count occurrences of each id
        from collections import Counter

        id_counts = Counter(metadata_ids)
        shared_ids = {id_val: count for id_val, count in id_counts.items() if count > 1}

        assert (
            not shared_ids
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' share metadata dict objects ({len(shared_ids)} shared dicts). "
            f"Each tool must have its own metadata dict. "
            f"Common cause: metadata dict created outside loop in factory. "
            f"Fix: Create metadata dict inside the tool creation loop."
        )

    def test_provenance_categories_not_buried(self) -> None:
        """
        Verify provenance-required categories are not buried behind non-provenance primary.

        If ANY category in the list requires provenance, the PRIMARY (first) category
        must also require provenance. This prevents gaming by reordering categories
        to dodge provenance validation.

        Example violation: categories: ["UTILITY", "IMPORT"] — IMPORT requires provenance
        but UTILITY (primary) does not, so provenance checks would be bypassed.

        This validates TEST-09: Category ordering integrity.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if len(categories) < 2:
                continue  # Single-category tools can't have ordering issues

            # Parse all categories, skip if any are invalid (caught by other test)
            parsed = []
            for cat_str in categories:
                try:
                    parsed.append(AquadifCategory(cat_str))
                except ValueError:
                    break
            else:
                primary = parsed[0]
                secondary_cats = parsed[1:]

                primary_requires = primary in PROVENANCE_REQUIRED
                any_secondary_requires = any(
                    c in PROVENANCE_REQUIRED for c in secondary_cats
                )

                if any_secondary_requires and not primary_requires:
                    buried = [
                        c.value for c in secondary_cats if c in PROVENANCE_REQUIRED
                    ]
                    violations.append((tool_name, primary.value, buried))

        assert (
            not violations
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' have provenance-required categories buried behind non-provenance primary:\n"
            + "\n".join(
                f"  - Tool '{tool}': primary '{primary}' doesn't require provenance, "
                f"but secondary {buried} does. Reorder so a provenance-required category is first."
                for tool, primary, buried in violations
            )
        )

    def test_minimum_viable_parent(self) -> None:
        """
        Verify parent agents have minimum viable category set.

        Parent agents (domain experts with child agents) must provide:
        - IMPORT: Ability to load data
        - QUALITY: Ability to assess data quality
        - ANALYZE or DELEGATE: Either perform analysis or delegate to children

        This validates TEST-06: Minimum viable parent agent capabilities.
        """
        # Parent-only requirements do not apply to standalone agents. Return
        # rather than skip so the aggregate contract continues its other checks.
        if not self.is_parent_agent:
            return

        tools = self._require_tools()

        # Collect all unique categories across all tools
        all_categories = set()
        for tool in tools:
            categories = tool.metadata.get("categories", [])
            all_categories.update(categories)

        # Check for minimum viable set
        has_import = "IMPORT" in all_categories
        has_quality = "QUALITY" in all_categories
        has_analyze_or_delegate = (
            "ANALYZE" in all_categories or "DELEGATE" in all_categories
        )

        missing = []
        if not has_import:
            missing.append("IMPORT")
        if not has_quality:
            missing.append("QUALITY")
        if not has_analyze_or_delegate:
            missing.append("ANALYZE or DELEGATE")

        assert (
            not missing
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Parent agent '{self.factory_name}' missing minimum viable categories. "
            f"Has: {sorted(all_categories)}. "
            f"Missing: {', '.join(missing)}. "
            f"Parent agents need: IMPORT + QUALITY + (ANALYZE or DELEGATE)"
        )

    def test_provenance_ast_validation(self) -> None:
        """
        Verify tools with provenance metadata call log_tool_usage(ir=ir).

        This is the CRITICAL test addressing the Phase 1 eval finding that
        100% of agents copied provenance boilerplate mechanically without
        adjusting based on metadata flags. This test catches metadata-runtime
        disconnect at test time by parsing tool source code.

        Rules:
        - If tool declares provenance=True: MUST call log_tool_usage(ir=ir)
        - If primary category requires provenance: tool SHOULD declare provenance=True
          (caught by test_provenance_tools_have_flag)

        This validates TEST-08: Provenance AST validation.
        """
        from lobster.config.aquadif import PROVENANCE_REQUIRED, AquadifCategory

        tools = self._require_tools()

        violations = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            categories = tool.metadata.get("categories", [])

            if not categories:
                continue  # Already caught by metadata presence test

            # Get primary category
            primary_category_str = categories[0]
            try:
                primary_category = AquadifCategory(primary_category_str)
            except ValueError:
                continue  # Already caught by category validity test

            # Check provenance requirements
            requires_prov = primary_category in PROVENANCE_REQUIRED
            declared_prov = tool.metadata.get("provenance", False)

            # Only validate if tool declares provenance OR category requires it
            if not (declared_prov or requires_prov):
                continue

            # Inspect the callable that implements the tool, which for async
            # tools is the coroutine rather than .func — see _tool_implementations.
            #
            # Source that cannot be read or parsed is a FAILURE, not a skip. This
            # used to warn and `continue`, which turned an uninspectable tool into
            # a passing one and hid real violations behind a warning nobody reads.
            implementations = self._tool_implementations(tool)
            if not implementations:
                violations.append(
                    (
                        tool_name,
                        primary_category.value,
                        "declares provenance but exposes no inspectable "
                        "implementation (both .func and .coroutine are None)",
                    )
                )
                continue

            has_ir_call = False
            unreadable = []
            for fn in implementations:
                try:
                    source = inspect.getsource(fn)
                except (OSError, TypeError) as e:
                    unreadable.append(
                        f"{getattr(fn, '__name__', fn)}: source unavailable "
                        f"({e.__class__.__name__})"
                    )
                    continue

                tree = self._parse_tool_source(source)
                if tree is None:
                    unreadable.append(
                        f"{getattr(fn, '__name__', fn)}: source could not be parsed"
                    )
                    continue

                if self._has_log_tool_usage_with_ir(tree):
                    has_ir_call = True
                    break

            if declared_prov and not has_ir_call and unreadable:
                violations.append(
                    (
                        tool_name,
                        primary_category.value,
                        "declares provenance=True but its provenance call could "
                        "not be verified — " + "; ".join(unreadable),
                    )
                )
                continue

            # Validate: if provenance is declared, the call must exist
            if declared_prov and not has_ir_call:
                reason = "declares provenance=True"
                if requires_prov:
                    reason += f" (and primary category '{primary_category.value}' requires it)"
                reason += (
                    " but does NOT call log_tool_usage with a real ir "
                    "(absent, or passed literally as ir=None)"
                )
                violations.append((tool_name, primary_category.value, reason))

        assert (
            not violations
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' have provenance metadata-runtime disconnect:\n"
            + "\n".join(
                f"  - Tool '{tool}' (category: {cat}): {reason}"
                for tool, cat, reason in violations
            )
            + "\n\nFix: Ensure tools with provenance=True call log_tool_usage(ir=ir) in their"
            "\nimplementation, passing an AnalysisStep — `ir=None` does not satisfy this."
            "\nIf the tool genuinely produces nothing reproducible, the category and"
            "\nprovenance flag are wrong, not the call."
        )

    def test_tool_params_no_bare_collections(self) -> None:
        """
        Verify no tool parameters use bare collection types (list, dict, tuple, set).

        Bare collection types produce JSON schemas without `items`/`properties`
        fields, which causes 400 INVALID_ARGUMENT on Gemini API. All collection
        type hints must be parameterized: list[str], dict[str, Any], etc.

        This validates TEST-10: Gemini-compatible tool schemas.
        """
        import types
        import typing

        tools = self._require_tools()

        violations = []
        # Bare builtin collection types that produce invalid JSON schema
        bare_builtins = {list, dict, tuple, set}
        # typing module aliases (unparameterized) — also produce bare schemas
        bare_typing = {typing.List, typing.Dict, typing.Tuple, typing.Set}
        bare_builtins | bare_typing

        def _is_bare(hint) -> str | None:
            """Return type name if hint is a bare collection, else None."""
            if hint in bare_builtins:
                return hint.__name__
            if hint in bare_typing:
                # e.g. typing.List -> "List"
                return getattr(hint, "__name__", None) or str(hint).split(".")[-1]
            return None

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))

            # Get the underlying function
            func = getattr(tool, "func", None)
            if func is None:
                continue

            try:
                hints = typing.get_type_hints(func)
            except Exception:
                continue  # nosec B112 # Unavailable optional type hints do not redefine the tool schema.

            sig = inspect.signature(func)

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "return"):
                    continue

                hint = hints.get(param_name)
                if hint is None:
                    continue

                # Unwrap Optional/Union to check the inner type
                actual = hint
                origin = typing.get_origin(actual)
                if origin is types.UnionType or origin is typing.Union:
                    # e.g. list | None -> check each arg
                    for arg in typing.get_args(actual):
                        if arg is type(None):
                            continue
                        name = _is_bare(arg)
                        if name:
                            violations.append((tool_name, param_name, name))
                else:
                    name = _is_bare(actual)
                    if name:
                        violations.append((tool_name, param_name, name))

        assert (
            not violations
        ), (  # nosec B101 # Contract-test assertion, not runtime security enforcement.
            f"Tools in '{self.agent_module}' have bare collection type hints "
            f"(breaks Gemini API — missing 'items' in JSON schema):\n"
            + "\n".join(
                f"  - Tool '{tool}', param '{param}': bare `{typ}` → "
                f"use `{typ}[str]` or appropriate parameterized type"
                for tool, param, typ in violations
            )
            + "\n\nFix: Replace `list` with `list[str]`, `dict` with `dict[str, Any]`, etc."
        )

    def test_all_contract_requirements(self) -> None:
        """
        Run all contract validation tests together.

        This is a convenience method that runs all validation checks
        in a single test, useful for quick verification.

        Includes both Phase 1 plugin contract tests and Phase 2 AQUADIF tests.
        """
        # Phase 1: Plugin contract tests
        self.test_factory_has_standard_params()
        self.test_no_deprecated_handoff_tools()
        self.test_agent_config_exists()
        self.test_agent_config_has_name()
        self.test_agent_config_has_tier_requirement()

        # Phase 2: AQUADIF contract tests (basic metadata)
        self.test_tools_have_aquadif_metadata()
        self.test_categories_are_valid()
        self.test_categories_capped_at_three()
        self.test_provenance_tools_have_flag()
        self.test_metadata_objects_are_unique()
        self.test_provenance_categories_not_buried()

        # Phase 2: AQUADIF contract tests (advanced)
        self.test_minimum_viable_parent()  # Applies only to parent agents
        self.test_provenance_ast_validation()

        # Phase 3: Schema compatibility tests
        self.test_tool_params_no_bare_collections()


# ===========================================================================
# Executed figure contract: tool-case coverage
# ===========================================================================
# Analysis tools never create figures; the eight transitional producers save only through
# stage -> log_tool_usage -> commit_staged. What is asserted is OBSERVED behaviour of real
# invocations on the cases a package supplies (calls and their arguments, files, catalog,
# legacy readers, named dataset fields), not static source. It says nothing about inputs,
# configurations or code paths the cases do not exercise, and it confines nothing: it is a
# test-time check, not a runtime guard.
#
# Detection boundary: figure files are ``.html``/``.png`` outside the managed
# store, any change to the managed store (``plots/``) outside a producer's commit, and
# figure-shaped JSON accepted by the central payload validator. Other image formats (svg,
# jpg, webp, pdf, ...) are NOT detected.

#: File types treated as rendered figures when written outside the managed store.
FIGURE_FILE_SUFFIXES = (".html", ".png")

#: AnnData fields compared for producer cases. ``raw`` means raw.X, raw.var and raw.varm.
PRODUCER_DATASET_FIELDS = (
    "X",
    "obs",
    "var",
    "uns",
    "obsm",
    "varm",
    "obsp",
    "varp",
    "layers",
    "raw",
)

_STORE_KINDS = ("store_commit", "store_execution", "store_rendition")


@dataclass(frozen=True)
class FigureToolCase:
    """One representative VALID invocation of an agent tool.

    ``setup(dm, monkeypatch)`` runs on a fresh real ``DataManagerV2`` BEFORE the real
    factory is built (after the harness's default LLM guard, so a case's own patches win);
    its patches stay live through factory, invocation and ``assert_success``, and are undone
    afterwards. ``assert_success(reply, dm)`` must prove the expected science output (a
    returned error string is a failure even inside a success ToolMessage). For a producer,
    ``args["save_plot"] is False`` makes the case a preview, which never counts as the
    producer's saving case.
    """

    tool_name: str
    args: Mapping[str, Any]
    setup: Callable[[Any, Any], None]
    assert_success: Callable[[Any, Any], None]
    label: str = ""

    @property
    def is_preview(self) -> bool:
        return self.args.get("save_plot") is False


class FigureContractReport(UserWarning):
    """Coverage report of the contract harness (also recorded as the ``d8_coverage`` property)."""


def _file_tree(root) -> Dict[str, str]:
    """``posix relpath -> sha256`` of every regular file (symlinks recorded by target)."""
    import hashlib
    import os

    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if os.path.islink(path):
                out[rel] = "symlink:" + os.readlink(path)
                continue
            with open(path, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _tree_delta(before: Dict[str, str], after: Dict[str, str]):
    added = sorted(k for k in after if k not in before)
    changed = sorted(k for k in after if k in before and before[k] != after[k])
    removed = sorted(k for k in before if k not in after)
    return added, changed, removed


def _figure_shaped(obj: Any) -> bool:
    if not isinstance(obj, dict) or not isinstance(obj.get("data"), list):
        return False
    traces = obj["data"]
    return isinstance(obj.get("layout"), dict) or (
        bool(traces) and all(isinstance(t, dict) for t in traces)
    )


def _plotly_json_problem(data: bytes) -> Optional[str]:
    """Central strict semantics for figure-shaped JSON (bounded detection, disclosed).

    A candidate is a JSON object with a ``data`` list and either a ``layout`` object or
    non-empty all-object traces (also nested under ``figure``). Candidates are judged by
    ``lobster.core.figures.payload.validate_figure_bytes`` (layout optional). Valid ->
    figure. A candidate the strict parser refuses (NaN/Infinity) or that is too large
    cannot be cleared and is reported. Anything else is not called a figure.
    """
    import json

    from lobster.core.figures.errors import FigureInvalidError, safe_reason
    from lobster.core.figures.payload import validate_figure_bytes

    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None
    candidates = [(obj, data)]
    if isinstance(obj, dict) and isinstance(obj.get("figure"), dict):
        candidates.append((obj["figure"], None))
    for candidate, raw in candidates:
        if not _figure_shaped(candidate):
            continue
        if raw is None:
            raw = json.dumps(candidate, allow_nan=True).encode("utf-8")
        try:
            validate_figure_bytes(raw)
        except FigureInvalidError as exc:
            reason = safe_reason(exc)
            if reason.endswith(("non_strict_json", "figure_too_large")):
                return f"figure-shaped JSON that cannot be cleared ({reason})"
            continue
        return "a Plotly figure JSON"
    return None


def _feed(h, obj: Any) -> None:
    """Deterministic digest of AnnData field values (one small helper per value kind)."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if obj is None:
        h.update(b"N")
    elif isinstance(obj, pd.DataFrame):
        h.update(b"F" + repr((obj.shape, [str(c) for c in obj.columns])).encode())
        h.update(
            repr((obj.columns.name, obj.index.name, str(obj.index.dtype))).encode()
        )
        h.update(repr([str(t) for t in obj.dtypes]).encode())
        h.update(repr([str(i) for i in obj.index]).encode())
        for col in obj.columns:
            series = obj[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                h.update(
                    repr((list(series.cat.categories), series.cat.ordered)).encode()
                )
            try:
                hashed = pd.util.hash_pandas_object(series, index=False).to_numpy()
                h.update(hashed.tobytes())
            except TypeError:  # unhashable cells (lists, dicts)
                h.update(repr(series.tolist()).encode())
    elif isinstance(obj, pd.Series):
        _feed(h, obj.to_frame())
    elif sp.issparse(obj):
        m = obj.tocsr()
        h.update(b"S" + repr((m.shape, str(m.dtype))).encode())
        for part in (m.data, m.indices, m.indptr):
            h.update(np.ascontiguousarray(part).tobytes())
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            h.update(b"O" + repr(obj.shape).encode())
            for item in obj.ravel():
                _feed(h, item)
        else:
            h.update(b"A" + repr((obj.shape, str(obj.dtype))).encode())
            h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, Mapping):
        h.update(b"M" + str(len(obj)).encode())
        for key in sorted(obj, key=str):
            h.update(repr(key).encode())
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L" + str(len(obj)).encode())
        for item in obj:
            _feed(h, item)
    elif hasattr(obj, "shape") and hasattr(obj, "__getitem__"):  # array-like (backed)
        _feed(h, np.asarray(obj[()]))
    else:
        h.update(b"V" + type(obj).__name__.encode() + repr(obj).encode())


def _dataset_digests(dm) -> Dict[str, Dict[str, str]]:
    """``modality -> {field -> digest}`` for :data:`PRODUCER_DATASET_FIELDS`."""
    import hashlib

    out: Dict[str, Dict[str, str]] = {}
    for name in sorted(dm.list_modalities()):
        adata = dm.get_modality(name)
        fields: Dict[str, str] = {}
        for field_name in PRODUCER_DATASET_FIELDS:
            h = hashlib.sha256()
            if field_name == "raw":
                raw = adata.raw
                _feed(
                    h,
                    None if raw is None else (raw.X, raw.var, dict(raw.varm.items())),
                )
            elif field_name in ("obsm", "varm", "obsp", "varp", "layers"):
                _feed(h, dict(getattr(adata, field_name).items()))
            else:
                _feed(h, getattr(adata, field_name))
            fields[field_name] = h.hexdigest()
        out[name] = fields
    return out


def _catalog(dm) -> Dict[str, str]:
    """``ref -> availability`` from the native reader (a fault raises: never 'empty')."""
    from lobster.core.figures.listing import figure_ref

    return {
        figure_ref(item): str(item.get("availability") or "available")
        for item in dm._saved_figure_records()
    }


def _readers(dm) -> Dict[str, Any]:
    """What the real legacy reader APIs return (ids/refs only)."""
    pm = dm.plot_manager
    latest, _, _ = pm.get_latest_plots()
    history, _, _ = pm.get_plot_history()
    return {
        "latest": [(p.get("id"), p.get("figure_ref")) for p in latest],
        "history": [p.get("id") for p in history],
    }


def _found_by_id(dm, plot_id) -> bool:
    _, stats, _ = dm.plot_manager.get_plot_by_id(plot_id)
    return bool(stats.get("found"))


def _install_figure_spies(monkeypatch, events: list, dm, workspace) -> None:
    """Wrap (never replace) every figure-writing seam and ``log_tool_usage``.

    Events are ``(kind, phase, data)``. ``stage``/``end`` carries the staged id;
    ``commit_staged``/``start`` carries ``tool_call_id``, ``staged_ids``, ``activity``, the
    catalog and the file tree at that moment; ``commit_staged``/``end`` carries the
    results and the file tree; ``log``/``start`` carries the catalog; ``log``/``end`` the
    returned activity.
    """
    import functools

    from lobster.core.figures.store import LocalFigureStore
    from lobster.core.plot_manager import PlotManager
    from lobster.core.runtime.data_manager import DataManagerV2

    def spy(cls, attr, kind, on_start=None, on_end=None):
        original = getattr(cls, attr)

        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            events.append((kind, "start", on_start(args, kwargs) if on_start else None))
            try:
                result = original(self, *args, **kwargs)
            except BaseException:
                events.append((kind, "raise", None))
                raise
            events.append((kind, "end", on_end(result) if on_end else result))
            return result

        monkeypatch.setattr(cls, attr, wrapper)

    def commit_start(args, kwargs):
        activity = args[0] if args else kwargs.get("activity")
        return {
            "tool_call_id": kwargs.get("tool_call_id"),
            "staged_ids": list(kwargs.get("staged_ids") or []),
            "activity": activity,
            "catalog": _catalog(dm),
            "tree": _file_tree(workspace),
        }

    def commit_end(result):
        return {"results": list(result or []), "tree": _file_tree(workspace)}

    spy(PlotManager, "add_plot", "stage", on_end=lambda r: r[0])
    spy(PlotManager, "commit_staged", "commit_staged", commit_start, commit_end)
    spy(DataManagerV2, "log_tool_usage", "log", on_start=lambda a, k: _catalog(dm))
    spy(LocalFigureStore, "commit", "store_commit")
    spy(LocalFigureStore, "commit_execution", "store_execution")
    spy(LocalFigureStore, "add_rendition", "store_rendition")


def _default_llm_guard(monkeypatch, factory_module_path: str) -> None:
    """Replace LLM creation for factory builds; applied BEFORE ``case.setup`` so a case's
    own ``create_llm`` patch (at either site) wins."""
    import importlib

    from lobster.config import llm_factory

    monkeypatch.setattr(llm_factory, "create_llm", lambda *a, **k: MagicMock())
    module = importlib.import_module(factory_module_path)
    if hasattr(module, "create_llm"):
        monkeypatch.setattr(module, "create_llm", lambda *a, **k: MagicMock())


def _build_tools(factory, dm) -> list:
    """Real factory with the real data manager (fresh per case; no metadata cache)."""
    graph = factory(
        data_manager=dm,
        callback_handler=MagicMock(),
        agent_name="test",
        delegation_tools=[],
        workspace_path=dm.workspace_path,
    )
    return _extract_tools(graph)


def _invoke(tool, args: Mapping[str, Any], tool_call_id: str) -> Tuple[Any, str]:
    """Invoke as a graph tool call; returns ``(content, status)``.

    Executes exactly the implementation :func:`figure_tool_key` identifies: the coroutine
    whenever one exists, else ``func``. The other interface of a dual tool is not exercised.
    """
    import asyncio

    call = {
        "type": "tool_call",
        "id": tool_call_id,
        "name": tool.name,
        "args": dict(args),
    }
    if getattr(tool, "coroutine", None) is not None:
        out = asyncio.run(tool.ainvoke(call))
    else:
        out = tool.invoke(call)
    return getattr(out, "content", out), getattr(out, "status", "success")


def _is_store_path(rel: str, refs) -> bool:
    """Managed path a trusted commit may create: a returned ref's revision folder or a
    store bookkeeping folder (``lobster.core.figures.paths.RESERVED_DIRS``)."""
    from lobster.core.figures import paths as P

    parts = rel.split("/")
    if len(parts) < 3 or parts[0] != P.PLOTS_DIR:
        return False
    if parts[1] in P.RESERVED_DIRS:
        return True
    return f"{parts[1]}@{parts[2]}" in refs and len(parts) >= 4


def _file_violations(
    workspace, before, after, managed_ok=lambda rel: False
) -> List[str]:
    """Figures written outside the trusted path (bounded detection, see module note)."""
    import os

    problems: List[str] = []
    added, changed, removed = _tree_delta(before, after)
    for rel in added + changed + removed:
        if rel.split("/")[0] == "plots":
            if not (rel in added and managed_ok(rel)):
                problems.append(f"changed the managed figure folder: {rel}")
            continue
        if rel in removed:
            continue
        lower = rel.lower()
        if lower.endswith(FIGURE_FILE_SUFFIXES):
            problems.append(f"wrote a figure file: {rel}")
        elif lower.endswith(".json"):
            with open(os.path.join(workspace, *rel.split("/")), "rb") as fh:
                verdict = _plotly_json_problem(fh.read())
            if verdict:
                problems.append(f"wrote {verdict}: {rel}")
    return problems


def _indices(events, kind, phase) -> List[int]:
    return [i for i, (k, ph, _) in enumerate(events) if k == kind and ph == phase]


def _no_figure_violations(
    events, dm, catalog_before, readers_before, producer
) -> List[str]:
    problems = []
    staged = _indices(events, "stage", "end")
    if staged and not producer:
        problems.append(f"staged {len(staged)} figure(s) (add_plot)")
    for kind in ("commit_staged",) + _STORE_KINDS:
        n = len(_indices(events, kind, "start"))
        if n:
            problems.append(f"{kind} called {n} time(s)")
    if _catalog(dm) != catalog_before:
        problems.append("the figure catalog changed")
    if _readers(dm) != readers_before:
        problems.append("legacy figure readers changed")
    for i in _indices(events, "stage", "end"):
        if _found_by_id(dm, events[i][2]):
            problems.append(f"get_plot_by_id shows staged figure {events[i][2]}")
    return problems


def _producer_violations(events, dm, workspace, tool_call_id, before) -> List[str]:
    """Saving producer: stage -> log -> exactly one commit_staged bound to THIS call."""
    problems: List[str] = []
    staged = [events[i][2] for i in _indices(events, "stage", "end")]
    stage_idx = _indices(events, "stage", "end")
    logs = _indices(events, "log", "end")
    commits = _indices(events, "commit_staged", "start")
    if not staged:
        problems.append("no figure was staged")
    if len(commits) != 1:
        problems.append(f"commit_staged called {len(commits)} time(s), expected 1")
        return problems
    c_start = commits[0]
    call = events[c_start][2]
    if call["tool_call_id"] != tool_call_id:
        problems.append("commit_staged did not carry the injected tool_call_id")
    ids = list(call["staged_ids"])
    if not ids or len(ids) != len(set(ids)) or sorted(ids) != sorted(staged):
        problems.append(
            "commit_staged staged_ids are not the figures staged in this call, each once"
        )
    last_stage = max(stage_idx) if stage_idx else -1
    if last_stage > c_start:
        problems.append("staged after commit_staged")
    logs_between = [i for i in logs if last_stage < i < c_start]
    if not logs_between:
        problems.append("log_tool_usage did not run between staging and commit_staged")
    elif events[logs_between[-1]][2] is None:
        problems.append("log_tool_usage returned no activity (no recorded provenance)")
    elif call["activity"] is not events[logs_between[-1]][2]:
        problems.append(
            "commit_staged activity is not the activity logged in this call"
        )
    for i in _indices(events, "log", "start"):
        if events[i][2] != before["catalog"]:
            problems.append("the catalog changed before log_tool_usage")
            break
    if call["catalog"] != before["catalog"]:
        problems.append("the catalog changed before commit_staged")
    ends = _indices(events, "commit_staged", "end")
    if not ends:
        problems.append("commit_staged raised")
        return problems
    c_end = ends[0]
    for kind in _STORE_KINDS:
        if any(not (c_start < i < c_end) for i in _indices(events, kind, "start")):
            problems.append(f"{kind} ran outside commit_staged")
    results = events[c_end][2]["results"]
    if not results:
        problems.append("commit_staged returned no result")
    # Declared contract: one CommitResult per staged id, in order (candidate_index 0..n-1).
    complete = len(results) == len(ids)
    if not complete:
        problems.append(
            f"commit_staged returned {len(results)} result(s) for {len(ids)} id(s)"
        )
    elif [getattr(r, "candidate_index", None) for r in results] != list(
        range(len(ids))
    ):
        problems.append("commit results are not one per staged id in order")
    statuses = {getattr(r, "status", None) for r in results}
    if not statuses <= {"available", "pending"}:
        problems.append(
            f"commit results not available/pending: {sorted(map(str, statuses))}"
        )
    available = {r.ref for r in results if getattr(r, "status", None) == "available"}
    returned = {r.ref for r in results if getattr(r, "ref", None)}
    # Managed-store changes: none before or after the commit; inside it, only new files of
    # the returned refs or store bookkeeping; nothing pre-existing modified or removed.
    tree_start, tree_end = call["tree"], events[c_end][2]["tree"]
    problems.extend(
        _file_violations(workspace, before["tree"], tree_start)
        + _file_violations(
            workspace, tree_start, tree_end, lambda rel: _is_store_path(rel, returned)
        )
        + _file_violations(workspace, tree_end, _file_tree(workspace))
    )
    catalog_after = _catalog(dm)
    if any(catalog_after.get(r) != a for r, a in before["catalog"].items()):
        problems.append("existing catalog entries changed")
    new = {r: a for r, a in catalog_after.items() if r not in before["catalog"]}
    if {r for r, a in new.items() if a == "available"} != available:
        problems.append(
            "available catalog entries differ from the returned available refs"
        )
    pending = {r.ref for r in results if getattr(r, "status", None) == "pending"}
    if not {r for r, a in new.items() if a != "available"} <= pending:
        problems.append("catalog has unavailable entries not returned as pending")
    rb, ra = before["readers"], _readers(dm)
    n_latest, n_hist = len(rb["latest"]), len(rb["history"])
    if (
        ra["latest"][:n_latest] != rb["latest"]
        or ra["history"][:n_hist] != rb["history"]
    ):
        problems.append("legacy figure readers were rewritten")
    else:
        new_latest = ra["latest"][n_latest:]
        if not {ref for _, ref in new_latest} <= available:
            problems.append(
                "legacy readers show a figure that is not an available returned ref"
            )
        if [pid for pid, _ in new_latest] != ra["history"][n_hist:]:
            problems.append("get_plot_history differs from get_latest_plots")
        if not {pid for pid, _ in new_latest} <= set(call["staged_ids"]):
            problems.append(
                "legacy readers show an id that was not committed in this call"
            )
    for sid, result in zip(ids, results, strict=True) if complete else ():
        shown = _found_by_id(dm, sid)
        if shown != (getattr(result, "status", None) == "available"):
            problems.append(f"get_plot_by_id({sid}) does not match its commit result")
    return problems


@contextmanager
def _fresh_case_context(factory, factory_module_path: str, workspace, setup=None):
    """Fresh real ``DataManagerV2`` -> default LLM guard -> ``setup(dm, mp)`` -> real
    factory. Patches stay live until the context exits. Shared by execution and coverage.
    A factory that cannot be built raises (fail closed; no dependency skip)."""
    import pytest

    from lobster.core.runtime.data_manager import DataManagerV2

    with pytest.MonkeyPatch.context() as mp:
        dm = DataManagerV2(workspace_path=workspace)
        _default_llm_guard(mp, factory_module_path)
        if setup is not None:
            setup(dm, mp)
        yield dm, mp, _build_tools(factory, dm)


def check_figure_tool_case(
    case: FigureToolCase,
    *,
    factory,
    factory_module_path: str,
    workspace,
    tool_call_id: str,
) -> List[str]:
    """Run one case on a fresh real ``DataManagerV2``; return its violations ([] = holds)."""
    import pytest

    from lobster.config.aquadif import figure_tool_key, is_transitional_figure_producer

    name = case.label or case.tool_name
    with _fresh_case_context(factory, factory_module_path, workspace, case.setup) as (
        dm,
        _mp,
        built,
    ):
        tools = {t.name: t for t in built}
        if case.tool_name not in tools:
            return [f"{name}: tool '{case.tool_name}' is not exposed by the factory"]
        tool = tools[case.tool_name]
        producer = is_transitional_figure_producer(figure_tool_key(tool))

        before = {
            "tree": _file_tree(workspace),
            "catalog": _catalog(dm),
            "readers": _readers(dm),
            "data": _dataset_digests(dm) if producer else None,
        }
        events: list = []
        spies = pytest.MonkeyPatch()
        _install_figure_spies(spies, events, dm, workspace)
        try:
            content, status = _invoke(tool, case.args, tool_call_id)
        except Exception as exc:  # a tool error is a case failure, never a pass
            return [f"{name}: the tool raised {type(exc).__name__}"]
        finally:
            spies.undo()

        problems: List[str] = []
        if status == "error":
            problems.append("the tool returned an error ToolMessage")
        try:
            case.assert_success(content, dm)
        except AssertionError as exc:
            problems.append(f"assert_success failed: {exc}")

        if producer and not case.is_preview:
            problems.extend(
                _producer_violations(events, dm, workspace, tool_call_id, before)
            )
        else:
            problems.extend(
                _no_figure_violations(
                    events, dm, before["catalog"], before["readers"], producer
                )
            )
            problems.extend(
                _file_violations(workspace, before["tree"], _file_tree(workspace))
            )
        if producer:
            data_after = _dataset_digests(dm)
            if set(data_after) != set(before["data"]):
                problems.append("the set of modalities changed")
            for mod in sorted(set(before["data"]) & set(data_after)):
                diff = [
                    f
                    for f in PRODUCER_DATASET_FIELDS
                    if before["data"][mod][f] != data_after[mod][f]
                ]
                if diff:
                    problems.append(f"modality '{mod}' changed: {', '.join(diff)}")
        return [f"{name}: {p}" for p in problems]


def _template_problems(
    tool_name: str, metadata: Mapping[str, Any], catalog
) -> List[str]:
    """``suggests_templates`` names must resolve via ``list_templates()`` (lazy catalog)."""
    if "suggests_templates" not in metadata:
        return []
    names = metadata["suggests_templates"]
    if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
        return [f"{tool_name}: suggests_templates must be a list of template names"]
    if not names:
        return []
    known = catalog()
    if known is None:
        return [
            f"{tool_name}: suggests_templates is set but the template catalog "
            "(lobster.figure_templates.list_templates) is not importable"
        ]
    return [
        f"{tool_name}: suggested template not in list_templates(): {n}"
        for n in names
        if n not in known
    ]


def _template_catalog():
    """Lazy, once-per-coverage-pass: ``set`` of template names, or ``None`` if absent."""
    cache: Dict[str, Any] = {}

    def get():
        if "names" not in cache:
            try:
                from lobster.figure_templates import list_templates
            except ImportError:
                cache["names"] = None
            else:
                cache["names"] = {
                    d.get("name") for d in list_templates() if isinstance(d, Mapping)
                }
        return cache["names"]

    return get


class FigureContractTestMixin:
    """Executed figure contract for an agent package's tool cases.

    Use alongside :class:`AgentContractTestMixin` (same ``agent_module`` /
    ``factory_name`` / ``factory_module`` attributes) and implement
    :meth:`figure_tool_cases`. Coverage (bounded): every factory-exposed tool
    whose PRIMARY category is one of the seven provenance-required categories needs at
    least one case or an explicit ``figure_contract_skips`` entry; DELEGATE and UTILITY
    (and any other non-provenance primary) are reported as not covered; the named trusted
    submitters are excluded and not counted; the eight transitional producers can never be
    skipped and need a saving case. ``suggests_templates`` names must resolve via
    ``list_templates()``. The existing no-buried-provenance check in
    :class:`AgentContractTestMixin` keeps the primary category honest.
    """

    agent_module: ClassVar[str]
    factory_name: ClassVar[str]
    factory_module: ClassVar[Optional[str]] = None
    #: ``{tool_name: reason}`` for covered non-producer tools without a case (reported).
    figure_contract_skips: ClassVar[Mapping[str, str]] = {}

    def figure_tool_cases(self) -> List[FigureToolCase]:
        """Return the package's :class:`FigureToolCase` list (valid invocations)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement figure_tool_cases() for figure-tool coverage"
        )

    def _figure_factory(self):
        import importlib

        module_path = self.factory_module or self.agent_module
        module = importlib.import_module(module_path)
        return getattr(module, self.factory_name), module_path

    def figure_contract_coverage(self, workspace) -> Tuple[List[str], str]:
        """``(violations, report)`` for the exposed tools vs the supplied cases.

        Inventory: every supplied case gets its own fresh context
        (real ``DataManagerV2`` at a distinct workspace, default LLM guard, ``case.setup``,
        real factory); primitive descriptors are frozen at once and united across cases.
        A same-name tool with a different key or primary category fails; each case's tool
        must be exposed in its OWN context; every observed ``suggests_templates`` value is
        validated. With no cases, one fresh empty-``DataManagerV2`` baseline is built (for
        all-skipped or zero-covered packages). The report states the observed setups; it is
        not a claim about every configuration, and not proof of execution.
        """
        from pathlib import Path

        from lobster.config import aquadif

        factory, module_path = self._figure_factory()
        cases = list(self.figure_tool_cases())
        skips = dict(self.figure_contract_skips)
        catalog = _template_catalog()
        problems: List[str] = []
        union: Dict[str, Tuple[str, Optional[str]]] = {}

        def observe(index, setup):
            ws = Path(workspace) / f"inventory_{index}"
            ws.mkdir(parents=True)
            with _fresh_case_context(factory, module_path, ws, setup) as (
                _dm,
                _mp,
                tools,
            ):
                seen = {}
                for tool in tools:
                    metadata = getattr(tool, "metadata", None) or {}
                    categories = metadata.get("categories") or []
                    primary = categories[0] if categories else None
                    seen[tool.name] = (aquadif.figure_tool_key(tool), primary)
                    for p in _template_problems(tool.name, metadata, catalog):
                        if p not in problems:
                            problems.append(p)
            return seen

        observations = (
            [(c, observe(i, c.setup)) for i, c in enumerate(cases)]
            if cases
            else [(None, observe(0, None))]
        )
        for case, seen in observations:
            if case is not None and case.tool_name not in seen:
                problems.append(
                    f"case {case.label or case.tool_name}: tool not exposed in its own setup"
                )
            for name, desc in seen.items():
                if name in union and union[name] != desc:
                    problems.append(
                        f"{name}: key or primary category differs across setups"
                    )
                union.setdefault(name, desc)

        with_cases: List[str] = []
        not_covered: List[str] = []
        excluded: List[str] = []
        skipped: List[str] = []
        for tool_name, (key, primary_value) in sorted(union.items()):
            if key in aquadif.TRUSTED_FIGURE_SUBMITTERS:
                excluded.append(tool_name)
                continue
            own = [c for c in cases if c.tool_name == tool_name]
            if aquadif.is_transitional_figure_producer(key):
                if tool_name in skips:
                    problems.append(
                        f"{tool_name}: a transitional producer cannot be skipped"
                    )
                if not any(not c.is_preview for c in own):
                    problems.append(
                        f"{tool_name}: a transitional producer needs a saving case"
                    )
                with_cases.append(tool_name)
                continue
            try:
                primary = aquadif.AquadifCategory(primary_value)
            except ValueError:
                problems.append(f"{tool_name}: no valid primary AQUADIF category")
                continue
            if primary not in aquadif.FIGURE_CONTRACT_COVERED:
                not_covered.append(f"{tool_name} ({primary.value})")
            elif own:
                with_cases.append(tool_name)
            elif tool_name in skips:
                reason = skips[tool_name]
                if not isinstance(reason, str) or not reason.strip():
                    problems.append(
                        f"{tool_name}: skip needs a non-empty string reason"
                    )
                skipped.append(f"{tool_name} ({primary.value})")  # names only, no text
            else:
                problems.append(f"{tool_name}: no case and no skip ({primary.value})")
        for tool_name in sorted(set(skips) - set(union)):
            problems.append(f"skip names an unexposed tool: {tool_name}")
        basis = (
            f"inventory across {len(cases)} supplied case setup(s)"
            if cases
            else "inventory: one empty-DataManagerV2 baseline (no cases)"
        )
        report = (
            f"D8 figure contract for {self.factory_name} ({basis}): "
            f"with cases (run by test_figure_tool_cases_hold): {with_cases or 'none'}; "
            f"skipped (reasons in source, not echoed): {skipped or 'none'}; "
            f"not covered (non-provenance primary): {not_covered or 'none'}; "
            f"excluded trusted submitter, not counted: {excluded or 'none'}."
        )
        return problems, report

    def test_figure_contract_coverage(self, tmp_path, record_property) -> None:
        import warnings

        problems, report = self.figure_contract_coverage(tmp_path / "coverage")
        record_property("d8_coverage", report)
        warnings.warn(FigureContractReport(report), stacklevel=1)
        assert not problems, (  # nosec B101 # Contract-test assertion.
            "D8 coverage:\n  - " + "\n  - ".join(problems)
        )

    def test_figure_tool_cases_hold(self, tmp_path) -> None:
        factory, module_path = self._figure_factory()
        problems: List[str] = []
        for i, case in enumerate(self.figure_tool_cases()):
            workspace = tmp_path / f"case_{i}"
            workspace.mkdir()
            problems.extend(
                check_figure_tool_case(
                    case,
                    factory=factory,
                    factory_module_path=module_path,
                    workspace=workspace,
                    tool_call_id=f"d8_call_{i}",
                )
            )
        assert not problems, (  # nosec B101 # Contract-test assertion.
            "D8 figure contract:\n  - " + "\n  - ".join(problems)
        )
