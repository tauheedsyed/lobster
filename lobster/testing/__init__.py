"""
Lobster Testing Module - Shared testing utilities for all packages.

This module provides consistent mock objects, fixtures, and contract test
mixins that all agent packages and integration tests can import. It eliminates
duplication and ensures workspace_path is always Path (not str) per Phase 5
decision 05-02.

Public API:
    - MockDataManager: Mock DataManagerV2 with in-memory behavior
    - MockLLM: Mock LLM with configurable responses
    - MockProvenanceTracker: Mock provenance tracker for IR
    - AgentContractTestMixin: Test mixin for plugin API compliance validation

Example - Using mocks:
    >>> from lobster.testing import MockDataManager, MockLLM
    >>> from pathlib import Path
    >>>
    >>> # Create mock data manager
    >>> dm = MockDataManager(workspace_path=Path('/tmp/test'))
    >>> dm.add_modality('test', adata)
    >>>
    >>> # Create mock LLM with configurable responses
    >>> llm = MockLLM(default_response='Analysis complete')
    >>> llm.set_response_sequence(['First', 'Second', 'Third'])

Example - Contract testing:
    >>> from lobster.testing import AgentContractTestMixin
    >>>
    >>> class TestMyAgent(AgentContractTestMixin):
    ...     agent_module = 'my_package.my_agent'
    ...     factory_name = 'my_agent'
    ...
    >>> # Run with pytest - validates plugin API compliance and AQUADIF metadata
"""

import importlib

# Exports load on first use, so a light submodule (for example
# ``lobster.testing.replay_contract``) can be imported without pulling in the
# scientific stack through the mocks and mixins below.
_EXPORTS = {
    "AgentContractTestMixin": "lobster.testing.contract_mixins",
    "create_test_workspace": "lobster.testing.fixtures",
    "MockDataManager": "lobster.testing.mock_data_manager",
    "MockProvenanceTracker": "lobster.testing.mock_data_manager",
    "MockLLM": "lobster.testing.mock_llm",
    "MockLLMResponse": "lobster.testing.mock_llm",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    # Mock objects
    "MockDataManager",
    "MockLLM",
    "MockLLMResponse",
    "MockProvenanceTracker",
    # Contract test mixins
    "AgentContractTestMixin",
    # Factory functions
    "create_test_workspace",
]
