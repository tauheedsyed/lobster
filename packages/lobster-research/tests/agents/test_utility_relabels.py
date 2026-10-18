"""Research tools that only touch the publication queue or metadata cache are UTILITY.

They change no analysis data, so they declare no provenance requirement.
"""

import pytest

from lobster.agents.research.research_agent import research_agent
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools

UTILITY = {"categories": ["UTILITY"], "provenance": False}


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_dataset_metadata",
        "validate_dataset_metadata",
        "process_publication_entry",
        "process_publication_queue",
    ],
)
def test_queue_and_metadata_tools_are_utility(tool_name, tmp_path):
    tools = factory_tools(
        research_agent,
        "lobster.agents.research.research_agent",
        RecordingDataManager(workspace_path=tmp_path),
    )
    assert tools[tool_name].metadata == UTILITY
    assert tools[tool_name].tags == ["UTILITY"]
