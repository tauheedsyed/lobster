"""lookup_variant queries Ensembl for one variant and stores nothing: a UTILITY."""

from lobster.agents.genomics.variant_analysis_expert import variant_analysis_expert
from lobster.testing.provenance_replay import RecordingDataManager, factory_tools


def test_lookup_variant_is_utility(tmp_path):
    tools = factory_tools(
        variant_analysis_expert,
        "lobster.agents.genomics.variant_analysis_expert",
        RecordingDataManager(workspace_path=tmp_path),
    )
    assert tools["lookup_variant"].metadata == {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    assert tools["lookup_variant"].tags == ["UTILITY"]
