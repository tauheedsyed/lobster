"""Tests for plugin validation checks."""

from lobster.scaffold import scaffold_agent
from lobster.scaffold.validators import validate_plugin


class TestValidatePlugin:
    """Test validate_plugin on scaffolded packages."""

    def test_scaffold_passes_all_checks(self, tmp_path):
        """A freshly scaffolded package should pass all validation checks."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test agent",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        failures = [r for r in results if not r.passed]
        assert not failures, "Scaffolded package should pass all checks:\n" + "\n".join(
            f"  FAIL [{r.check}] {r.message}" for r in failures
        )

    def test_pep420_violation_detected(self, tmp_path):
        """Detect PEP 420 violations (extra __init__.py)."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        # Create forbidden __init__.py
        (pkg_dir / "lobster" / "__init__.py").write_text("# bad")

        results = validate_plugin(pkg_dir)
        pep420_results = [r for r in results if r.check == "PEP 420 compliance"]
        assert any(not r.passed for r in pep420_results)

    def test_entry_points_validated(self, tmp_path):
        """Entry points must end with :AGENT_CONFIG."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        ep_results = [r for r in results if r.check == "Entry points"]
        assert any(r.passed for r in ep_results)

    def test_agent_config_position_validated(self, tmp_path):
        """AGENT_CONFIG position check works."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        config_results = [r for r in results if r.check == "AGENT_CONFIG position"]
        assert any(r.passed for r in config_results)

    def test_factory_signature_validated(self, tmp_path):
        """Factory signature check works."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        factory_results = [r for r in results if r.check == "Factory signature"]
        assert any(r.passed for r in factory_results)

    def test_aquadif_metadata_validated(self, tmp_path):
        """AQUADIF metadata check works."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        aquadif_results = [r for r in results if r.check == "AQUADIF metadata"]
        assert any(r.passed for r in aquadif_results)

    def test_provenance_calls_validated(self, tmp_path):
        """Provenance call check uses has_provenance_call from aquadif.py (Delta 1)."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        prov_results = [r for r in results if r.check == "Provenance calls"]
        assert any(r.passed for r in prov_results)

    def test_import_boundaries_validated(self, tmp_path):
        """Import boundary check works."""
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        import_results = [r for r in results if r.check == "Import boundaries"]
        assert any(r.passed for r in import_results)

    def test_children_scaffold_passes(self, tmp_path):
        """Package with children should also pass all checks."""
        pkg_dir = scaffold_agent(
            name="epigenomics_expert",
            display_name="Epigenomics Expert",
            description="Epigenomics analysis",
            tier="free",
            children=["methylation_expert", "chromatin_expert"],
            output_dir=tmp_path,
        )
        results = validate_plugin(pkg_dir)
        failures = [r for r in results if not r.passed]
        assert (
            not failures
        ), "Scaffolded package with children should pass all checks:\n" + "\n".join(
            f"  FAIL [{r.check}] {r.message}" for r in failures
        )


class TestHollowProvenanceTransition:
    """`ir=None` was a documented bridge: validate-plugin warns before failing."""

    @staticmethod
    def _provenance_results(tmp_path, transform):
        pkg_dir = scaffold_agent(
            name="test_expert",
            display_name="Test Expert",
            description="Test",
            tier="free",
            output_dir=tmp_path,
        )
        shared = next(pkg_dir.rglob("shared_tools.py"))
        shared.write_text(transform(shared.read_text()))
        results = validate_plugin(pkg_dir)
        return [r for r in results if r.check == "Provenance calls"]

    def test_only_ir_none_is_a_warning_not_a_failure(self, tmp_path):
        results = self._provenance_results(
            tmp_path, lambda src: src.replace("ir=ir,", "ir=None,")
        )
        assert results and all(r.passed for r in results)
        assert any(
            r.severity == "warning" and "ir=None" in r.message for r in results
        )

    def test_no_ir_keyword_still_fails(self, tmp_path):
        results = self._provenance_results(
            tmp_path, lambda src: src.replace("ir=ir,", "")
        )
        assert any(not r.passed for r in results)

    def test_real_ir_passes_without_warning(self, tmp_path):
        results = self._provenance_results(tmp_path, lambda src: src)
        assert results and all(r.passed and r.severity == "error" for r in results)
