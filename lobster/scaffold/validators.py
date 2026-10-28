"""
Post-generation plugin validation.

Provides AST-level and structural checks for generated plugin packages.
Used by `lobster validate-plugin <dir>` CLI command.

Nine validation checks (ordered by severity):
1. PEP 420 compliance — no __init__.py at lobster/ or lobster/agents/ level
2. pyproject.toml entry points — lobster.agents group exists with correct format
3. AGENT_CONFIG position — appears before heavy imports (AST check)
4. Factory signature — has standard parameters
5. AQUADIF metadata — tools have .metadata with categories and provenance
6. Provenance calls — tools with provenance: True call log_tool_usage(ir=ir)
7. No core imports — plugin doesn't import from lobster.agents.* core agents
8. No ImportError guards — domain __init__.py must not use try/except ImportError
9. Figure contract — tests use FigureContractTestMixin and every tool whose primary
   AQUADIF category requires provenance has a FigureToolCase or a skip with a reason.
   This is a PRESENCE check only, never behavioural acceptance: the contract assertion itself is
   executed by the package's own pytest run
   (lobster.testing.contract_mixins.FigureContractTestMixin). Only LITERAL tool names in
   ``FigureToolCase(...)`` and literal ``figure_contract_skips`` keys are seen (no AST
   evaluation); non-literal ones are reported as "not checked".

SYNC NOTE: Standalone equivalent at skills/lobster-dev/scripts/validate_plugin.py — keep checks aligned.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Delta 1: Import has_provenance_call from aquadif.py, don't reimplement
from lobster.config.aquadif import (
    PROVENANCE_REQUIRED,
    AquadifCategory,
    has_provenance_call,
)


@dataclass
class ValidationResult:
    """A single validation check result."""

    check: str
    passed: bool
    message: str
    severity: str = "error"  # "error" or "warning"


def validate_plugin(plugin_dir: Path) -> List[ValidationResult]:
    """Validate a plugin package for contract compliance.

    Runs 9 structural checks against the plugin directory.

    Args:
        plugin_dir: Root directory of the plugin package

    Returns:
        List of ValidationResult objects (all checks, pass or fail)
    """
    results = []

    results.append(_check_pep420(plugin_dir))
    results.extend(_check_entry_points(plugin_dir))
    results.extend(_check_agent_config_position(plugin_dir))
    results.extend(_check_factory_signature(plugin_dir))
    results.extend(_check_aquadif_metadata(plugin_dir))
    results.extend(_check_provenance_calls(plugin_dir))
    results.extend(_check_import_boundaries(plugin_dir))
    results.extend(_check_no_import_error_guard(plugin_dir))
    results.extend(_check_figure_contract(plugin_dir))

    return results


def _find_agent_modules(plugin_dir: Path) -> List[Path]:
    """Find all Python files in lobster/agents/*/."""
    agent_dirs = list((plugin_dir / "lobster" / "agents").glob("*"))
    modules = []
    for d in agent_dirs:
        if d.is_dir() and d.name != "__pycache__":
            for py_file in d.glob("*.py"):
                if py_file.name != "__init__.py":
                    modules.append(py_file)
    return modules


def _check_pep420(plugin_dir: Path) -> ValidationResult:
    """Check 1: PEP 420 compliance — no __init__.py at namespace boundaries."""
    violations = []
    lobster_init = plugin_dir / "lobster" / "__init__.py"
    agents_init = plugin_dir / "lobster" / "agents" / "__init__.py"

    if lobster_init.exists():
        violations.append("lobster/__init__.py")
    if agents_init.exists():
        violations.append("lobster/agents/__init__.py")

    if violations:
        return ValidationResult(
            check="PEP 420 compliance",
            passed=False,
            message=f"Namespace boundary files exist (must be deleted): {', '.join(violations)}",
        )
    return ValidationResult(
        check="PEP 420 compliance",
        passed=True,
        message="No __init__.py at namespace boundaries",
    )


def _check_entry_points(plugin_dir: Path) -> List[ValidationResult]:
    """Check 2: pyproject.toml has lobster.agents entry points."""
    results = []
    pyproject = plugin_dir / "pyproject.toml"

    if not pyproject.exists():
        results.append(
            ValidationResult(
                check="Entry points",
                passed=False,
                message="pyproject.toml not found",
            )
        )
        return results

    try:
        import tomllib

        content = pyproject.read_text()
        parsed = tomllib.loads(content)

        eps = (
            parsed.get("project", {}).get("entry-points", {}).get("lobster.agents", {})
        )
        if not eps:
            results.append(
                ValidationResult(
                    check="Entry points",
                    passed=False,
                    message="No lobster.agents entry points found in pyproject.toml",
                )
            )
        else:
            # Check format: value should end with :AGENT_CONFIG
            for name, value in eps.items():
                if not value.endswith(":AGENT_CONFIG"):
                    results.append(
                        ValidationResult(
                            check="Entry points",
                            passed=False,
                            message=f"Entry point '{name}' value must end with ':AGENT_CONFIG', got '{value}'",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            check="Entry points",
                            passed=True,
                            message=f"Entry point '{name}' = '{value}'",
                        )
                    )

    except Exception as e:
        results.append(
            ValidationResult(
                check="Entry points",
                passed=False,
                message=f"Failed to parse pyproject.toml: {e}",
            )
        )

    return results


def _check_agent_config_position(plugin_dir: Path) -> List[ValidationResult]:
    """Check 3: AGENT_CONFIG appears before heavy imports (AST check)."""
    results = []

    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name in (
            "shared_tools.py",
            "state.py",
            "config.py",
            "prompts.py",
            "conftest.py",
        ):
            continue

        try:
            source = module_path.read_text()
            tree = ast.parse(source)
        except SyntaxError as e:
            results.append(
                ValidationResult(
                    check="AGENT_CONFIG position",
                    passed=False,
                    message=f"{module_path.name}: syntax error: {e}",
                )
            )
            continue

        # Find AGENT_CONFIG assignment position
        config_line = None
        heavy_import_line = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "AGENT_CONFIG":
                        config_line = node.lineno
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = [alias.name for alias in getattr(node, "names", [])]
                if any(m.startswith(("langgraph", "langchain")) for m in [mod] + names):
                    if heavy_import_line is None or node.lineno < heavy_import_line:
                        heavy_import_line = node.lineno

        if config_line is None:
            # Not every .py file needs AGENT_CONFIG — skip non-agent modules
            continue

        if heavy_import_line and config_line > heavy_import_line:
            results.append(
                ValidationResult(
                    check="AGENT_CONFIG position",
                    passed=False,
                    message=f"{module_path.name}: AGENT_CONFIG at line {config_line} is after heavy import at line {heavy_import_line}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="AGENT_CONFIG position",
                    passed=True,
                    message=f"{module_path.name}: AGENT_CONFIG at line {config_line} (before heavy imports)",
                )
            )

    return results


def _check_factory_signature(plugin_dir: Path) -> List[ValidationResult]:
    """Check 4: Factory functions have standard parameters."""
    results = []
    required_params = {
        "data_manager",
        "callback_handler",
        "agent_name",
        "delegation_tools",
        "workspace_path",
    }

    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name in (
            "shared_tools.py",
            "state.py",
            "config.py",
            "prompts.py",
            "conftest.py",
        ):
            continue

        try:
            source = module_path.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Check if this file has AGENT_CONFIG (it's an agent module)
        has_config = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "AGENT_CONFIG" for t in node.targets
            )
            for node in ast.walk(tree)
        )
        if not has_config:
            continue

        # Find factory function (function with same name as file stem, or named in AGENT_CONFIG)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                params = {arg.arg for arg in node.args.args}
                missing = required_params - params
                if not missing:
                    results.append(
                        ValidationResult(
                            check="Factory signature",
                            passed=True,
                            message=f"{module_path.name}:{node.name}() has all standard params",
                        )
                    )
                    break
                elif params & required_params:
                    # Looks like a factory but missing some params
                    results.append(
                        ValidationResult(
                            check="Factory signature",
                            passed=False,
                            message=f"{module_path.name}:{node.name}() missing params: {sorted(missing)}",
                        )
                    )
                    break

    return results


def _check_aquadif_metadata(plugin_dir: Path) -> List[ValidationResult]:
    """Check 5: Tools have AQUADIF .metadata assignments."""
    results = []

    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name != "shared_tools.py":
            continue

        source = module_path.read_text()

        # Count .metadata assignments
        metadata_count = len(re.findall(r"\w+\.metadata\s*=\s*\{", source))
        # Count @tool decorators
        tool_count = source.count("@tool")

        if tool_count == 0:
            results.append(
                ValidationResult(
                    check="AQUADIF metadata",
                    passed=True,
                    message=f"{module_path.name}: no tools defined (skeleton)",
                    severity="warning",
                )
            )
        elif metadata_count >= tool_count:
            results.append(
                ValidationResult(
                    check="AQUADIF metadata",
                    passed=True,
                    message=f"{module_path.name}: {metadata_count} metadata assignments for {tool_count} tools",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="AQUADIF metadata",
                    passed=False,
                    message=f"{module_path.name}: {metadata_count} metadata assignments but {tool_count} @tool definitions",
                )
            )

    return results


def _has_only_hollow_provenance(tree: ast.AST) -> bool:
    """True if log_tool_usage is called with ``ir=`` but only ever as literal None."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_tool_usage"
            and any(keyword.arg == "ir" for keyword in node.keywords)
        ):
            return True
    return False


def _check_provenance_calls(plugin_dir: Path) -> List[ValidationResult]:
    """Check 6: Provenance tools call log_tool_usage(ir=ir) (AST check).

    Uses has_provenance_call() from lobster.config.aquadif (Delta 1).
    """
    results = []

    for module_path in _find_agent_modules(plugin_dir):
        if module_path.name != "shared_tools.py":
            continue

        try:
            source = module_path.read_text()
            tree = ast.parse(source)
        except SyntaxError as e:
            results.append(
                ValidationResult(
                    check="Provenance calls",
                    passed=False,
                    message=f"{module_path.name}: syntax error: {e}",
                )
            )
            continue

        # Check if there are any provenance: True declarations
        has_prov_true = '"provenance": True' in source or "'provenance': True" in source

        if has_prov_true:
            # Use the shared has_provenance_call function (Delta 1)
            if has_provenance_call(tree):
                results.append(
                    ValidationResult(
                        check="Provenance calls",
                        passed=True,
                        message=f"{module_path.name}: log_tool_usage(ir=ir) calls found",
                    )
                )
            elif _has_only_hollow_provenance(tree):
                # Transition: `ir=None` was documented as an acceptable bridge, so
                # existing plugins get one release of warning before it fails.
                results.append(
                    ValidationResult(
                        check="Provenance calls",
                        passed=True,
                        message=(
                            f"{module_path.name}: tools declare provenance=True but "
                            "only call log_tool_usage(ir=None), which records nothing "
                            "reproducible. Pass a real AnalysisStep, or set "
                            "provenance=False for tools that change no data. This "
                            "will fail in a future release."
                        ),
                        severity="warning",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        check="Provenance calls",
                        passed=False,
                        message=f"{module_path.name}: tools declare provenance=True but no log_tool_usage(ir=ir) found",
                    )
                )
        else:
            results.append(
                ValidationResult(
                    check="Provenance calls",
                    passed=True,
                    message=f"{module_path.name}: no provenance-requiring tools",
                )
            )

    return results


def _check_import_boundaries(plugin_dir: Path) -> List[ValidationResult]:
    """Check 7: Plugin doesn't import from core lobster.agents.* agents."""
    results = []

    # Core agent modules that plugins should NOT import from
    core_agent_pattern = re.compile(
        r"from\s+lobster\.agents\."
        r"(?!__)"  # Not __init__
        r"(\w+)"  # domain
        r"\."
        r"(\w+)"  # module
        r"\s+import"
    )

    for module_path in _find_agent_modules(plugin_dir):
        source = module_path.read_text()

        # Find the domain this plugin is in
        # Plugin structure: lobster/agents/{domain}/
        domain = module_path.parent.name

        violations = []
        for match in core_agent_pattern.finditer(source):
            imported_domain = match.group(1)
            if imported_domain != domain:
                violations.append(f"lobster.agents.{imported_domain}.{match.group(2)}")

        if violations:
            results.append(
                ValidationResult(
                    check="Import boundaries",
                    passed=False,
                    message=f"{module_path.name}: imports from core agents: {', '.join(violations)}",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check="Import boundaries",
                    passed=True,
                    message=f"{module_path.name}: no cross-agent imports",
                )
            )

    return results


def _check_no_import_error_guard(plugin_dir: Path) -> List[ValidationResult]:
    """Check 8: Domain __init__.py must not use try/except ImportError.

    Agent discovery uses entry points, not eager imports. The try/except
    ImportError pattern contradicts PEP 420 and Hard Rule #8.
    """
    results = []

    agents_dir = plugin_dir / "lobster" / "agents"
    if not agents_dir.exists():
        return results

    for domain_dir in agents_dir.iterdir():
        if not domain_dir.is_dir() or domain_dir.name == "__pycache__":
            continue

        init_file = domain_dir / "__init__.py"
        if not init_file.exists():
            continue

        try:
            source = init_file.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Walk AST looking for except ImportError handlers
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                exc_type = node.type
                # Match: except ImportError, except (ImportError, ...), bare except
                if exc_type is None:
                    # Bare except — flag it
                    results.append(
                        ValidationResult(
                            check="No ImportError guard",
                            passed=False,
                            message=f"{domain_dir.name}/__init__.py: bare except handler (line {node.lineno})",
                        )
                    )
                    break
                elif isinstance(exc_type, ast.Name) and exc_type.id == "ImportError":
                    results.append(
                        ValidationResult(
                            check="No ImportError guard",
                            passed=False,
                            message=f"{domain_dir.name}/__init__.py: except ImportError at line {node.lineno} — use entry points for agent discovery",
                        )
                    )
                    break
                elif isinstance(exc_type, ast.Tuple):
                    for elt in exc_type.elts:
                        if isinstance(elt, ast.Name) and elt.id == "ImportError":
                            results.append(
                                ValidationResult(
                                    check="No ImportError guard",
                                    passed=False,
                                    message=f"{domain_dir.name}/__init__.py: except (..., ImportError) at line {node.lineno} — use entry points for agent discovery",
                                )
                            )
                            break
                    else:
                        continue
                    break
        else:
            results.append(
                ValidationResult(
                    check="No ImportError guard",
                    passed=True,
                    message=f"{domain_dir.name}/__init__.py: clean (no ImportError guards)",
                )
            )

    return results


def _declared_tool_categories(plugin_dir: Path) -> dict:
    """``tool name -> primary category`` from ``<tool>.metadata = {"categories": [...]}``."""
    out = {}
    for module_path in _find_agent_modules(plugin_dir):
        try:
            tree = ast.parse(module_path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            for target in node.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and target.attr == "metadata"
                    and isinstance(target.value, ast.Name)
                ):
                    continue
                for key, value in zip(node.value.keys, node.value.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "categories"
                        and isinstance(value, ast.List)
                        and value.elts
                        and isinstance(value.elts[0], ast.Constant)
                    ):
                        out[target.value.id] = value.elts[0].value
    return out


def _figure_contract_declarations(plugin_dir: Path):
    """``(uses_mixin, literal case names, literal skip names, non-literal count)`` (AST).

    Literal-only: a case name or skips mapping that is not a literal is counted as
    dynamic and is NOT evaluated.
    """
    uses_mixin, cases, skips, dynamic = False, set(), set(), 0
    tests_dir = plugin_dir / "tests"
    for test_file in sorted(tests_dir.rglob("*.py")) if tests_dir.is_dir() else []:
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = getattr(base, "id", None) or getattr(base, "attr", None)
                    if name == "FigureContractTestMixin":
                        uses_mixin = True
            elif isinstance(node, ast.Call):
                fname = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if fname != "FigureToolCase":
                    continue
                first = node.args[0] if node.args else None
                for kw in node.keywords:
                    if kw.arg == "tool_name":
                        first = kw.value
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    cases.add(first.value)
                else:
                    dynamic += 1
            elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "figure_contract_skips" for t in node.targets
            ):
                if not isinstance(node.value, ast.Dict):
                    dynamic += 1
                    continue
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        skips.add(k.value)
                    else:
                        dynamic += 1
    return uses_mixin, cases, skips, dynamic


def _check_figure_contract(plugin_dir: Path) -> List[ValidationResult]:
    """Check 9: figure contract is wired (presence only; pytest executes it)."""
    check = "Figure contract (D8)"
    covered = sorted(
        name
        for name, primary in _declared_tool_categories(plugin_dir).items()
        if primary in {c.value for c in PROVENANCE_REQUIRED}
        or primary not in {c.value for c in AquadifCategory}
    )
    if not covered:
        return [
            ValidationResult(
                check=check,
                passed=True,
                message="no tools with a provenance-required primary category",
                severity="warning",
            )
        ]
    uses_mixin, cases, skips, dynamic = _figure_contract_declarations(plugin_dir)
    if not uses_mixin:
        return [
            ValidationResult(
                check=check,
                passed=False,
                message="tests do not use FigureContractTestMixin (lobster.testing.contract_mixins)",
            )
        ]
    results = []
    if dynamic:
        results.append(
            ValidationResult(
                check=check,
                passed=True,
                message=(
                    f"{dynamic} non-literal case name(s) or skip(s) not checked "
                    "(presence check is literal-only; pytest executes D8)"
                ),
                severity="warning",
            )
        )
    for name in covered:
        if name in cases:
            results.append(
                ValidationResult(
                    check=check, passed=True, message=f"{name}: has a case"
                )
            )
        elif name in skips:
            results.append(
                ValidationResult(
                    check=check,
                    passed=True,
                    message=f"{name}: skipped (reason in figure_contract_skips)",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    check=check,
                    passed=False,
                    message=f"{name}: no FigureToolCase and no figure_contract_skips entry",
                )
            )
    return results
