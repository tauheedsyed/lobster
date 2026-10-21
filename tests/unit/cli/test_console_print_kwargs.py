"""Rich ``Console.print`` has no ``flush`` argument.

Classic chat with ``--no-stream`` once called ``console.print(..., flush=True)``
and every question failed with a TypeError before reaching the agents.
"""

import ast
import inspect
from pathlib import Path

from rich.console import Console

CLI = Path(__file__).resolve().parents[3] / "lobster" / "cli_internal"


def test_console_print_calls_use_only_rich_keywords():
    allowed = set(inspect.signature(Console.print).parameters)
    bad = []
    for path in sorted(CLI.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "print"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "console"
            ):
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in allowed:
                        bad.append(f"{path.name}:{node.lineno} {keyword.arg}=")
    assert bad == []
