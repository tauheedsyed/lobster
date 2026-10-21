"""Minimal environment shared by tool-driven child processes."""

import os


def minimal_child_env(*, home_default: str = "") -> dict[str, str]:
    """Return the established custom-code allowlist, without ambient credentials."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", home_default),
        "USER": os.environ.get("USER", ""),
        "TMPDIR": os.environ.get(
            "TMPDIR", "/tmp"
        ),  # nosec B108 # Child environment fallback only; nothing is created here.
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", ""),
        # Empty PYTHONPATH preserves the custom-code import-path policy.
        "PYTHONPATH": "",
        "PYTHONHOME": os.environ.get("PYTHONHOME", ""),
    }
