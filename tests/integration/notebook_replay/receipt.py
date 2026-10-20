"""Environment receipt for the notebook replay lane (standard library only).

The same code runs in the live capture process and, as a notebook cell, in the
kernel that executes the exported notebook. Equal receipts show that the
notebook ran in the environment the live session ran in.
"""

import hashlib
import json

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

# Fields that describe the environment. Equal fields mean the same interpreter,
# the same installed distributions and the same thread settings.
ENVIRONMENT_FIELDS = (
    "executable",
    "version",
    "prefix",
    "base_prefix",
    "distributions_sha256",
    "distributions",
    "thread_env",
    "pythonpath",
    "pythonhome",
    "user_site",
)

# Source of a function body that runs unchanged in both places. It defines
# ``collect_receipt()`` and nothing else at module level.
RECEIPT_CODE = """
def collect_receipt():
    import hashlib as _hashlib
    import importlib.metadata as _metadata
    import os as _os
    import re as _re
    import site as _site
    import sys as _sys

    _names = {}
    for _dist in _metadata.distributions():
        _name = _re.sub(r"[-_.]+", "-", _dist.metadata["Name"]).lower()
        _names[_name] = _dist.version
    _listing = "\\n".join(f"{n}=={v}" for n, v in sorted(_names.items()))
    _threads = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "PYTHONHASHSEED",
    )
    return {
        "executable": _sys.executable,
        "version": _sys.version,
        "prefix": _sys.prefix,
        "base_prefix": _sys.base_prefix,
        "distributions_sha256": _hashlib.sha256(_listing.encode()).hexdigest(),
        "distributions": len(_names),
        "thread_env": {_k: _os.environ.get(_k) for _k in _threads},
        "pythonpath": _os.environ.get("PYTHONPATH"),
        "pythonhome": _os.environ.get("PYTHONHOME"),
        "user_site": bool(_site.ENABLE_USER_SITE),
    }
"""


def collect_receipt():
    """Receipt of the current process, from the same code a kernel runs."""
    scope = {}
    exec(compile(RECEIPT_CODE, "<receipt>", "exec"), scope)  # nosec B102
    return scope["collect_receipt"]()


def environment_digest(receipt):
    """SHA-256 over the environment fields of a receipt."""
    fields = {key: receipt[key] for key in ENVIRONMENT_FIELDS}
    text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_peak_bytes():
    """Peak memory of this container's whole process tree, if cgroup v2 says."""
    for path in (
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory.max_usage_in_bytes",
    ):
        try:
            with open(path) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return None
