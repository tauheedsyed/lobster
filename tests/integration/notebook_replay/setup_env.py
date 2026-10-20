"""Verify a fresh replay environment against the lock, then describe it.

Runs with the new environment's interpreter after installation. Every
installed distribution must be pinned to exactly that version in the
constraints exported from ``uv.lock``, except the distributions built from
this source tree, which must carry the source tree's version. Anything else
(an unlocked package, a different version, a published wheel standing in for
a source package) fails the setup.

Writes env.json: the receipt fields of the interpreter and the sorted
distribution listing. ``--unverified`` (the unisolated dev lane) skips the
lock check and says so in env.json.

Usage: python setup_env.py --constraints C --source-wheels DIR --out env.json
       python setup_env.py --unverified --out env.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

from receipt import collect_receipt, memory_peak_bytes

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        match = _PIN.match(line.strip())
        if match:
            name, version = normalize(match.group(1)), match.group(2)
            pins.setdefault(name, set()).add(version)
    return pins


def installed():
    import importlib.metadata as metadata

    found = {}
    for dist in metadata.distributions():
        name = normalize(dist.metadata["Name"])
        if name in found:
            raise SystemExit(f"distribution installed twice: {name}")
        found[name] = dist.version
    return found


def check(found, pins, source_dists):
    problems = []
    for name, version in sorted(found.items()):
        if name in source_dists:
            if version != source_dists[name]:
                problems.append(
                    f"{name}=={version} is not the source tree's {source_dists[name]}"
                )
        elif name not in pins:
            problems.append(f"{name}=={version} is not in the lock")
        elif version not in pins[name]:
            problems.append(
                f"{name}=={version} differs from the lock {sorted(pins[name])}"
            )
    missing = sorted(set(source_dists) - set(found))
    problems.extend(f"{name} from the source tree is not installed" for name in missing)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--constraints")
    parser.add_argument("--source-wheels", type=Path)
    parser.add_argument("--unverified", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    found = installed()
    source_dists = {}
    if args.unverified:
        if args.constraints or args.source_wheels:
            parser.error("--unverified takes no constraints or wheels")
    else:
        if not (args.constraints and args.source_wheels):
            parser.error("--constraints and --source-wheels are required")
        for wheel in sorted(args.source_wheels.glob("*.whl")):
            name, version = wheel.name.split("-")[:2]
            source_dists[normalize(name)] = version
        if not source_dists:
            raise SystemExit(f"no source wheels in {args.source_wheels}")
        problems = check(found, read_pins(args.constraints), source_dists)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            raise SystemExit(
                f"environment does not match the lock ({len(problems)} problems)"
            )
    record = {
        "verified_against_lock": not args.unverified,
        "receipt": collect_receipt(),
        "distributions": [
            f"{name}=={version}" for name, version in sorted(found.items())
        ],
        "source_distributions": source_dists,
        "memory_peak_bytes": memory_peak_bytes(),
    }
    args.out.write_text(json.dumps(record, sort_keys=True, indent=1))
    if args.unverified:
        print(f"environment NOT verified against the lock: {len(found)} distributions")
    else:
        print(f"environment matches the lock: {len(found)} distributions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
