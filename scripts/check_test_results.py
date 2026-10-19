"""Reject empty, fully skipped, malformed, or failing JUnit test evidence."""

import argparse
import xml.etree.ElementTree as ET  # nosec B405 # Parser target below rejects every DOCTYPE before entity use.
from collections import Counter
from pathlib import Path


class _RejectDoctype(ET.TreeBuilder):
    """Reject DTD declarations through the parser, including encoded input."""

    def doctype(self, name, pubid, system):
        raise ValueError("DOCTYPE declarations are prohibited in JUnit evidence")


def _parse(path: Path):
    parser = ET.XMLParser(
        target=_RejectDoctype()
    )  # nosec B314 # Parser target rejects DTDs/entities, including UTF-16 input.
    return ET.parse(
        path, parser=parser
    ).getroot()  # nosec B314 # DOCTYPE-rejecting parser target; entities/DTDs cannot be used.


def executed_counts(path: Path) -> Counter:
    """Count executed (not skipped) cases per ``classname::function``.

    Parametrize suffixes (``[...]``) are dropped: their text can depend on
    library reprs, but the number of cases per test function must not change.
    """
    return Counter(
        f"{case.get('classname', '')}::{case.get('name', '').split('[', 1)[0]}"
        for case in _parse(path).iter("testcase")
        if case.find("skipped") is None
    )


def read_expected(path: Path) -> Counter:
    """Read ``<classname::function> <count>`` lines; blanks and ``#`` comments ignored."""
    expected = Counter()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        test_id, _, count = line.rpartition(" ")
        if not test_id or not count.isdigit() or int(count) < 1 or test_id in expected:
            raise ValueError(f"{path}:{number}: expected '<test id> <count>'")
        expected[test_id] = int(count)
    return expected


def check_expected(results, expected: Counter) -> None:
    """Raise ValueError unless the executed tests are exactly ``expected``.

    A missing or smaller count means a test was deleted, renamed, deselected or
    skipped; an unexpected or larger count means the expected list is stale.
    Either way the selection changed without review, so the evidence fails.
    """
    if not expected:
        raise ValueError("expected test list is empty")
    executed = Counter()
    for path in results:
        executed.update(executed_counts(path))
    details = [
        f"{test_id}: expected {expected.get(test_id, 0)}, executed {executed.get(test_id, 0)}"
        for test_id in sorted(set(expected) | set(executed))
        if expected.get(test_id, 0) != executed.get(test_id, 0)
    ]
    if details:
        raise ValueError(
            "executed tests differ from the expected list:\n  " + "\n  ".join(details)
        )


def check_result(path: Path, require_no_skips: bool = False) -> int:
    """Return executed test count; raise ValueError for unusable evidence."""
    root = _parse(path)
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError(f"{path}: no test cases")
    if any(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    ):
        raise ValueError(f"{path}: failing test cases")
    # Also reject suite-level errors, which pytest can emit during collection.
    for suite in root.iter("testsuite"):
        if int(suite.get("failures", "0")) or int(suite.get("errors", "0")):
            raise ValueError(f"{path}: suite failures or collection errors")
    executed = sum(case.find("skipped") is None for case in cases)
    if not executed:
        raise ValueError(f"{path}: all test cases skipped")
    if require_no_skips and executed != len(cases):
        raise ValueError(f"{path}: required test cases skipped")
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--require-no-skips", action="store_true")
    parser.add_argument(
        "--expect-tests",
        type=Path,
        help="file of '<classname::function> <count>' lines the results must match exactly",
    )
    parser.add_argument(
        "--write-expected",
        type=Path,
        help="write the executed test IDs of the results to this file and exit",
    )
    args = parser.parse_args()
    try:
        if args.write_expected:
            counts = Counter()
            for path in args.results:
                counts.update(executed_counts(path))
            args.write_expected.write_text(
                "".join(f"{item} {counts[item]}\n" for item in sorted(counts)),
                encoding="utf-8",
            )
            print(f"{args.write_expected}: {len(counts)} expected test functions")
            return
        for path in args.results:
            executed = check_result(path, require_no_skips=args.require_no_skips)
            print(f"{path}: {executed} executed test cases")
        if args.expect_tests:
            check_expected(args.results, read_expected(args.expect_tests))
            print(f"{args.expect_tests}: executed tests match the expected list")
    except (OSError, ET.ParseError, ValueError) as exc:
        parser.exit(1, f"Invalid test evidence: {exc}\n")


if __name__ == "__main__":
    main()
