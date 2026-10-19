"""Fail-control tests for CI evidence validation (no credentials required)."""

import importlib.util
import subprocess  # nosec B404 # Invoke the local validator using fixed Python argv in a fixture.
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_test_results.py"
SPEC = importlib.util.spec_from_file_location("check_test_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestCIResults(unittest.TestCase):
    def check_xml(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_text(text)
            return MODULE.check_result(path)

    def test_executed_pass(self):
        self.assertEqual(
            self.check_xml('<testsuite><testcase name="ok"/></testsuite>'), 1
        )

    def test_partial_skip_counts_only_executed(self):
        self.assertEqual(
            self.check_xml(
                "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>"
            ),
            1,
        )

    def test_required_partial_skip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_text(
                "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>"
            )
            with self.assertRaises(ValueError):
                MODULE.check_result(path, require_no_skips=True)

    def test_empty_placeholder_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml(
                '<testsuites><testsuite tests="0" failures="0"/></testsuites>'
            )

    def test_all_skipped_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml("<testsuite><testcase><skipped/></testcase></testsuite>")

    def test_failures_and_errors_rejected(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.check_xml(f"<testsuite><testcase><{tag}/></testcase></testsuite>")

    def test_suite_collection_error_rejected(self):
        with self.assertRaises(ValueError):
            self.check_xml('<testsuite errors="1"><testcase/></testsuite>')

    def test_cli_bad_inputs_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            for text in (
                None,
                "not XML",
                "<testsuite><testcase><skipped/></testcase></testsuite>",
            ):
                if text is not None:
                    path.write_text(text)
                result = subprocess.run(  # nosec B603 # Current Python executable and fixture paths, no shell interpolation.
                    [sys.executable, str(SCRIPT), str(path)], capture_output=True
                )
                self.assertEqual(result.returncode, 1)

    def test_doctype_rejected_even_without_entities(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml("<!DOCTYPE testsuite><testsuite><testcase/></testsuite>")

    def test_internal_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'
            )

    def test_external_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY value SYSTEM "file:///nonexistent-secret">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'
            )

    def test_nested_entity_dtd_rejected(self):
        with self.assertRaisesRegex(ValueError, "DOCTYPE"):
            self.check_xml(
                '<!DOCTYPE testsuite [<!ENTITY a "x"><!ENTITY b "&a;&a;">'
                '<!ENTITY c "&b;&b;">]><testsuite><testcase name="&c;"/></testsuite>'
            )

    def test_utf16_entity_dtd_rejected(self):
        source = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
            '<testsuite><testcase name="&value;"/></testsuite>'
        )
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            with (
                self.subTest(encoding=encoding),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "result.xml"
                path.write_bytes(source.encode(encoding))
                with self.assertRaisesRegex(ValueError, "DOCTYPE"):
                    MODULE.check_result(path)

    def test_utf16_without_dtd_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_bytes(
                '<testsuite><testcase name="valid"/></testsuite>'.encode("utf-16")
            )
            self.assertEqual(MODULE.check_result(path), 1)

    def test_doctype_mention_inside_comment_is_not_a_declaration(self):
        self.assertEqual(
            self.check_xml(
                "<!-- DOCTYPE is just comment text --><testsuite><testcase/></testsuite>"
            ),
            1,
        )

    def test_hostile_xml_cli_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.xml"
            path.write_bytes(
                '<!DOCTYPE testsuite [<!ENTITY value "expanded">]>'
                '<testsuite><testcase name="&value;"/></testsuite>'.encode("utf-16")
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                capture_output=True,  # nosec B603 # Fixed local validator/fixture argv, no shell interpolation.
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"DOCTYPE", result.stderr)


EXPECTED_XML = (
    '<testsuite><testcase classname="m.T" name="test_a[x]"/>'
    '<testcase classname="m.T" name="test_a[y]"/>'
    '<testcase classname="m" name="test_b"/></testsuite>'
)


class TestExpectedTests(unittest.TestCase):
    def check(self, xml_text, expected_text):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.xml"
            result.write_text(xml_text)
            expected = Path(directory) / "expected.txt"
            expected.write_text(expected_text)
            MODULE.check_expected([result], MODULE.read_expected(expected))

    def test_exact_match_accepted(self):
        self.check(EXPECTED_XML, "# header\nm.T::test_a 2\n\nm::test_b 1\n")

    def test_missing_function_rejected(self):
        xml = EXPECTED_XML.replace('<testcase classname="m" name="test_b"/>', "")
        with self.assertRaisesRegex(ValueError, "m::test_b: expected 1, executed 0"):
            self.check(xml, "m.T::test_a 2\nm::test_b 1\n")

    def test_lost_parametrized_case_rejected(self):
        with self.assertRaisesRegex(ValueError, "m.T::test_a: expected 3, executed 2"):
            self.check(EXPECTED_XML, "m.T::test_a 3\nm::test_b 1\n")

    def test_unexpected_test_rejected(self):
        with self.assertRaisesRegex(ValueError, "m::test_b: expected 0, executed 1"):
            self.check(EXPECTED_XML, "m.T::test_a 2\n")

    def test_skipped_case_does_not_count(self):
        xml = EXPECTED_XML.replace(
            '<testcase classname="m" name="test_b"/>',
            '<testcase classname="m" name="test_b"><skipped/></testcase>',
        )
        with self.assertRaisesRegex(ValueError, "m::test_b: expected 1, executed 0"):
            self.check(xml, "m.T::test_a 2\nm::test_b 1\n")

    def test_counts_combine_across_result_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.xml"
            first.write_text(EXPECTED_XML)
            second = Path(directory) / "second.xml"
            second.write_text(
                '<testsuite><testcase classname="n" name="test_c"/></testsuite>'
            )
            MODULE.check_expected(
                [first, second],
                {"m.T::test_a": 2, "m::test_b": 1, "n::test_c": 1},
            )

    def test_malformed_or_empty_expected_list_rejected(self):
        for text in (
            "m::test_b\n",
            "m::test_b 0\n",
            "m::test_b two\n",
            "m::test_b 1\nm::test_b 1\n",
            "# only a comment\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check(EXPECTED_XML, text)

    def test_cli_writes_and_checks_expected_list(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.xml"
            result.write_text(EXPECTED_XML)
            expected = Path(directory) / "expected.txt"
            command = [sys.executable, str(SCRIPT)]
            written = subprocess.run(  # nosec B603 # Fixed local validator/fixture argv, no shell interpolation.
                [*command, "--write-expected", str(expected), str(result)],
                capture_output=True,
            )
            self.assertEqual(written.returncode, 0)
            self.assertEqual(expected.read_text(), "m.T::test_a 2\nm::test_b 1\n")
            checked = subprocess.run(  # nosec B603 # Fixed local validator/fixture argv, no shell interpolation.
                [*command, "--expect-tests", str(expected), str(result)],
                capture_output=True,
            )
            self.assertEqual(checked.returncode, 0)
            expected.write_text("m.T::test_a 2\n")
            stale = subprocess.run(  # nosec B603 # Fixed local validator/fixture argv, no shell interpolation.
                [*command, "--expect-tests", str(expected), str(result)],
                capture_output=True,
            )
            self.assertEqual(stale.returncode, 1)
            self.assertIn(b"m::test_b", stale.stderr)


if __name__ == "__main__":
    unittest.main()
