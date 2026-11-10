#!/usr/bin/env python3

import base64
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCANNER_PATH = Path(__file__).with_name("ast_security_scanner.py")
SPEC = importlib.util.spec_from_file_location("ast_security_scanner", SCANNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load scanner from {SCANNER_PATH}")
SCANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCANNER
SPEC.loader.exec_module(SCANNER)


class ASTSecurityScannerTests(unittest.TestCase):
    def test_project_allowlist_is_loaded_from_security_tool_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            allowlist_path = (
                Path(temp_dir) / "tools" / "security" / "ast-scan-allowlist.txt"
            )
            allowlist_path.parent.mkdir(parents=True)
            allowlist_path.write_text("custom.example\n", encoding="utf-8")

            SCANNER._load_url_allowlist(Path(temp_dir))

        self.assertIn("custom.example", SCANNER._url_allowlist)

    def test_own_source_detection_handles_mirrored_checkout(self):
        self.assertTrue(SCANNER._is_own_source(str(SCANNER_PATH)))
        self.assertTrue(
            SCANNER._is_own_source(
                "/tmp/pr-code/tools/security/ast_security_scanner.py"
            )
        )
        self.assertTrue(
            SCANNER._is_own_source("/tmp/pr-code/tools/security/scan_malicious_code.py")
        )
        self.assertFalse(SCANNER._is_own_source("/tmp/pr-code/src/application.py"))

    def test_metadata_service_patterns_keep_detection_values(self):
        self.assertEqual(SCANNER.METADATA_SERVICE_IP, "169.254.169.254")
        self.assertEqual(
            SCANNER.METADATA_SERVICE_HOST,
            "metadata.google.internal",
        )
        self.assertIn(
            SCANNER.METADATA_SERVICE_IP,
            SCANNER.HIGH_RISK_CREDENTIAL_STRINGS,
        )
        self.assertIn(
            SCANNER.METADATA_SERVICE_HOST,
            SCANNER.HIGH_RISK_CREDENTIAL_STRINGS,
        )

    def test_download_and_execute_is_still_high_severity(self):
        source = b"""
def install_payload(url):
    payload = requests.get(url)
    os.system(payload.text)
"""
        result = SCANNER.ScanResult()

        SCANNER.scan_file_ast("src/install_payload.py", source, result)

        findings = [
            finding
            for finding in result.findings
            if finding.category == "AST_DOWNLOAD_EXECUTE"
        ]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_download_without_execution_is_not_flagged_as_dropper(self):
        source = b"""
def fetch_payload(url):
    return requests.get(url)
"""
        result = SCANNER.ScanResult()

        SCANNER.scan_file_ast("src/fetch_payload.py", source, result)

        self.assertNotIn(
            "AST_DOWNLOAD_EXECUTE",
            {finding.category for finding in result.findings},
        )


# Built by join, like the scanner's own constants, so this test file is not
# itself flagged as a cloud-metadata reference.
METADATA_HOST = ".".join(("metadata", "google", "internal"))
GO_METADATA_SOURCE = (
    "package feed\n\nconst IdentityURL = "
    f'"http://{METADATA_HOST}/computeMetadata/v1/instance/identity"\n'
).encode()


class CredentialExemptionTests(unittest.TestCase):
    """Reviewed cloud-metadata references in `ast-scan-exemptions.toml`."""

    EXEMPTED_PATH = "src/pkg/feed/idtoken.go"

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        # Restore global scanner policy state between tests.
        self.addCleanup(SCANNER._load_credential_exemptions, Path(self._temp.name))

    def _write_exemptions(self, body: str):
        path = self.root / "tools" / "security" / "ast-scan-exemptions.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        SCANNER._load_credential_exemptions(self.root)

    def _write_go_file(self, relpath: str) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(GO_METADATA_SOURCE)
        return path

    @staticmethod
    def _b64(pattern: str) -> str:
        return base64.b64encode(pattern.encode()).decode()

    def _exempt_entry(self, path: str, pattern: str = METADATA_HOST) -> str:
        return (
            "[[exemption]]\n"
            f'path = "{path}"\n'
            f'pattern_b64 = "{self._b64(pattern)}"\n'
            'reason = "Workload identity token minted at runtime; no stored secret."\n'
        )

    def _credential_refs(self, source_path: Path) -> list:
        result = SCANNER.ScanResult()
        SCANNER.scan_file_ast(str(source_path), source_path.read_bytes(), result)
        return [f for f in result.findings if f.category == "AST_CREDENTIAL_REF"]

    def test_exemption_suppresses_matching_file_and_pattern(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))

        self.assertEqual(self._credential_refs(source_path), [])

    def test_exemption_does_not_apply_to_a_different_file(self):
        other = self._write_go_file("src/pkg/other/idtoken.go")
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))

        findings = self._credential_refs(other)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_unexempted_cloud_metadata_reference_still_fails_high(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions("# no entries\n")

        findings = self._credential_refs(source_path)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)
        self.assertIn("metadata.google.internal", findings[0].message)

    def test_exemption_for_a_different_pattern_does_not_suppress(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions(
            self._exempt_entry(self.EXEMPTED_PATH, SCANNER.METADATA_SERVICE_IP)
        )

        findings = self._credential_refs(source_path)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_exemption_without_a_reason_is_ignored(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions(
            "[[exemption]]\n"
            f'path = "{self.EXEMPTED_PATH}"\n'
            f'pattern_b64 = "{self._b64(METADATA_HOST)}"\n'
            'reason = "   "\n'
        )

        findings = self._credential_refs(source_path)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_exemption_with_undecodable_pattern_is_ignored(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions(
            "[[exemption]]\n"
            f'path = "{self.EXEMPTED_PATH}"\n'
            'pattern_b64 = "not!valid!base64"\n'
            'reason = "Encoding is wrong, so this must not apply."\n'
        )

        findings = self._credential_refs(source_path)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_exempted_pattern_does_not_hide_another_marker_in_the_literal(self):
        source_path = self.root / self.EXEMPTED_PATH
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(
            (
                "package feed\n\nconst Paths = "
                f'"http://{METADATA_HOST}/ and ~/.ssh/id_rsa"\n'
            ).encode()
        )
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))

        findings = self._credential_refs(source_path)

        # The reviewed hostname is exempt; the unreviewed marker in the same
        # string literal is still reported, whichever pattern is visited first.
        self.assertEqual(len(findings), 1)
        self.assertIn("~/.ssh", findings[0].message)

    def test_without_a_toml_parser_exemptions_apply_to_nothing(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        parser = SCANNER.tomllib
        self.addCleanup(setattr, SCANNER, "tomllib", parser)
        SCANNER.tomllib = None
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))

        findings = self._credential_refs(source_path)

        self.assertEqual(SCANNER._credential_exemptions, {})
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_malformed_exemption_file_exempts_nothing(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions("[[exemption]\npath = broken\n")

        findings = self._credential_refs(source_path)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_unused_exemption_is_reported_as_stale(self):
        # The exempted file does not exist, so nothing consumed the entry.
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))
        result = SCANNER.ScanResult()

        SCANNER._report_stale_exemptions(result)

        stale = [f for f in result.findings if f.category == "AST_EXEMPTION_STALE"]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0].severity, SCANNER.Severity.MEDIUM)
        self.assertIn(self.EXEMPTED_PATH, stale[0].message)

    def test_used_exemption_is_not_reported_as_stale(self):
        source_path = self._write_go_file(self.EXEMPTED_PATH)
        self._write_exemptions(self._exempt_entry(self.EXEMPTED_PATH))
        self._credential_refs(source_path)
        result = SCANNER.ScanResult()

        SCANNER._report_stale_exemptions(result)

        self.assertNotIn(
            "AST_EXEMPTION_STALE",
            {finding.category for finding in result.findings},
        )

    def test_shipped_exemptions_file_entries_are_well_formed(self):
        repo_root = Path(__file__).resolve().parents[2]
        SCANNER._load_credential_exemptions(repo_root)
        self.addCleanup(SCANNER._load_credential_exemptions, repo_root)

        self.assertTrue(SCANNER._credential_exemptions)
        for (path, pattern), reason in SCANNER._credential_exemptions.items():
            self.assertIn(pattern, SCANNER.CREDENTIAL_STRINGS)
            self.assertTrue(reason.strip(), f"{path} exemption needs a reason")
            self.assertTrue(
                (repo_root / path).is_file(),
                f"exempted path does not exist: {path}",
            )


if __name__ == "__main__":
    unittest.main()
