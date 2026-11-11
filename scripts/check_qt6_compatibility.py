#!/usr/bin/env python3
"""Run QGIS's pinned migration checker against an extracted release package.

Requires real QGIS 4/PyQt6 bindings and tokenize-rt. The official dry run can
exit successfully while reporting problems, so validate its output as well as
its exit code. An intentionally incompatible fixture checks the enum detector.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.request import urlopen


CHECKER_COMMIT = "940b9447b4a842803c5cceb08b6fabb9f0a8a115"
CHECKER_URL = (
    "https://raw.githubusercontent.com/qgis/QGIS/" + CHECKER_COMMIT
    + "/scripts/pyqt5_to_pyqt6/pyqt5_to_pyqt6.py"
)
CHECKER_SHA256 = "c6734b909bf29d06c1d173ed03f03e4ee87abf29308932a3f0b1833288f837de"
HEADER = "=== dry_run mode | Start Logs ==="
EXPECTED_CONTROL_MESSAGES = {
    "Enum error, add 'ActionOnExistingFile' before 'CreateOrOverwriteFile'",
    "Enum error, add 'ActionOnExistingFile' before 'CreateOrOverwriteLayer'",
    "Enum error, add 'WriterError' before 'NoError'",
}


def source_hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def run_check(checker, root):
    process = subprocess.run(
        [sys.executable, str(checker), str(root), "--dry_run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
        timeout=180,
    )
    output = process.stdout + process.stderr
    issues = [
        {"file": filename, "line": int(line), "column": int(column), "message": message}
        for filename, line, column, message in re.findall(
            r"^(.+\.py):(\d+):(\d+) - (.+)$", output, re.MULTILINE
        )
    ]
    remaining = []
    for line in output.splitlines():
        if line.strip() and line.strip() != HEADER and not re.match(
            r"^.+\.py:\d+:\d+ - .+$", line
        ):
            remaining.append(line)
    return {
        "exit_code": process.returncode, "issues": issues,
        "other_output": remaining, "dry_run_started": HEADER in output,
        "output": output,
    }


def verify(plugin_root, report):
    # The upstream checker otherwise falls back to a partial Qt-only scan.
    for name in ("qgis.core", "qgis.gui", "qgis.analysis", "qgis._3d"):
        importlib.import_module(name)
    from qgis.core import Qgis
    from PyQt6.QtCore import QT_VERSION_STR

    if Qgis.QGIS_VERSION_INT < 40000 or not QT_VERSION_STR.startswith("6."):
        raise RuntimeError("The Qt6 preflight requires QGIS 4 and PyQt6")
    report["runtime"] = {"qgis": Qgis.QGIS_VERSION, "qt": QT_VERSION_STR}
    before = source_hashes(plugin_root)
    if not before or not (plugin_root / "metadata.txt").is_file():
        raise RuntimeError("Expected the extracted plugin directory")
    report["python_files"] = len(before)
    with tempfile.TemporaryDirectory(prefix="archaeotrace-qt6-check-") as temp:
        folder = Path(temp)
        with urlopen(CHECKER_URL, timeout=30) as response:
            payload = response.read(1024 * 1024 + 1)
        if hashlib.sha256(payload).hexdigest() != CHECKER_SHA256:
            raise RuntimeError("Official checker SHA-256 mismatch")
        checker = folder / "pyqt5_to_pyqt6.py"
        checker.write_bytes(payload)
        control_root = folder / "positive-control"
        control_root.mkdir()
        (control_root / "writer_enums.py").write_text(
            "from qgis.core import QgsVectorFileWriter\n"
            "file_action = QgsVectorFileWriter.CreateOrOverwriteFile\n"
            "layer_action = QgsVectorFileWriter.CreateOrOverwriteLayer\n"
            "writer_error = QgsVectorFileWriter.NoError\n", encoding="utf-8"
        )
        control = run_check(checker, control_root)
        report["positive_control"] = control
        if (
            control["exit_code"] != 0 or not control["dry_run_started"]
            or control["other_output"] or len(control["issues"]) != 3
            or {issue["message"] for issue in control["issues"]} != EXPECTED_CONTROL_MESSAGES
        ):
            raise RuntimeError("The checker did not reproduce all three known enum findings")
        result = run_check(checker, plugin_root)
        report["plugin"] = result
        report["source_unchanged"] = before == source_hashes(plugin_root)
        if (
            result["exit_code"] != 0 or not result["dry_run_started"]
            or result["issues"] or result["other_output"]
            or not report["source_unchanged"]
        ):
            raise RuntimeError("Qt6 checker reported findings, failed, or changed source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "checker_commit": CHECKER_COMMIT, "checker_url": CHECKER_URL,
        "checker_sha256": CHECKER_SHA256, "passed": False,
    }
    try:
        verify(args.plugin_root.resolve(strict=True), report)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    rendered = json.dumps(report, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
