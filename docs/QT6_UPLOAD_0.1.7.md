# ArchaeoTrace 0.1.7: Qt6 upload ZIP

Prepared on 12 November 2026 for manual upload.

The maintainer uploaded this exact ZIP at 08:27 KST. Repository security and
Qt6 checks passed; administrator approval remains pending.
[Server download and scan verification](evidence/qgis-resubmission-0.1.7.json)

The annotation writer now directly uses
`QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile`,
`QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer` and
`QgsVectorFileWriter.WriterError.NoError`. These are the three enum locations
identified by the QGIS repository's original check.

## Package

| Item | Value |
| --- | --- |
| Desktop filename | `ArchaeoTrace-0.1.7-QGIS-upload.zip` |
| ZIP root | `ai_vectorizer/` |
| Version | `0.1.7`, experimental |
| SHA-256 | `43ce32bec77ff03565e100643f7e3850d2d90ce0d92876cdec12f8dc50dea546` |
| Size / entries | 2,098,053 bytes / 53 |
| Source and CI | `2f6ca7614db2c2423ba350f8b28ecc257cb26b08` |

The only packaged file changed from the earlier `809909…` upload is
`core/annotation_project.py`. All 46 Python files match the source. The package
includes both language guides, passes the package check and has a clean CRC.
The previous submission is recorded [separately](QGIS_SUBMISSION_0.1.7.md).

## Validation

- Actual QGIS 3.44.8: 226 tests passed, no failures, errors or skips.
- Isolated plugin-loader profile: all seven scenarios passed.
- Pure Python: 1,246 passed, 208 skipped, 171 subtests passed.
- Official Qt6 checker in QGIS 4.2.1 / Qt 6.9.2: all three original findings
  reproduced in the control; **zero findings and zero other warnings** in the
  46 packaged Python files. Source remained unchanged.
- Remote validation: all 12 jobs passed, including native QGIS 3.22.16,
  3.44.13 and 4.2.1 and identical Linux/Windows ZIPs.
  [CI run](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43118150018) ·
  [Evidence](evidence/qt6-upload-0.1.7.json)

The official QGIS migration checker is pinned to
[`940b9447…`](https://github.com/qgis/QGIS/blob/940b9447b4a842803c5cceb08b6fabb9f0a8a115/scripts/pyqt5_to_pyqt6/pyqt5_to_pyqt6.py)
and verified by SHA-256. The CI wrapper first reproduces the original three
enum findings in a temporary control file, then requires no findings in the
extracted release ZIP. It rejects partial Qt-only scans, unexpected output and
source changes. This check now runs in the QGIS 4 CI job before the runtime tests.

The first CI attempt could not start the checker because the runtime image
lacked `PyQt6.QtQuickWidgets`. The CI environment was extended with the distro's
QtQuick bindings; the plugin ZIP did not change during that correction.

## 한국어

지적된 세 곳을 명시적인 enum 표기로 수정한 업로드용 ZIP입니다. 데스크톱 파일명은
`ArchaeoTrace-0.1.7-QGIS-upload.zip`이며 체크섬 파일도 함께 만들었습니다.
이 파일의 검증 기록은 이전에 제출한 ZIP과 구분합니다.
공식 Qt6 검사 오류 0건, 실제 QGIS 검사 226개, 원격 CI 12개 작업을 통과했습니다.
