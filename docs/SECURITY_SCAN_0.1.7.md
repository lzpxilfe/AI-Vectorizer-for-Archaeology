# 0.1.7 security notification review

12 November 2026. The earlier submission was removed by the maintainer; its
version page now returns 404. The prepared upload ZIP remains identified by
SHA-256 `43ce32bec77ff03565e100643f7e3850d2d90ce0d92876cdec12f8dc50dea546`.

## Conflicting email and scan results

The website previously displayed five passing checks, no warnings and no critical
issues, with **Validated (configured)** because `.secrets.baseline` was present.
The maintainer nevertheless received a BLOCKED email without issue details.

The public website code contains a reproducible mismatch:

1. With zero critical findings and a config file, the scan task stores
   `validated_with_config` and notifies staff that the plugin is ready for review.
2. The email function recognises only `validated` as success. Its fallback sends
   the BLOCKED template for `validated_with_config` as well.
3. Because there are no critical findings, the detail builder returns an empty
   findings explanation rather than identifying a security defect.

See the [status decision](https://github.com/qgis/QGIS-Plugins-Website/blob/06a2e650bef2a7d3971ff37fa9f327f3adf829f9/qgis-app/plugins/tasks/run_security_scan.py#L98-L155)
and [email branch](https://github.com/qgis/QGIS-Plugins-Website/blob/06a2e650bef2a7d3971ff37fa9f327f3adf829f9/qgis-app/plugins/tasks/run_security_scan.py#L195-L264).
The unchanged upstream email functions were executed locally with a capture
function in place of email delivery. `validated_with_config` plus a critical
count of zero reproduced the received BLOCKED/no-details text; `validated`
produced the expected success text. No external message was sent.

This reproduces the defect in public source commit `06a2e650…`, which matched
the default-branch file when checked. The deployed server commit and historical
database row were not inspected, so this is strong evidence for the notification
mismatch, not a claim to have read the production database.

## Review without the baseline

The byte-identical upload package was scanned with `detect-secrets 1.5.0`, using
both its default scan and the website's command, without applying the baseline
or disabling detectors. Each reported exactly nine high-entropy hexadecimal
strings, all matching the existing baseline:

- Eight public model-file SHA-256 integrity values.
- One pinned public EfficientSAM source commit.
- No other findings.

Each fingerprint was checked against the literal value and its role in source.
The existing release test now verifies this binding, in addition to the reviewed
file and line markers. This audit did not download all model weights again.

Removing the baseline alone would make these nine findings count as critical
under the website's [Secrets Detection rule](https://github.com/qgis/QGIS-Plugins-Website/blob/06a2e650bef2a7d3971ff37fa9f327f3adf829f9/qgis-app/plugins/security_scanner.py#L244-L338).
The reviewed baseline remains in the ZIP for administrator review. The model
hashes remain readable, and neither scanner rules nor integrity checks were
weakened. No product file changed during this investigation.

The [evidence record](evidence/security-status-0.1.7.json) binds the reproducer,
scanner results and public source. The [upload ZIP](QT6_UPLOAD_0.1.7.md) retains
its verified Qt6 result of zero findings.

## 한국어

설정 파일을 사용한 검사 통과 상태를 BLOCKED 메일로 보내는 공개 저장소 코드의
분기 오류를 재현했습니다. 이 결과는 기존 화면의 critical 0건과 메일의
`No details available`이 함께 나타난 이유를 설명합니다.

예외 목록 없이 다시 검사한 아홉 항목은 공개 모델 해시 여덟 개와 소스 commit
한 개였습니다. 목록만 삭제하면 이 항목들이 다시 검출되므로 기존 목록을 유지하고
관리자가 검토할 수 있도록 근거를 남깁니다. 현재 이전 제출은 삭제된 상태이며
이 조사에서 다시 업로드하거나 외부 메일·이슈를 보내지는 않았습니다.
