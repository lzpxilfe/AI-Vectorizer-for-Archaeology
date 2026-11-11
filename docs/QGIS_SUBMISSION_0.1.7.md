# ArchaeoTrace 0.1.7 QGIS submission

Status checked: 12 November 2026, Korea Standard Time.

Experimental 0.1.7 has been uploaded to the
[QGIS plugin repository](https://plugins.qgis.org/plugins/ai_vectorizer/version/0.1.7/).
The version remains **unapproved** and the latest approved version is 0.1.6.
Upload and automated checks do not replace the repository's manual review.

## Final package

| Item | Value |
| --- | --- |
| Source | `217ed92a08af19cc8092c88d10b3644db635e09b` |
| Filename | `ai_vectorizer-0.1.7.zip` |
| SHA-256 | `80990953cc69fd519b906d70138751d139f65c9d1553fdc671aeef739e87e722` |
| Size / entries | 2,098,182 bytes / 53 |
| QGIS range | 3.22–4.99; experimental |
| Repository download | [Submitted ZIP](https://plugins.qgis.org/plugins/ai_vectorizer/version/0.1.7/download/) |

The ZIP downloaded from QGIS after the update matches this hash exactly.
Linux and Windows CI builds reproduce it. All 46 Python files match the source.
Both Korean and English guides are included. Their candidate wording records
the pre-approval build; this page tracks the submission status.

The package includes the [code review fixes](RELEASE_REVIEW_2026-11-12.en.md),
[language update](LANGUAGE_REVIEW_2026-11-12.md), and scoped writer enums for
annotation GeoPackages. Only `core/annotation_project.py` changed from the
language candidate with SHA-256 `39c042074d4c2eb21f5dc39f12a3c1d37115028563c0a7d58e5f0e45e8f2b546`.
Previous ZIPs and the repository's historical frozen hashes remain unchanged.

## Validation

- Pure Python: 1,246 passed, 208 skipped, 171 subtests passed. QGIS-dependent
  tests were run separately in QGIS, rather than counted as passed here.
- Actual QGIS 3.44.8: 226 passed; no failures, errors or skips.
- Isolated plugin-loader profile: all seven scenarios passed; temporary profiles
  were removed and no user profile was modified.
- [Remote CI](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43115647182):
  all 12 jobs passed, including QGIS 3.22.16, 3.44.13 and 4.2.1, security,
  dependency checks, the CPU learning contract and reproducible packaging.
- Final repository security scan at 07:57 KST: all five checks passed, with
  zero warnings and zero critical issues. All 127 upload rules stayed enabled.

The repository labels the security result **Validated (configured)** because
the ZIP contains `.secrets.baseline`. Its nine entries match public model hashes
(eight) and a pinned source commit (one), not credentials. The baseline did not
change during submission. Administrators still review this configuration.

## Qt6 result awaiting repository refresh

The first upload received three enum-format findings in annotation project
creation. The final package uses `ActionOnExistingFile` and `WriterError` scoped
enums, with a fallback for older QGIS. Native annotation creation/resume and the
QGIS 4.2.1 CI run pass with this change.

The website's Qt6 tab still shows the **07:49 KST result for the first ZIP**,
including the old line numbers. It is not a confirmed rescan of the final ZIP.
In the website's public source, the [Qt6 task is scheduled only on creation](https://github.com/qgis/QGIS-Plugins-Website/blob/06a2e650bef2a7d3971ff37fa9f327f3adf829f9/qgis-app/plugins/signals.py#L10-L31),
while [updating an existing version schedules a security scan](https://github.com/qgis/QGIS-Plugins-Website/blob/06a2e650bef2a7d3971ff37fa9f327f3adf829f9/qgis-app/plugins/views.py#L2143-L2154).
The owner UI exposes no Qt6 rerun action. The deployed server commit was not
verified, but the observed timestamps agree with this lifecycle. Repository-side
Qt6 refresh and manual approval remain outstanding; no scan settings were weakened.

The [machine-readable evidence](evidence/qgis-submission-0.1.7.json) binds the
final artifact, tests, download and observed repository status. No release date,
Git tag or GitHub Release has been added while approval is pending.

## 한국어

0.1.7 최종 ZIP 업로드와 다운로드 해시 확인을 마쳤습니다. 공식 보안 검사는
치명적 문제 0건·경고 0건으로 통과했습니다. 기존 모델 해시 목록 때문에
설정 파일 사용 표시가 있으며 관리자 검토 대상입니다.

Qt6에서 지적한 세 곳은 수정하고 실제 QGIS 4에서도 검사했습니다. 다만 저장소
화면은 교체 전 결과를 표시하며 소유자용 재검사 기능이 없어 관리자 측 갱신이
필요한 것으로 판단합니다. 현재는 승인 대기이며 승인된 최신 판은 0.1.6입니다.

학습 보조의 기본 OFF와 기존 Korea 모델은 유지합니다. 재학습으로 일반 글자
공백 정확도가 개선됐다는 주장은 하지 않습니다.
