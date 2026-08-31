# ArchaeoTrace 0.1.6 release-candidate readiness

후보 버전 지정: 2026-08-31

이 문서는 Ink Centerline v2와 Smart Recovery를 포함한 experimental `0.1.6`
후보만 다룹니다. 2026-08-26 공개된 공식 QGIS `0.1.5`와 그보다 앞선 로컬 후보의
identity는 [`RELEASE_READINESS_0.1.5.md`](RELEASE_READINESS_0.1.5.md)에 보존합니다.

## Decision

- **로컬 설치·검증 후보: GO.** metadata, plugin changelog와 citation version을
  `0.1.6`으로 맞췄고 결정적 ZIP을 만들었습니다.
- **QGIS/GitHub 공개: 아직 수행하지 않음.** 이 버전 지정은 tag, GitHub Release,
  QGIS plugin 저장소 upload를 만들지 않습니다. 실제 공개일도 기록하지 않았습니다.
- **Stable 전환: NO-GO.** `experimental=True`를 유지합니다. 나머지 공개 benchmark,
  독립 주석 검수, Windows clean-profile GUI·HiDPI·언어 전환과 upgrade smoke가
  남았습니다.
- 실제 역사 지도 holdout이 완료되기 전에는 다른 도구보다 정확하거나 빠르다는
  표현을 사용하지 않습니다.

## Frozen repository-local candidate

| 항목 | 값 |
| --- | --- |
| 후보 파일명 | `ai_vectorizer-0.1.6.zip` |
| SHA-256 | `fffceee8607bdb19178b224e2c94493791d0175a1e35039368f0a41fa7447b7a` |
| 크기 | 1,688,121 bytes |
| ZIP entries | 35 |
| metadata | `0.1.6`, QGIS `3.22–4.99`, `experimental=True` |
| model weight/native binary | 포함하지 않음 |
| exact source commit | **PENDING:** version-bump 구현 commit이 원격에 올라간 뒤 결속 |
| exact CI | **PENDING:** 위 commit의 전체 matrix가 끝난 뒤 결속 |

ZIP은 1980-01-01 timestamp, 정렬된 entry, 고정 mode와 저장 압축을 사용합니다.
같은 plugin tree의 두 빌드는 byte-identical이어야 하며 Linux, Windows와 각 QGIS
matrix job도 동일 SHA-256을 검사합니다. `scripts/package_release.py`는 이 후보와
과거 `0.1.5` 후보의 경로를 모두 동결 보호합니다. metadata가 새 버전이어도 명시적
출력 경로로 과거 동결 ZIP을 덮어쓸 수 없습니다.

`dist/`, metadata-derived release tree와 `*.zip`은 의도적으로 Git에서 제외합니다.
따라서 Git에는 source, 결정적 builder, 동결 hash와 이 기록을 보존하고, exact ZIP
byte는 로컬 파일 또는 CI 재빌드 결과에서 위 SHA-256으로 확인합니다. 후보 ZIP이
Git에 commit돼 있다고 해석하지 않습니다.

## Candidate scope

- 9·15·31 source-pixel, RGB/명도와 source-grid tile 정규화를 사용하는
  `LineEvidence` 및 Ink Live-Wire v2
- 기본 OFF이며 사용자가 model을 명시적으로 설치하는
  `Smart Recovery (Experimental)`
- EfficientSAM-Ti mask를 최종 선이나 Ink와의 이진 OR로 쓰지 않고, 약한 Ink 구간의
  corridor prior로만 사용하는 보수적 champion/challenger 경로
- model 없음·손상·runtime 없음·inference 오류·취소·stale 결과에서 같은 Ink
  champion을 보존하는 fail-closed 동작
- `Freehand`, 정확한 0% cursor, QGIS edit-buffer·Undo와 기존 model index 0–5 보존
- LSD, HED, MobileSAM, SAM, Legacy Canny를 `Advanced / Legacy methods`에 유지
- 공개 8개 도엽·48개 crop 계약과 USGS HTMC 1개 도엽·draft crop 6개 착수

기능·실패 경계는 [`INK_V2_SMART_RECOVERY.md`](INK_V2_SMART_RECOVERY.md), 전체
module 책임은 [`FEATURES_AND_ARCHITECTURE.md`](FEATURES_AND_ARCHITECTURE.md), 사용자
변경은 [`../CHANGELOG.md`](../CHANGELOG.md)에 기록합니다.

## Verification

버전 번호를 지정하기 직전의 동일 plugin 기능 tree에서 다음을 확인했습니다.

- Python 3.10/3.12: 각 `477 passed, 59 skipped, 122 subtests`
- Python 3.8 no-dependency 계약: `272 passed, 8 skipped`; 전체 compile 통과
- macOS QGIS 3.44.8 runtime safety: `56/56`
- clean-profile QGIS 3.44.8: Freehand, 정확한 0%, Ink v2, rapid/cache click,
  Enhanced WYSIWYG 저장 통과
- Recovery lifecycle QGIS 3.44.8: model 없음·정상 실제 CPU inference·runtime 없음·
  손상·Repair·inference 실패 Ink 보존 통과
- 기능 commit `8420773`의
  [CI run 33343315402](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/33343315402):
  Python, dependency/security, Linux/Windows package와 QGIS
  3.22.16/3.44.13/4.2.1 모두 green
- `0.1.6` exact ZIP: deterministic build·source manifest·metadata parse 통과
- `0.1.6` worktree Python 3.12: `478 passed, 59 skipped, 125 subtests`
- `0.1.6` worktree Python 3.8 no-dependency 계약: `281 tests, 8 skipped`; 전체 compile 통과
- `0.1.6` worktree macOS QGIS 3.44.8 runtime safety: `56/56`
- 위 exact ZIP을 macOS QGIS 3.44.8 clean profile에 설치해 metadata `0.1.6`,
  Freehand, 정확한 0%, Ink v2, rapid/cache click deferral와 Enhanced WYSIWYG 저장 통과

위 기능 CI는 `0.1.6` 숫자를 넣기 전 plugin tree의 동작 증거입니다. 따라서 version
bump commit 자체의 exact ZIP·원격 CI는 위 표의 PENDING을 채우기 전까지 별도
attestation으로 주장하지 않습니다.

## Remaining gates

- 나머지 7개 도엽·42개 crop, 한국·한반도 자료의 항목별 재배포 권리와 독립 검수
- locked holdout의 Ink v2 및 Smart Recovery 통과 기준
- Windows 실제 QGIS clean-profile GUI, HiDPI와 한국어/영어 전환
- 공식 `0.1.5 → 0.1.6` upgrade, model 보존과 rollback smoke
- SBOM과 서명/attestation 적용 또는 미적용 사유
- 모델 download transport가 QGIS network manager가 아닌 Python transport를 사용해
  조직 proxy 설정을 자동 상속하지 못할 수 있음; 사용자가 시작한 설치만 실패하며
  trace는 같은 Ink champion으로 fail closed

이 조건은 현재 ZIP의 설치 테스트를 막지는 않지만 stable 승격, 정확도 순위와
공식 게시 판단에는 계속 적용됩니다.
