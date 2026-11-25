# ArchaeoTrace 0.1.8 experimental release

후보 준비·로컬 검사: 2026-11-23 · QGIS 업로드: 2026-11-23 · 승인 확인: 2026-11-25

공식 QGIS 공개판 `0.1.7`(2026-11-11)에서 패치 버전을 올린 experimental 공개판이다.
이 문서는 업로드한 ZIP과 검증, QGIS 게시를 결속하는 기록이다. 이전 공개 기록은
[0.1.7 제출 기록](QGIS_SUBMISSION_0.1.7.md)에 보존한다.

## Decision

- 버전: plugin metadata와 citation을 `0.1.8`로 맞추고 `experimental=True`를 유지한다.
- 공개 상태: QGIS 저장소에 게시·승인되었다(아래 QGIS publication). `CITATION.cff`의
  공개일은 QGIS 업로드일 `2026-11-23`(UTC)이다. GitHub Release `v0.1.8`로 Zenodo 버전
  DOI를 만들며, 그 전까지는 concept DOI를 쓴다.
- 정확도·속도: 사람 검수 Holdout 및 사용자 작업 시간 개선을 주장하지 않는다.

## Candidate scope

- DEM 창의 입력 점검 목록: 고도 필드 미선택, 등고선과 다른 표고점 좌표계, 저장하지 않은
  편집, QGIS를 닫으면 사라지는 임시 레이어를 레이어 이름과 함께 표시한다. 같은 종류는
  한 줄로 묶는다. 고도값과 도형은 지금처럼 `build_dem_request`가 생성 시작 때 전부 검사한다.
- 저장하지 않은 편집: 등고선·표고점 레이어에 편집이 남아 있으면 `DEM 생성`에서 오류로
  멈추지 않고 저장할지 묻는다. 저장하면 편집 모드를 유지한 채 생성을 이어가고, 거절하면
  편집을 그대로 둔 채 멈춘다. 추적이 진행 중일 때는 기존처럼 DEM 창을 열 수 없다.
- 사용자 제보: Sebastian Walczak의
  [#4](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/issues/4)를 반영했다.
  WMS 투명 배경 제보([#3](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/issues/3))는
  0.1.7에서 이미 수정되었다.
- 0.1.7 대비 ZIP에서 바뀐 파일은 `core/dem_spec.py`, `ui/dem_dialog.py`, `metadata.txt`와
  한·영 사용 설명서 다섯 개뿐이다. 항목 수(53)와 Python 파일 수(46)는 같다.

## Frozen repository-local candidate

| 항목 | 최종 후보 기록 |
| --- | --- |
| 파일명 | `ai_vectorizer-0.1.8.zip` |
| Source | 이 기록을 추가한 commit (제품 코드는 `a7930ac`) |
| ZIP SHA-256 | 3ecc7fa813da24b6f2a6e227dbf33a87b243ccae2414f628f11f3f0ca0d674fe |
| 크기 / entries | 2,110,206 bytes / 53 |
| metadata | `0.1.8`, QGIS `3.22–4.99`, `experimental=True` |
| model weights / 원본 지도 | ZIP에 포함하지 않음 |

결정적 builder가 같은 plugin tree에서 같은 bytes를 만든다. Git에는 source·builder·해시를
보존하고 ZIP은 별도 전달한다. `0.1.5`-`0.1.7`의 과거 동결 후보는 덮어쓰지 않는다.

## Verification

| 검사 | 결과·근거 |
| --- | --- |
| 일반 Python suite | 1,252 passed, 213 skipped, 171 subtests passed |
| 실제 QGIS | QGIS 3.34.4, CI 대상 15개 모듈 231 tests 통과, 실패·오류 0 |
| DEM 입력 점검 | 실제 QGIS 5 tests 추가: 점검 목록, CRS·필드, 저장 승인·거절, 저장된 입력 |
| 실제 TIN 생성 | Point/MultiPoint TIN→GeoTIFF→hillshade 통과 |
| packaged import | 위 ZIP을 풀어 QGIS 3.34.4에서 plugin entry point import 통과 |
| 결정적 ZIP / source 일치 | 생성·`--check` 통과, ZIP CRC 오류 없음 |
| 화면 | 한·영 DEM 창을 offscreen 렌더링해 점검 목록이 잘리지 않음을 확인 |
| 원격 CI | 제품 코드 `a7930ac`에서 12/12 통과(QGIS 3.22.16 · 3.44.13 · 4.2.1, 공식 Qt6 검사 포함) |

`qgis_clean_profile_smoke.py`는 macOS QGIS.app 실행 파일이 필요해 이번 Linux 검사에서는
실행하지 않았다. 대신 위 packaged import와 CI의 세 QGIS 버전 import smoke로 확인했다.

## QGIS repository checks reproduced

QGIS plugin 저장소의 보안 검사기가 쓰는 명령을 위 ZIP에 그대로 실행했다. 비교를 위해
승인된 0.1.7 공개 ZIP(SHA-256 `43ce32…`)에도 같은 명령을 실행했다.

| 검사 | 저장소 등급 | 0.1.8 결과 | 0.1.7 공개판 |
| --- | --- | --- | --- |
| Bandit (`-r -f json --quiet -ll`) | critical | 0건 | 0건 |
| detect-secrets (`--all-files`, ZIP 루트 기준) | critical | 9건, 모두 기존 baseline과 일치 | 같음 |
| Flake8 (`--max-line-length=120`) | info | 583건 | 583건 |
| 실행 권한 `.py` | info | 없음 | 없음 |
| 숨김·의심 파일 | warning | 없음 | 없음 |

detect-secrets의 9건은 공개 모델 파일의 SHA-256 무결성 값 8개와 고정한 EfficientSAM
source commit 1개다. 이번 변경으로 새로 생긴 항목은 없다. Flake8 지적은 0.1.7과 같은
수이며 이번 변경으로 늘지 않았다.

## QGIS publication

| 항목 | 기록 |
| --- | --- |
| 업로드 | 2026-11-23 06:38 UTC (KST 15:38), 버전 페이지 `version/0.1.8/` |
| 보안 검사 | `All checks passed (developer config files used)`, Pass Rate 100%, `Validated (configured)`; 감지된 설정 파일은 `.secrets.baseline` 하나 |
| 알림 메일 | 같은 시각 `BLOCKED`·`Critical issues found: No details available.` 메일 수신. 검사 결과와 맞지 않는 사이트 알림 버그([QGIS-Plugins-Website#486](https://github.com/qgis/QGIS-Plugins-Website/issues/486), 2026-11-23 확인 기준 미해결)이며 다시 업로드하지 않았다 |
| 승인 | 2026-11-25 확인, `Approved: yes`, experimental |
| 게시 ZIP 대조 | 저장소에서 내려받은 ZIP SHA-256 `3ecc7fa813da24b6f2a6e227dbf33a87b243ccae2414f628f11f3f0ca0d674fe`, 위 동결 후보와 같음 |

`.secrets.baseline`의 9개 항목은 공개 모델 무결성 해시 8개와 EfficientSAM source commit
1개이며, [0.1.7 보안 알림 검토](SECURITY_SCAN_0.1.7.md)에 항목별로 기록했다.

## Upload steps (as performed)

1. 위 SHA-256과 같은 `ai_vectorizer-0.1.8.zip`을 QGIS plugin 저장소의 ArchaeoTrace
   페이지에서 새 버전으로 업로드한다.
2. 업로드 후 사이트의 보안·Qt6 검사 결과를 확인하고, 저장소에서 내려받은 ZIP의 SHA-256이
   위 값과 같은지 대조한다.
3. 검사가 통과로 표시되는데도 BLOCKED 이메일이 오면 다시 올리지 않는다. 사이트의 알려진
   알림 버그([QGIS-Plugins-Website#486](https://github.com/qgis/QGIS-Plugins-Website/issues/486))이며
   [0.1.7 보안 알림 검토](SECURITY_SCAN_0.1.7.md)에 원인을 기록했다.
4. 승인 후 같은 변경에서 공개일을 `CITATION.cff`, `CHANGELOG.md` 제목과 이 기록에 넣고,
   GitHub Release `v0.1.8`로 Zenodo 버전 DOI를 만든 뒤 #4를 닫는다. 이 단계는
   2026-11-25에 진행했다.
