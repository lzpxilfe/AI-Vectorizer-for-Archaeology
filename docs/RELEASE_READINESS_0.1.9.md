# ArchaeoTrace 0.1.9 experimental release candidate

후보 준비·로컬 검사: 2026-11-26

공식 QGIS 공개판 `0.1.8`(2026-11-23)에서 패치 버전을 올린 experimental 공개 후보다.
이 문서는 업로드할 ZIP과 검증을 결속하는 기록이다. 이전 공개 기록은
[0.1.8 릴리스 기록](RELEASE_READINESS_0.1.8.md)에 보존한다.

## Decision

- 버전: plugin metadata와 citation을 `0.1.9`로 맞추고 `experimental=True`를 유지한다.
- 공개 상태: 아직 업로드하지 않았다. `CITATION.cff`에는 공개일을 넣지 않았고, Zenodo
  버전 DOI가 생길 때까지 concept DOI를 쓴다.
- 정확도·속도: 사람 검수 Holdout 및 사용자 작업 시간 개선을 주장하지 않는다. 지도
  시리즈별 기본 제공 프리셋은 넣지 않았다. 검수된 calibration 도엽에서 확인한 뒤에만
  추가한다.

## Candidate scope

- 지도 시리즈 프리셋([#5](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/issues/5)):
  추적 방식, 보조 강도, Smart Recovery, 프리핸드, 자동 경로, 학습 가이드, 선 색상 RGB,
  등고선 간격을 이름 붙여 저장·적용·이름 변경·삭제하고 `archaeotrace-trace-preset/1`
  JSON 파일로 내보내고 가져온다. 형식 오류, 알 수 없는 항목, 더 새로운 schema, dock이
  만들 수 없는 조합은 이유를 밝혀 거부한다.
- 실행 환경이 없는 선택 기능(ONNX Runtime, OpenCV, HED 모델, SAM backend, 가져온 학습
  모델)이 든 프리셋은 그 항목만 끄거나 Ink로 바꾸고 건너뛴 항목을 표시한다.
- 프로젝트 기억: 추적 시작이나 프리셋 적용 때 현재 설정을 래스터별로 QGIS 프로젝트에
  기록하고, 그 래스터를 다시 고르면 복원한다. 프로젝트 저장·다시 읽기 후에도 유지됨을
  실제 QGIS 테스트로 확인했다.
- 등고선 간격: 새 등고선의 해발값 창 기본값만 직전 값 ± 간격으로 바뀐다. 기존 선 연장과
  표고점은 바뀌지 않으며, 사용자가 확인한 값만 저장된다.
- 선 색상: `C`로 지정한 색은 `Shift+C`나 `색상 해제`로 지울 때까지 다음 추적에도
  적용된다(이전에는 현재 추적에만 적용). 저장되는 것은 RGB 세 값뿐이다.
- 사용자 제보: Sebastian Walczak의 Geoawesome 사례 글에서 지도마다 설정을 다시 찾아야
  했다는 지적을 반영했다.
- 0.1.8 대비 ZIP 변경: 새 파일 `core/trace_preset.py`, `ui/trace_presets_panel.py`;
  바뀐 파일 `tools/smart_trace_tool.py`, `ui/main_dialog.py`, `metadata.txt`, 한·영 사용
  설명서. 항목 수 53→55, Python 파일 46→48.

## Frozen repository-local candidate

| 항목 | 최종 후보 기록 |
| --- | --- |
| 파일명 | `ai_vectorizer-0.1.9.zip` |
| Source | 이 기록을 추가한 commit (기능 commit `9099107` 뒤 줄 길이 정리 포함) |
| ZIP SHA-256 | 340c3160b51fb851a3baa39262d50449872263391302a9fe4b254e126a3433f0 |
| 크기 / entries | 2,165,523 bytes / 55 |
| metadata | `0.1.9`, QGIS `3.22–4.99`, `experimental=True` |
| model weights / 원본 지도 | ZIP에 포함하지 않음 |

결정적 builder가 같은 plugin tree에서 같은 bytes를 만든다. Git에는 source·builder·해시를
보존하고 ZIP은 별도 전달한다. `0.1.5`-`0.1.8`의 과거 동결 후보는 덮어쓰지 않는다.

## Verification

| 검사 | 결과·근거 |
| --- | --- |
| 일반 Python suite | 1,271 passed, 225 skipped, 186 subtests passed |
| 실제 QGIS | QGIS 3.34.4, CI 대상 17개 모듈 243 tests 통과, 실패·오류 0 |
| 프리셋 형식 | pure-Python 18 tests: 왕복, 이름 규칙, 새 버전·알 수 없는 항목·누락·형식 오류 거부, 적용 계획, 해발값 제안 |
| 프리셋 dock | 실제 QGIS 7 tests: 저장·적용 왕복, 건너뛴 항목 표시, 이름 변경·삭제·내보내기·가져오기, 새 버전 거부, 읽을 수 없는 저장값 백업, 프로젝트 기억과 저장 후 복원, 추적 중 잠금 |
| 해발값 제안·선 색상 | 실제 QGIS: 제안·방향·기록, 표고점과 취소 제외, 간격 없음; 선 색상 유지·해제·프리셋 색 시작 |
| packaged import | 위 ZIP을 풀어 QGIS 3.34.4에서 plugin entry point import 통과 |
| 결정적 ZIP / source 일치 | 생성·`--check` 통과, ZIP CRC 오류 없음 |
| 화면 | 한·영 dock을 offscreen 렌더링해 프리셋 영역이 잘리지 않음을 확인; 작은 창에서 가로 스크롤이 생기지 않는다는 기존 테스트 통과 |
| 원격 CI | 기능 commit `9099107`에서 12/12 통과(QGIS 3.22.16 · 3.44.13 · 4.2.1, 공식 Qt6 검사와 새 QGIS 모듈 포함). 후보 commit `99bc783`에서도 12/12 통과했고, CI의 Linux build가 위와 같은 ZIP SHA-256 `340c3160…`을 만들었다 |

`qgis_clean_profile_smoke.py`는 macOS QGIS.app 실행 파일이 필요해 이번 Linux 검사에서는
실행하지 않았다. 대신 위 packaged import와 CI의 세 QGIS 버전 import smoke로 확인했다.

## QGIS repository checks reproduced

QGIS plugin 저장소의 보안 검사기가 쓰는 명령을 위 ZIP에 그대로 실행했다. 비교를 위해
승인된 0.1.8 공개 ZIP(SHA-256 `3ecc7fa8…`)에도 같은 명령을 실행했다.

| 검사 | 저장소 등급 | 0.1.9 결과 | 0.1.8 공개판 |
| --- | --- | --- | --- |
| Bandit (`-r -f json --quiet -ll`) | critical | 0건 | 0건 |
| detect-secrets (`--all-files`, ZIP 루트 기준) | critical | 9건, 모두 기존 baseline과 일치 | 같음 |
| Flake8 (`--max-line-length=120`) | info | 582건 | 583건 |
| 실행 권한 `.py` | info | 없음 | 없음 |
| 숨김·의심 파일 | warning | 없음 | 없음 |

detect-secrets의 9건은 공개 모델 파일의 SHA-256 무결성 값 8개와 고정한 EfficientSAM
source commit 1개다(`.secrets.baseline` 자체를 세면 27건으로, 0.1.8과 같다). 이번
변경으로 새로 생긴 항목은 없다. 새 파일 두 개에는 Flake8 지적이 없다.

## Upload steps

1. 위 SHA-256과 같은 `ai_vectorizer-0.1.9.zip`을 QGIS plugin 저장소의 ArchaeoTrace
   페이지에서 새 버전으로 업로드한다.
2. 업로드 후 사이트의 보안·Qt6 검사 결과를 확인하고, 저장소에서 내려받은 ZIP의 SHA-256이
   위 값과 같은지 대조한다.
3. 검사가 통과로 표시되는데도 BLOCKED 이메일이 오면 다시 올리지 않는다. 사이트의 알려진
   알림 버그([QGIS-Plugins-Website#486](https://github.com/qgis/QGIS-Plugins-Website/issues/486))이며
   [0.1.8 릴리스 기록](RELEASE_READINESS_0.1.8.md)에 같은 사례를 기록했다.
4. 승인 후 같은 변경에서 공개일을 `CITATION.cff`, `CHANGELOG.md` 제목과 이 기록에 넣고,
   GitHub Release `v0.1.9`로 Zenodo 버전 DOI를 만든 뒤 #5를 닫는다.
