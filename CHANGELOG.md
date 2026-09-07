# Changelog

이 파일은 사용자가 체감하는 변경을 기록합니다. 배포 버전의 source of truth는
`ai_vectorizer/metadata.txt`이며, 실제 공개일은 QGIS plugin 저장소나 GitHub Release에
같은 artifact를 게시한 날에만 기록합니다.

## Unreleased

다음 변경은 아직 동결된 `0.1.6` 후보의 metadata·tag·ZIP을 바꾸지 않습니다.

### Added

- `TraceGuidance`: line evidence와 분리된 QGIS-independent, immutable soft
  avoidance score 계약. Ink Live-Wire는 선택 영역을 hard wall이나 mask OR로
  쓰지 않고 비용만 높여 더 좋은 중심선 우회로가 있을 때만 선택합니다.
- Ink 추적 중 `Alt`+두 번 클릭으로 문자·기호·얼룩 회피 사각형을 표시하는
  session-only Manual Avoid Guidance. `Alt+Backspace`는 마지막 영역을,
  `Alt+Shift+Backspace`는 모든 영역을 제거합니다.
- 색 있는 등고선이 숫자 라벨 때문에 실제로 끊긴 경우를 위한 anchor-bound
  continuity bridge. 시작 anchor와 양쪽 centerline endpoint가 같은 비중성 색·방향을
  보일 때에만 Live-Wire 내부의 연속 score를 보강하며, `LineEvidence.centerline`과
  0.1.5 Ink 경로는 변경하지 않습니다.
- 검은·갈색 등고선의 숫자 공백에 쓰는 명시적 `⤴ 라벨 공백 연결` preview. 공백 전
  anchor를 확정하고 반대편 선에 커서를 둔 뒤 사용자가 버튼과 같은 끝점의 두 번째
  클릭으로 채택해야 하며, 이미지 분석·OCR·model 없이 두 사용자 endpoint와 local
  tangent만으로 제한된 Hermite segment를 만듭니다.

### Safety

- manual guidance는 Freehand/0% assist/SAM 경로를 바꾸지 않습니다. Recovery를 켠
  경우에도 같은 immutable soft cost를 challenger corridor에 전달하며, model mask를
  선으로 저장하거나 hard exclusion으로 쓰지 않습니다. 지도·selection은 저장·업로드·
  telemetry로 전송되지 않습니다.
- continuity bridge는 grayscale/neutral anchor, 색 불일치, 방향 불일치, 12px보다
  짧거나 96px보다 긴 공백, 복잡하게 갈라진 색 endpoint에서 동작하지 않습니다. 이때
  기존 Ink champion을 그대로 유지하며 virtual support를 이진 mask와 OR하지 않습니다.
- manual gap bridge는 3–128px 사이의 명시적 endpoint만 받고, local tangent를 읽지
  못하거나 detour가 1.25배를 넘으면 preview조차 만들지 않습니다. 저장은 같은
  endpoint를 다시 클릭해 사용자가 green preview를 확인한 뒤에만 일어납니다.

## 0.1.6 — experimental release candidate (not yet published)

2026-08-31에 현재 Ink v2·Smart Recovery 소스의 metadata를 `0.1.6`으로
정렬했습니다. 이는 설치·검증을 위한 후보이며 아직 QGIS plugin 저장소, GitHub
Release 또는 tag로 게시한 릴리스가 아닙니다. 공개일은 정확히 같은 ZIP을 실제로
게시할 때만 추가합니다.

### Added

- 9·15·31 source-pixel, RGB/명도, source-grid tile 정규화를 사용하는 연속
  `LineEvidence`와 이를 직접 소비하는 Ink Live-Wire v2
- 기본 OFF의 `Smart Recovery (Experimental)`: 검증된 EfficientSAM-Ti mask를
  저신뢰 Ink 구간의 corridor prior로만 사용하고, 안전 검사를 통과한 challenger만 채택
- `Ink`, `Recovering`, `Enhanced`, `Ink fallback` 상태, 명시적 model 설치와 현재
  구간 재시도 UI
- worker request v2의 선택적 `previous_xy`, 네 개의 독립 benchmark method ID와
  8개 도엽·48개 crop 공개 dataset 권리·주석 검증 template
- USGS HTMC public-domain 원본 1개와 source-pixel 그대로의 512px PNG crop 6개,
  immutable provenance·권리 snapshot·draft 중심선을 첫 공개 benchmark 묶음으로 추가

### Changed

- LSD, HED, MobileSAM, SAM과 Legacy Canny를 접힌 `Advanced / Legacy methods`로
  이동하되 기존 model index 0–5와 backend·사용자 model 파일은 유지
- raster provider read는 main thread에 남기고 Ink evidence, Recovery model
  hash/session 준비와 추론을 취소·generation·source identity를 검사하는 background
  task로 분리
- Ink v2는 block stretch 대신 native integer source DN과 합산 31px context를
  사용하며, 고정 range가 없는 float source에서는 사유를 표시하고 Ink v1을 유지
- v2용 확장 block과 0.1.5 호환 visible-extent fallback cache를 분리해, v2 실패가
  기준선의 mask나 map transform을 바꾸지 않도록 고정
- Recovery 입력을 native Byte raster로 제한하고 UInt16 등 더 넓은 정수형은 Ink v2를
  유지하도록 변경; raster `dataChanged` 때 stale evidence와 challenger를 즉시 폐기

### Safety

- model/runtime/hash/shape/inference/cancel/stale 실패에서 다른 backend로 조용히
  전환하지 않고 계산 전과 동일한 Ink champion을 유지
- model 설치 취소를 network read loop까지 전달하고, 이미 `Enhanced`인 preview를
  새 Ink champion으로 재사용하는 반복 Recovery를 차단
- 다음 anchor의 Live-Wire tree가 준비되기 전 빠른 클릭이 임시 직선 chord를 저장하던
  race를 막고, 계산된 Ink 경로와 필요한 Recovery quality gate가 끝난 뒤 같은 click
  target을 확정
- Recovery inference 실패 뒤 남던 stale request를 제거해 같은 Ink champion에서
  명시적으로 다시 시도할 수 있도록 수정
- 손상된 Recovery regular object는 사용자가 `Repair Recovery Model`을 명시적으로
  누른 경우에만 격리·재다운로드하고, 실패·취소 시 미완료 object는 원본을 복원하되
  이미 hash 검증·게시된 replacement는 유지하며 symlink·directory·Windows junction/
  reparse point 등 unsafe cache는 자동 변경하지 않도록 수정
- Recovery install/repair task 등록 실패는 버튼·상태를 즉시 복원하고, 마지막 artifact가
  이미 검증·게시된 뒤 도착한 cancel flag는 성공한 store transaction을 취소로 오표시하지
  않도록 commit boundary를 고정
- Ctrl+Z와 checkpoint rewind가 진행 중 Recovery·Live-Wire·pending click을 함께
  무효화하고, polygon close는 elevation 취소·layer 저장 실패에도 확정 경로와 화면의
  Enhanced 후보를 바꾸지 않는 transaction으로 처리
- 개발 source의 기본 package 명령이 동결된 `0.1.5` ZIP을 덮어쓰지 못하도록 하고
  CI는 격리된 `--output` current-source ZIP만 생성·검사
- QGIS Plugins Website의 Bandit `B110`/`B112` false-positive 대상인 best-effort
  cleanup·fallback에 지점별 suppression 근거를 기록하고, model SHA-256·upstream
  commit pin을 credential과 구분하는 검토된 `.secrets.baseline`을 current-source
  package에 포함; 같은 범위를 고정 Bandit·detect-secrets CI로 재검사
- 실제 재배포 권리 자료와 독립 주석 검수가 완료될 때까지 benchmark의
  `publication_ranking_eligible`을 false로 고정하고, 시작·끝점이 같거나 기하 길이가
  0인 prompt/reference는 hash가 맞아도 거부

## 0.1.5 — experimental QGIS release (2026-08-26)

`0.1.5`는 공식 QGIS plugin 저장소에 experimental release로 공개됐습니다.
공식 다운로드는 1,483,635 bytes, 30 entries, SHA-256
`24f1def6acd63d483ea6bf7c20b944f56507ead52190667ec4f35562fca6c964`입니다.
29개 일반 plugin entry는 commit `0675cad`와 byte-identical이지만, upload ZIP의
`.secrets.baseline`은 해당 commit blob과 다릅니다. repository에 보존된
`dist/ai_vectorizer-0.1.5.zip`(SHA-256 `d2925198…`, 29 entries)은 commit
`89b9f20`의 이전 로컬 후보이며 공식 artifact identity가 아닙니다.

Git history의 `0.1.5–0.1.7`과 미공개 worktree의 `0.1.8` 표시는 추가 QGIS
릴리스가 아닌 개발 metadata였습니다. 이력은 rewrite하지 않으며,
Ink v2·Smart Recovery와 공개 이후 수정은 위 `0.1.6` 후보에서 관리합니다.

### Added

- 검은 인쇄 획을 한 픽셀 중심선으로 만드는 기본 Ink Centerline과 제한된
  direction-aware Live-Wire
- literal 0–100% assist, green path preview, anchor undo와 Freehand baseline
- contour elevation, Spot Heights, 실험적 linear-TIN DEM과 GDAL hillshade workflow
- 검증된 MobileSAM/SAM/HED model storage와 상태 진단
- 최종 ordered centerline을 평가하는 격리 benchmark 및 EfficientSAM-Ti ONNX
  benchmark adapter
- 결정적 release builder, Python/QGIS CI matrix와 QGIS runtime safety tests

### Changed

- 새 feature와 기존 contour 연장, elevation field·attribute 변경을 QGIS edit buffer와
  하나의 Undo command로 통합
- model을 plugin 폴더 대신 QGIS profile에 보존하고 고정 URL·크기·SHA-256,
  staged write, atomic publication과 rollback 적용
- 기본 ZIP은 추가 pip 없이 Ink/Freehand를 유지하고 OpenCV·SAM stack을 선택
  dependency로 분리
- release 버전을 `0.1.5`로 정규화하고 문서를 기능·구현·안전·검증 중심으로 재구성

### Fixed

- QGIS에 로드된 Shapefile과 같은 경로를 새 output으로 선택할 때 발생할 수 있던
  dataset 교체와 data loss 차단
- DEM/hillshade pair 게시 중 path race, 기존 output 변경과 partial result rollback
- model download·migration 실패 시 기존 checkpoint 손상과 symlink target 우회
- raster block의 과대한 allocation, 잘못된 dtype 해석, trace cost overflow와 NaN 입력
- plugin unload, layer removal, dialog 종료 뒤 남을 수 있던 task, timer, rubber band와
  preview resource 정리

### Known limitations

- 완성·독립 검수된 역사 지도 benchmark dataset과 제품 정확도 근거는 아직
  없습니다.
- topology QA, DEM uncertainty/NoData와 provenance sidecar는 아직 구현되지 않았습니다.
- 공식 0.1.5 source 시점의 [CI run 32924318782](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/32924318782)은
  QGIS 3.22/4.2 runtime job이 green이 아니었습니다. 후속 commit `30e18f6`의
  [run 33339770178](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/33339770178)에서
  3.22.16/3.44.13/4.2.1 current-source 행렬이 통과했지만 이는 공식 ZIP을
  소급해 다시 검증한 것이 아닙니다.
- Windows clean-profile GUI, `0.1.4 → 0.1.5` upgrade와 역사 지도 독립 검수는
  stable 전환·다음 release 전 남은 gate입니다. QGIS 3.22/3.44/4.2 원격
  matrix와 macOS 3.44.8 clean-profile current-source smoke는 통과했습니다.

## 0.1.4 — previous QGIS repository baseline

이 저장소 정리의 기준이 된 직전 QGIS plugin 저장소 버전입니다. 이전 변경의 세부
내용은 Git history와 QGIS plugin 저장소의 version record를 참고하세요.
