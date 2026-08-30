# Changelog

이 파일은 사용자가 체감하는 변경을 기록합니다. 배포 버전의 source of truth는
`ai_vectorizer/metadata.txt`이며, 실제 공개일은 QGIS plugin 저장소나 GitHub Release에
같은 artifact를 게시한 날에만 기록합니다.

## Unreleased

다음 공개판에 들어갈 변경을 이 아래에 먼저 기록합니다. 일상 개발 중에는 plugin
metadata 버전을 올리지 않습니다.

### Added

- 9·15·31 source-pixel, RGB/명도, source-grid tile 정규화를 사용하는 연속
  `LineEvidence`와 이를 직접 소비하는 Ink Live-Wire v2
- 기본 OFF의 `Smart Recovery (Experimental)`: 검증된 EfficientSAM-Ti mask를
  저신뢰 Ink 구간의 corridor prior로만 사용하고, 안전 검사를 통과한 challenger만 채택
- `Ink`, `Recovering`, `Enhanced`, `Ink fallback` 상태, 명시적 model 설치와 현재
  구간 재시도 UI
- worker request v2의 선택적 `previous_xy`, 네 개의 독립 benchmark method ID와
  8개 도엽·48개 crop 공개 dataset 권리·주석 검증 template

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
- 개발 source의 기본 package 명령이 동결된 `0.1.5` ZIP을 덮어쓰지 못하도록 하고
  CI는 격리된 `--output` current-source ZIP만 생성·검사
- QGIS Plugins Website의 Bandit `B110`/`B112` false-positive 대상인 best-effort
  cleanup·fallback에 지점별 suppression 근거를 기록하고, model SHA-256·upstream
  commit pin을 credential과 구분하는 검토된 `.secrets.baseline`을 current-source
  package에 포함; 같은 범위를 고정 Bandit·detect-secrets CI로 재검사
- 실제 재배포 권리 자료와 독립 주석 검수가 완료될 때까지 benchmark의
  `publication_ranking_eligible`을 false로 고정

## 0.1.5 — release candidate

`0.1.5`는 공식 QGIS plugin 저장소의 `0.1.4` 다음 공개 후보입니다. Git history의
`0.1.5–0.1.7`과 미공개 worktree의 `0.1.8`은 QGIS 저장소에 게시되지 않은 개발
metadata였습니다. 이력은 보존하고 그 변경을 하나의 `0.1.5` 후보로 정리했습니다.
공개일은 아직 확정하지 않았습니다.

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

- 실제 역사 지도 benchmark dataset과 제품 정확도 근거는 아직 없습니다.
- topology QA, DEM uncertainty/NoData와 provenance sidecar는 아직 구현되지 않았습니다.
- QGIS 3.22/3.44/4.2 remote package matrix와 clean-profile GUI 검증은 공개 전 남은
  gate입니다.

## 0.1.4 — latest QGIS repository baseline

이 저장소 정리의 기준이 된 직전 QGIS plugin 저장소 버전입니다. 이전 변경의 세부
내용은 Git history와 QGIS plugin 저장소의 version record를 참고하세요.
