# Changelog

이 파일은 사용자가 체감하는 변경을 기록합니다. 배포 버전의 source of truth는
`ai_vectorizer/metadata.txt`이며, 실제 공개일은 QGIS plugin 저장소나 GitHub Release에
같은 artifact를 게시한 날에만 기록합니다.

## Unreleased

다음 공개판에 들어갈 변경을 이 아래에 먼저 기록합니다. 일상 개발 중에는 plugin
metadata 버전을 올리지 않습니다.

## 0.1.5 — release candidate

### Packaging security-scan remediation

- QGIS Plugins Website의 Bandit `B110`/`B112` false-positive 대상인 best-effort
  cleanup·fallback에 지점별 suppression 근거를 기록했습니다. 원래의 오류 보존 및
  fallback 동작은 바꾸지 않습니다.
- 모델 artifact SHA-256과 upstream commit pin 9개를 검토한
  `ai_vectorizer/.secrets.baseline`에 등록하고, release ZIP에 포함했습니다. 이는
  credential이 아니라 다운로드 무결성 검증 값입니다.

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
