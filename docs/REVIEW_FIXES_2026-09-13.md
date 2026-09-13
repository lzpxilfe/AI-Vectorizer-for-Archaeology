# 전체 코드 리뷰 후속 수정 — 2026-09-13

리뷰 기준 `c52639f`에서 확인된 네 문제를 수정했습니다. 배포 metadata는 0.1.6을 유지하며
기존 Desktop ZIP을 덮어쓰지 않습니다.

| 문제 | 변경 | 검증 |
| --- | --- | --- |
| DEM 계산 중 생성·변경된 출력 덮어쓰기 | 승인 전에 출력·sidecar의 존재와 파일 상태를 request에 보관하고 게시 직전 비교 | 생성·수정·교체·삭제·sidecar 출현 모두 거부; 다른 출력과 사용자 파일 보존; 승인된 동일 파일은 교체 가능 |
| 연장 대상 피처의 변경 | 시작 클릭에서 원본 WKB를 보관하고 저장 전 비교 | 이동·방향 반전·내부 정점 변경·삭제 시 거부; 사용자 편집·확정 앵커·Undo index 보존; 정상 연장과 단일 Undo 유지 |
| 반 픽셀 지도 위치 오차 | float 인덱스의 원점을 첫 픽셀 중심으로 통일; 정수 선택은 containing cell 기준; 회피 영역도 정렬 | 비정방 픽셀과 raster 경계, 실제 GeoTIFF → cache → 다른 canvas CRS → 저장된 벡터 좌표 확인 |
| Recovery가 보조 강도를 무시 | worker에 강도 snapshot 전달, smoothing 뒤 blend하고 그 결과를 평가·표시 | 0·10·50·100%에 따른 곡률, endpoint 보존, 실제 arbiter에 전달된 경로 확인; benchmark에도 동일 순서 적용 |

Recovery benchmark adapter는 `ink-v2-effsam-recovery-worker/2`로 구분합니다. 보조 강도
적용 전의 안전 판정을 그대로 재사용하지 않으며, 변경된 경로의 hash를 결과에 기록합니다.

## 실행 결과

- Python 3.12 / NumPy 1.26.4 / SciPy 1.15.3 / scikit-image 0.24.0: **557 passed, 86 skipped**.
- 격리된 macOS QGIS 3.44.8의 runtime safety / manual gap / source safety: **108 tests OK**.
- 현재 소스의 결정적 ZIP 생성·동기화 확인 및 clean-profile 설치 smoke 여섯 시나리오 통과.
- 모델 다운로드는 수행하지 않았습니다. Recovery 강도 검사는 결정적 후보 경로와 실제
  arbiter를 사용하며, 실제 ONNX 추론 품질을 측정하는 실험은 아닙니다.

기존 모델·고지도 정확도의 일반화 한계는 그대로입니다. 이번 검증은 데이터 보존,
좌표 정합과 사용자 보조 강도 동작을 대상으로 합니다.
