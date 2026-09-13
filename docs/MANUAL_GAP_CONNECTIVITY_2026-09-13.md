# 라벨 공백의 연결 지지 검증 — 2026-09-13

## 문제와 변경

기존 `a712654`의 contextual 방향 추정은 바깥쪽 12px cone 안의 중심선 점들을
모두 평균냈습니다. 분리된 잡점, 짧은 가까운 조각과 먼 조각, 앵커에서 같은 거리의
평행선도 하나의 방향을 반환했습니다. 기하적으로 가늘게 모였다는 사실만으로는
같은 선이라고 판단할 수 없었습니다.

이제 반경 안의 8-connected component를 분리합니다. 가장 가까운 component만
사용하고, 다른 component가 최근접 거리 + 1px 이내면 방향을 반환하지 않습니다.
가까운 조각이 짧다고 먼 조각을 대신 선택하지 않습니다. 한 component 내에서도
연속된 두 1px 진행 구간의 가로 폭이 각각 2.5px를 넘으면 분기로 보고 거부합니다.
기존 score·span·주축·chord 정렬·곡선 길이 검사는 유지합니다.

일반 Ink 검출, 자동 Live-Wire 경로, endpoint 위치와 사용자 확인 동작은 변경하지
않습니다. 이 검사는 작은 local sampler가 실패한 뒤 사용하는 contextual fallback에
적용됩니다.

## 재현과 결과

- 변경 전 core 입력의 잡점·분리 조각·평행선 × 0°/90° 여섯 경우가 방향을 반환해
  거부 회귀 테스트에 실패했습니다. 변경 후 모두 거부합니다.
- 원래 gray glyph fixture의 `[110:125, 108:113]`을 종이색으로 지우면 짧은 scan loss가
  생깁니다. 기존 sampler는 분리된 조각을 합쳤지만 새 sampler는 거부합니다.
- 같은 fixture의 `(101,114)` 부근에서 `(112,119)`까지 추가한 회색 획이 바깥쪽 분기를
  만듭니다. 기존 sampler는 평균 방향을 반환했지만 새 sampler는 거부합니다.
- 기존 endpoint와 reference를 유지한 두 raster 변형과 90° 회전을 suite에 추가했습니다.
  제품과 같은 sampler·builder·assist blend에서 여섯 positive preview를 유지하고,
  여섯 negative는 기존 Ink route/hash를 그대로 유지합니다.
- reference를 변경해도 성공·실패 사례의 생성 경로/hash는 변하지 않는지 검사합니다.
- 실제 QGIS에서 새 네 negative의 preview 요청 실패가 기존 preview와 확정 anchor를
  보존하는지 검사합니다. 기존 detector → preview → endpoint 확정 테스트도 유지합니다.

```bash
python -m benchmarks.manual_gap_shadow --output-dir "$(mktemp -d)"
python -m pytest -q tests/test_manual_gap_bridge.py tests/test_manual_gap_shadow.py
```

현재 Python 3.12 / NumPy 1.26.4 / SciPy 1.15.3 / scikit-image 0.24.0에서 전체 suite
**555 passed, 81 skipped**. 격리된 macOS QGIS 3.44.8에서 runtime safety·manual gap·source
safety **103 tests OK**. CLI 회귀는 두 번 생성한 JSON과 route/hash의 결정성을 확인합니다.

## 검증 범위

합성 fixture와 제한된 반경의 topology 검사입니다. 실제 고지도 독립 holdout이나
모든 글자·교차부 구분을 입증하지 않습니다. 실제 선에 스캔 누락이 있으면 제안을
거부하는 경우가 늘 수 있으며, 이때 사용자는 다른 anchor 또는 Freehand로 이어갈 수
있습니다. 이미 Desktop에 만든 0.1.6 배포 준비 ZIP에는 이 후속 변경이 들어 있지 않습니다.
