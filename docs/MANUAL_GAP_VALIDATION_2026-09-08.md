# 수동 라벨 공백 연결 검증 — 2026-09-08

`Unreleased` source의 공백 연결을 제품 코드와 같은 입력·방향 추정으로 검증했습니다.
동결된 0.1.6 후보 ZIP·metadata·tag는 변경하지 않았습니다.

## 수정한 동작

- `G`로 현재 커서 위치의 연결을 요청하고, `Esc`로 proposal만 취소합니다.
  hover는 proposal을 유지하며 끝점 클릭으로만 채택합니다.
- 같은 cache generation, anchor, 표시 경로인지 검사하고 화면/원본 모두 2px 이내의
  끝점 클릭만 허용합니다. Alt/Shift/Ctrl은 bridge를 확정하지 않습니다.
- 늦은 Live-Wire 완료와 Recovery 재시도가 proposal을 덮어쓰지 않습니다.
  source/cache 변경, 닫기 취소, 저장 실패, 좌표 변환 실패를 검증했습니다.
- 보조 강도를 곡선에 적용합니다. 먼저 3px 원의 가까운 중심선을 읽고, 주변 방향이
  충돌하거나 공백 진행 방향과 맞지 않으면 두 명시적 anchor의 공백 반대편 12px
  one-sided centerline 지지를 별도로 검사합니다. 양쪽 모두 가까운 지지, 4px 이상
  span, 80% 이상의 주축 집중도, 45° 이내 chord 정렬을 보일 때만 그 방향을 사용합니다.
  숫자가 있는 공백 쪽은 탐색하지 않으며, 지지가 부족·분산되거나 평행선만 보이면
  기존 Ink를 유지합니다.

## 합성 fixture 결과와 한계

기존 `manual_gap_shadow`의 p95 0.67px는 미리 지정한 tangent로 만든 기하 테스트
결과였습니다. 실제 QGIS sampler의 결과로 해석할 수 없습니다. 이제 두 경로 모두
동일한 endpoint 구간으로 평가하며, reference는 생성이 끝난 선을 채점할 때만 읽습니다.

| 조건 | 사례 수 | 제품과 같은 sampler 결과 |
| --- | ---: | --- |
| 기존 글자 인접 prompt, 갈색/무채색, 0°/90° | 4 | local sampler는 글자 방향 때문에 거부; one-sided fallback으로 preview 생성, coverage 100%, 평행선 전환 0 |
| 방향이 분명한 공백, 무채색, 0°/90° | 2 | 연결 preview 생성, 공백 coverage 100%, 평행선 전환 0 |
| 숫자+원격 평행선만 있고 대상 등고선 없음, 무채색, 0°/90° | 2 | fallback이 outward 지지를 찾지 못해 거부, 기존 Ink 경로/hash 그대로 유지 |

첫 네 사례는 같은 명시적 prompt·이미지·제품 kernel에서 어려운 숫자 인접 공백을
통과합니다. 방향은 reference를 보지 않고, endpoint pair가 정한 공백의 **바깥쪽**
centerline만으로 추정합니다. 뒤 두 사례는 기존 Ink도 통과하는 실행 확인용 예시입니다.
이 결과는 여전히 고지도 holdout, 모델 순위 또는 실제 사용자 작업 시간 개선의 근거가
아닙니다.

다음 명령은 여섯 개 무손실 PPM과 `result.json`을 만듭니다. JSON에는 실제 경로,
입력 prompt, 추정 tangent, 설정·소스·reference hash, dependency version이 포함됩니다.

```bash
python -m benchmarks.manual_gap_shadow --output-dir "$(mktemp -d)"
```

## 실행 검증

- 앞선 baseline에서 Python 3.12 전체 suite는 **531 passed, 77 skipped, 137 subtests
  passed**였고 QGIS 3.44.8 runtime safety는 **57/57**이었습니다. 이 수치는 아래
  one-sided fallback 변경 전의 baseline이며, 현 변경의 전체-suite 결과로 재사용하지
  않습니다.
- macOS **QGIS 3.44.8** 격리 프로필에서 manual gap suite **18/18**을 다시 실행했습니다.
  여기에는 실제 detector → local 실패 → outward sampler → preview → 표시 경로 확정이
  포함되며, 숫자와 원격 평행선만 있는 negative control이 기존 Ink를 유지하는지도
  확인합니다.
- product-parity CLI를 두 번 실행해 `result.json`과 출력 JSON이 byte-identical임을
  확인했습니다. 네 glyph-adjacent fixture 모두 preview/coverage 100%/parallel switch 0,
  두 clear-gap control도 preview를 유지했고, 두 glyph-parallel negative control은
  bridge를 만들지 않았습니다.
- 새 current-source ZIP을 별도 임시 경로에 생성·동기화 검사했고, SHA-256은
  `132b834167c60454f2a2eb6af16065725b29b33e67f2fc4bb4579c4b0ac38486`입니다. 새 QGIS
  profile의 plugin loader smoke도 통과했습니다. Freehand, 0%, Ink, 빠른 클릭, cache
  계산 중 클릭, Enhanced preview 확정 여섯 시나리오와 Recovery model 없는 기본 경로를
  확인했습니다. 동결된 0.1.6 release candidate는 변경하지 않았습니다.

향후 우선순위는 실제 고지도 holdout에서 오연결·추가 anchor·Undo를 함께 측정하고,
한 endpoint의 바깥쪽 지지가 실제로 끊긴 사례에서도 fail-closed 동작을 유지하는지
검증하는 것입니다.
