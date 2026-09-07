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
- 보조 강도를 곡선에 적용합니다. 가까운 중심선을 먼저 읽고, 주변 방향이 충돌하거나
  공백 진행 방향과 맞지 않으면 수동 bridge 요청을 거부합니다.

## 합성 fixture 결과와 한계

기존 `manual_gap_shadow`의 p95 0.67px는 미리 지정한 tangent로 만든 기하 테스트
결과였습니다. 실제 QGIS sampler의 결과로 해석할 수 없습니다. 이제 두 경로 모두
동일한 endpoint 구간으로 평가하며, reference는 생성이 끝난 선을 채점할 때만 읽습니다.

| 조건 | 사례 수 | 제품과 같은 sampler 결과 |
| --- | ---: | --- |
| 기존 글자 인접 prompt, 갈색/무채색, 0°/90° | 4 | 방향 불명확으로 거부, 기존 Ink 경로/hash 그대로 유지 |
| 방향이 분명한 공백, 무채색, 0°/90° | 2 | 연결 preview 생성, 공백 coverage 100%, 평행선 전환 0 |

첫 네 사례의 어려운 공백은 **아직 해결되지 않았습니다**. 뒤 두 사례는 기존 Ink도
통과하는 실행 확인용 예시이며 난제의 정확도 개선을 입증하지 않습니다. 이 결과는
고지도 holdout, 모델 순위 또는 실제 사용자 작업 시간 개선의 근거가 아닙니다.

다음 명령은 여섯 개 무손실 PPM과 `result.json`을 만듭니다. JSON에는 실제 경로,
입력 prompt, 추정 tangent, 설정·소스·reference hash, dependency version이 포함됩니다.

```bash
python -m benchmarks.manual_gap_shadow --output-dir "$(mktemp -d)"
```

## 실행 검증

- Python 3.12 전체 suite: **531 passed, 77 skipped, 137 subtests passed**.
  일반 Python의 QGIS skip은 아래 실제 런타임 실행으로 별도 검증했습니다.
- macOS **QGIS 3.44.8**: 기존 runtime safety **57/57**, 새 manual gap **17/17**.
  새 테스트에는 실제 detector → 방향 추정 → preview → 표시 경로 확정과 난제 거부가
  포함됩니다. QGIS CI 매트릭스에도 새 suite를 추가했습니다.
- 현재 소스 ZIP 생성·동기화 검사 및 새 QGIS 프로필의 plugin loader 검사 통과.
  Freehand, 0%, Ink, 빠른 클릭, cache 계산 중 클릭, Enhanced preview 확정의
  여섯 시나리오를 실행했습니다. Recovery model은 설치되지 않은 상태였습니다.
- 검증한 current-source ZIP SHA-256:
  `99bc6ad5b8aef787e131ca0b6d4ed42e3db419e4e6148f42b7f1dd9cd6c1a266`.
  이 ZIP은 동결된 0.1.6 release candidate와 별도의 임시 build입니다.

향후 우선순위는 글자 옆에서도 같은 등고선의 방향을 읽는 방법과, 실제 고지도에서
오연결·추가 anchor·Undo를 함께 측정하는 것입니다.
