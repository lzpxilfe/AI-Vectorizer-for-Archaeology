# 일반 글자 공백 실패 평가·경로 진단 — 2026-10-19

새 `O` 공백에서 자동 연결 후보가 생기지 않은 직접 원인은 **목표 글자 그룹 검출
누락**이었다. 세 방법의 일반 경로는 같은 우회를 유지했다. `O`의 성분은 검출기
후보를 통과하지만 `CH’IL-BONG` 그룹이 성립하지 않는다. 기존 `a`에는 별도의
창 전체 continuity gate가 있었다. 이번 변경은 이 원인을 관측하는 평가·로그이며,
검출 조건·경로 선택·모델·기본 OFF 설정은 바꾸지 않았다.

## 평가 범위와 원본 동결

기존 validation/development의 원본 native512 창 21개를 검토했다. 새 학습이나
공식 Calibration/Holdout 열람은 하지 않았다. 출처·권리·도엽 좌표·이미지 해시는
[`crop index`](data/korea-general-text-crop-index-v1.json)에 유지한다. 새 픽셀 파일은
저장소에 추가하지 않았다. 평가셋은 다음 6개 클릭 쌍의 **개발 진단 파일럿**이다.
6개 모두 확인된 실패이거나 독립적인 지도라는 뜻은 아니다.

| 원본 글자·쌍 | 원본에서 고정한 분류 | 새 모델 after 분기 | 목표 영역의 기록된 proposal |
|---|---|---|---|
| CH’IL-BONG의 파란 `O` 왼쪽 획 | 같은 선으로 추정 가능한 양성 1건 | `observed_fallback` | 9개 중 겹침 0개 |
| 같은 시작점→따로 보이는 아래 이웃 선 | 다른 선인 음성 1건 | `pixel` | 9개 중 겹침 0개 |
| 파란 Surach’on | 짝 불확실 | `observed_fallback` | 6개 중 겹침 0개 |
| 검정 `邑` | 짝 불확실 | `pixel` | 6개 중 겹침 0개 |
| 검정 `壬` | 짝 불확실 | `pixel` | 검출 영역 0개 |
| Odae의 파란 N/G·격자 주변 | 짝 불확실 | `pixel` | 기록된 16개 중 겹침 2개 |

[`실행 fixture v1`](data/korea-general-text-failure-cases-v1.json) 5건과
[`추가 fixture v1`](data/korea-general-text-failure-extra-cases-v1.json) 1건은 모델 출력
열람·실행 전에 원본만 보고 고정했다. 붉은 클릭·가시 팔·가려진 사각형·별도 이웃 선은
[`원본 검토 기록`](data/korea-general-text-failure-raw-assessment-v1.json)에 결속했다.
사람 승인 0건, 학습 사용 불가, 성능 순위 발표 불가다. 음성·불확실 5건과 숨은 곡선은
정확도 점수가 없다. bbox 겹침만으로 올바른 선 끝점이나 곡선을 확정하지 않는다.

원본 가시 참조가 클릭 앞 1픽셀·끝 클릭 뒤 약 2.24픽셀까지 포함한 문제를 출력 열람
전에 발견했다. QGIS 실행은 이미 시작된 상태였으므로 원래 fixture·결과·점수를
보존하고, [`v2`](data/korea-general-text-failure-cases-v2.json)를 **실행 후·출력 검토 전
참조 범위 수정 명세**로 분리했다. v2는 두 freeze 플래그가 모두 false이며 실행
fixture라고 주장하지 않는다. 동일 v1 저장 경로에 클릭 사이의 가시 참조만 재채점했고,
`derivation.json`에 실행 v1 SHA·채점 v2 SHA·결과 SHA를 따로 기록했다. 분류·클릭·
전체 원본 가시 팔·가려진 영역·이웃 선은 같다. strict fixture loader는 완화하지 않았다.

## 재현 조건과 검증

실제 QGIS 3.44.8, 동일 unreleased ZIP, 원본 RGB·native cell-centre 마우스 클릭,
CPUExecutionProvider·ORT 1.31.0·1/1 thread·sequential 조건이다. `Ink`, 기존 Korea,
새 일반 글자 모델의 after 결과끼리 비교한다. 과거 도구 키 `usgs_after`는 이번에도
**old-model 인자로 넣은 기존 Korea**이며 USGS 전용 모델이 아니다. `korea_after`가
새 일반 글자 모델이다. learned support 0.8, avoidance 0, 격자 회피 ON,
자동 가림 연결 ON, Smart Recovery OFF를 동일하게 적용했다.

- ZIP SHA-256: `8fe6814901b241dde814cfa0f84ee7b7613a577d00e646d1f00e9bb725afa4dc`.
  실행 ZIP의 Python 46개 파일이 현재 제품 source와 일치하고 package check를 통과했다.
- 실제 QGIS import와 runtime **178 tests, 실패·오류·skip 0**. 일반 Python의 skip으로
  QGIS 검증을 대신하지 않았다.
- 일반 Python suite는 **1,215 passed, 160 skipped, 171 subtests passed**였다.
  QGIS 의존 skip과 실제 QGIS 검사 결과를 별도로 기록한다.
- 기존 17건 × before/after × 세 방법의 **102개 저장 경로와 입력이 진단 추가 전과
  정확히 같다**. 새 6건은 36개 trace를 수집했다. 새 6건 전체가 old/new 동일하다는
  뜻은 아니다. `邑`·`壬`의 일반 경로는 다르지만 정답 불확실이라 우열을 매기지 않는다.
- 새 6건의 한 번 Undo·격리 profile 제거를 확인했다. 양성 O는 양쪽 클릭 오차 0이다.
  음성 O·불확실 壬·Odae에는 각각 끝 클릭과 3.162/6.083/6.083px 차이가 있는 원본
  snapping 경로가 저장됐다. 음성 후보 제외는 클릭 간 일반 경로 자체를 거부한다는 뜻이 아니다.
- 모델 접근 단계에서 멈춘 최초 실행 로그는 보존했다. 같은 SHA·bytes의 모델을 작업용
  임시 폴더에 복사한 뒤 전체 검증을 완료했다. 모델 저장소 guard·권한은 바꾸지 않았다.

## O: 모델이 달라도 같은 우회가 나오는 지점

클릭은 XY `[135,180]`→`[151,182]`, 가려진 영역은 `[138,178,144,184]`다.
Ink·기존 Korea·새 모델의 after 저장 좌표가 완전히 같다. **가시 F1@2px 0.180,
가시 p95 8.932px**로 글자 `O`의 획을 돌아간다. 이 값은 클릭 범위로 자른 v2 재채점이며,
원래 v1 점수 F1 0.273도 남겼다. 숨은 곡선의 모양을 맞혔는지는 채점하지 않았다.
지정한 아래 이웃 선까지 최소거리 2.030px로 1px 접촉은 없지만, 전체 이웃 선에 대한
무접촉 보장은 아니다.

학습 보조는 실제로 적용됐다. O 창에서 기존/새 모델이 원본 confidence보다 강한
픽셀은 각각 76,876/76,893개다. 원본 skeleton 위치에서 confidence가 추가로 상승한
픽셀은 둘 다 0이고,
최종 분기는 `observed_fallback`이다. 모델은 soft 비용 지지이며 글자 그룹 검출을
대체하거나 실제 끝점 조건을 해제하지 않는다. 로그의 복합 ordinary 이유
`pixel_detour_or_collinear_gap_rejected`만으로 둘 중 어느 항인지 단정하지 않는다.

O에서는 continuity bridge가 0이라 detector가 정상 호출됐다. 반환된 9개 bbox를
tree 원점 `[0,20]`으로 원본 좌표에 옮기면 y=202–270이며 목표와 겹치는 영역은 0개다.
가장 가까운 bbox도 20.224px 떨어져 있다. 창 전체의 `connector_rejected`, 738쌍 중
735개 `arm_tangent_unavailable`는 **다른 글자 영역**의 집계다. 이를 O 끝점 거부
원인으로 읽어 tangent 검사를 완화하면 잘못된 수정이 된다.

동일 cached RGB의 현재 검출기를 읽기 전용으로 재구성한 결과:

- O 성분은 원본 bbox `[138,175,158,193]`, 면적 128px, skeleton 53px이며 크기·면적
  필터를 통과한다. 다만 독립 분기가 없는 모양이다.
- `CH’IL-BONG`의 허용 성분 중 분기 모양은 H 하나다. `_candidate_pairs`는 양쪽 모두
  분기한 성분을 seed로 요구하고 `_find_groups`도 분기 성분 2개 이상을 요구하므로
  이 단어에 seed 쌍이 0개다. O 자체를 크게 못 찾는 문제가 아니라 **그룹 생성 조건**이다.
- 회색 채널의 O 모양은 파란 잉크 비율 약 0.723으로 blue exclusion에 걸린다.
  이는 색 채널 중복 억제이며 회색 threshold만 바꿔 해결될 근거는 없다.

## 기존 a: 별도의 창 전체 gate

기존 `a`의 검출기는 호출되기 전에 `continuity_bridge`로 건너뛴다.
root component에서 생성된 bridge 28픽셀 중 19픽셀이 confidence를 올리지만,
목표 공백에서 최소 72.56px, 저장 일반 경로에서 64.54px 떨어져 있다. 그럼에도
창 전체 `np.any(...)`가 observed·숫자·글자 builder를 모두 끈다.
**이 a는 현재 가시 F1 1.000인 보존 사례**다. gate 범위가 거칠다는 증거이며,
gate를 바꾸면 성능이 좋아진다는 실험 결과는 아니다. O 실패의 원인과도 다르다.

## 구현한 로그와 재실행

`build_livewire_tree(..., diagnostics={})`와 `tree.trace_with_diagnostics(target)`가
검출 채널·성분/그룹 거부·region별 후보 거부·native/학습 혼합·실제 선택 분기를 기록한다.
build snapshot은 불변 JSON 문자열이고 target 진단은 호출마다 새 dict다. 정상 tool의
`trace_diagnostics_enabled`는 기본 false이며 UI나 상시 파일 기록을 추가하지 않았다.
진단 OFF에서도 작은 helper 호출은 생기므로 무비용이라고 주장하지 않는다.
영역 기록은 최대 16개·검출 채널은 최대 2개·자동 연결 target 시도는 최대 2개로 제한한다.

기존 `qgis_grid_comparison.py`의 `--trace-diagnostics-dir`가 작업 worker의 기록을 켠다.
요약기는 실행 fixture SHA·case identity를 검사하며 before/after 혼합 없이 after만 읽는다.

```bash
python scripts/summarize_gap_diagnostics.py \
  --result /path/to/result.json --scores /path/to/scores.json \
  --fixtures learning/data/korea-general-text-failure-cases-v1.json \
  --output /path/to/new-decision-summary.json
```

`text_stage`의 builder 거부는 창 전체 범위임을 명시한다. 별도 `text_localization`이
원점 보정 bbox·가려진 영역 겹침·거리·기록 완전성을 표시한다. bbox 겹침은 끝점 정답이
아니며 일부 기록만 남은 창에서 미검출을 확정하지 않는다. v2 파생 채점은 strict 실행
요약으로 위장하지 않고 별도의 derivation을 사용한다.

## 다음 수정 순서

1. **O의 단어 그룹 seed 조건을 좁게 보완한다.** 2개 branched 성분에만 의존하지 않는
   보조 그룹 증거를 먼저 별도 실험한다. 예를 들어 분기 성분 1개와 여러 compact 성분의
   정렬·간격·색·반복 기호 제외를 함께 요구한다. O 단독으로 threshold를 푸는 방식은
   피한다. 참조 bbox를 이용한 oracle 주입은 진단용에만 쓰고 제품 입력에 넣지 않는다.
2. **후보가 O를 덮은 다음** 실제 원본 끝점·가시 팔·후보 유일성·옆 선·표시 접촉 검사를
   그대로 적용해 어디서 거부되는지 다시 측정한다. 검출 성공을 경로 성공으로 세지 않는다.
   음성 O와 보류 4건을 함께 재실행하며 보류 사례에 임의 정답을 주지 않는다.
3. **continuity gate의 대상 범위는 별도 실험으로 분리한다.** 현재 a/숫자/深 U자 보존과
   새 O 효과를 섞지 않고, bridge가 현재 경로·목표에 영향을 주는 조건을 검증한다.
   창 gate를 무조건 삭제하는 수정은 아직 근거가 없다.
4. 위 경로 병목을 해소한 후 독립 개발 crop·사람 검수 양성/음성을 늘린다.
   실제 letter 잉크·흰 여유·접촉 구조와 현재 `word_gaps` 증강 차이를 비교한 뒤
   추가 학습을 결정한다. 이번 6건은 학습에 넣지 않고 공식 Holdout도 열지 않는다.

채택 gate는 O의 가시 팔·정확 클릭 개선, 지정 이웃 선 무접촉, 음성 자동 후보 제외,
기존 17건 좌표·Undo 보존과 CPU 비용 측정이다. 이번 작업에서 성능 개선이나
기본 모델 교체는 주장하지 않는다.

## 근거 파일

공개 가능한 좌표·해시·요약은
[`diagnosis JSON`](data/korea-general-text-failure-diagnosis-2026-10-19.json)에 결속한다.
원본/실행 산출물은 저장소 밖 기존 task workspace의
`work/unseen-korea-text-gap/failure-evaluation-2026-10-19/`에 보존했다.

- `validated/existing-validation-execution.json`: 전체 실행·102개 동일성·profile 정리
- `validated/regression-*/result.json`, `scores.json`, `decision-summary-localized.json`:
  변경하지 않은 실행 결과·원래 점수·좌표 보정 진단
- `validated/derived-visible-reference-v2/`: 수정 참조 점수·전체 우회 비교 PNG·SHA derivation
- `general-text-target-local-public-evidence.json`, `o-glyph-group-forensics.json`:
  창 gate와 목표 검출 miss의 읽기 전용 재구성
- `reference-scope-pre-review-audit.json`, `v2-provenance-clarification.json`:
  참조 범위·freeze metadata 수정 감사
- `final-pure-tests.log`, `validated/qgis-runtime.log`, `work-state.json`: 검증·인수인계
