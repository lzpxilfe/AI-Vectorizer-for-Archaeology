# 학습 보조의 600 U자 후속 점검 — 2026-10-16

비교 기준은 `main`의 `7b48d39`와 이전 대각선 수정 ZIP
`b7d96e84552f4e6837ac62b91e98ea36a44864b3bc80a464569c158bb155db9d`다.
모델·데이터·공식 버전은 변경하지 않는다. 기존 가시선 참조·unknown 영역·클릭을
고정하고 이전/현재 ZIP을 각각 일회용 QGIS profile에서 비교한다.

## Ink보다 못해진 원인

원본 Byte RGB, 기존 native evidence, 원본 좌표에 고정된 ONNX 출력과 격자 비용을
사용해 경로를 분리했다. 이 진단에는 평가 참조나 unknown 사각형을 사용하지 않았다.
시작 XY `(225,326)`, 목표 `(252,324)`의 실제 공백 끝점은 `(226,339)`와 `(242,336)`이다.

Ink의 일반 경로는 48개 점으로 양쪽 실제 끝점을 모두 지난다. 두 학습 보조의 일반
경로는 28개 점으로 위쪽을 가로질러 첫 끝점에서 각각 11.40px·11.05px, 반대 끝점에서
8.06px 떨어진다. 자동 연결의 별도 곡선은 세 방법 모두 정확히 같은 87개 점이었다.
따라서 두 모델의 곡선 모양이나 끝점 선택이 달라진 것이 원인은 아니다.

경로 비용은 다음과 같았다. 같은 방법 안의 두 경로를 비교하는 진단 값이다.

| 방법 | 원본 Ink 경로 비용 | 짧은 학습 경로 비용 |
| --- | ---: | ---: |
| Ink | 80.62 | 84.14 / 83.89 |
| USGS guide | 68.55 | 55.26 |
| 한국 guide | 67.31 | 58.03 |

Ink 행의 두 값은 USGS·한국 guide가 각각 선택한 경로를 Ink 비용으로 측정한 값이다.

두 guide는 빈 곳까지 퍼지는 지지 점수로 짧은 경로의 native 밖 12개 점을 보강했다.
그 점들의 관측 지지 평균은 0.129였지만, 학습 지지를 합친 값은 0.613·0.568이 됐다.
그 결과 원래 좋은 Ink 경로가 문자 공백 보존 검사에 들어오지 못했다. 학습 경로는
실제 끝점을 건너뛰어 검사에서 탈락하고, 공통 자동 곡선이 선택되어 평활화도 생략됐다.
Ink만 좋은 일반 경로와 기존 평활화를 보존하면서 가시선 오차의 차이가 남았다.

## 수정 범위와 이웃 선 보호

학습 지지가 있고 원본에서 유효한 문자 공백 후보가 만들어진 worker에서만, 학습
지지를 제외한 Ink 참조 tree를 추가로 준비한다. 원본 evidence·격자·입력 방향과
같은 window를 사용하고 자동 문자 검출을 재귀 실행하지 않는다.

커서가 다음 조건을 모두 만족하는 공백에 도달했을 때만 이 참조를 검토한다.

- 원본 그래프에서 확인된 실제 끝점 쌍과 연결 후보가 있다.
- 참조가 그 두 끝점을 순서대로 지나고, 후보와 마스크 밖 가시선을 양방향 1.25px
  안에서 유지한다. 루프·복수 숫자 통과·과도한 우회는 기존 검사로 거부한다.
- 현재 보조율을 적용한 표시 경로가 옆 관측선과 접촉하지 않는다.

통과하면 같은 참조 경로에 현재 보조율과 기존 Ink 평활화를 적용한다. 모호한 목표나
다른 등고선의 끝점에는 참조가 새 연결 권한을 주지 않는다. 수동·색상·학습 회피 비용이
활성화된 문맥, 0%, 검출 OFF에는 이 참조를 준비하지 않는다. 준비가 실패하면 기존
경로 선택을 유지하고 취소는 전파한다. 커서 이동 처리에서 모델 추론·graph build는 하지 않는다.
기존의 전역 `learned_route_guard`·관측 gate는 켜지 않는다.

참조를 항상 Ink로 우선하는 방식이 아니므로, 정상 학습 경로와 유효한 공백이 없는
사례에는 기존 선택을 유지한다. 제품은 사례 ID, 평가 참조나 unknown 영역을 읽지 않는다.

## 비교·재현 기록

실제 QGIS 3.44.8의 NumPy 중심선 backend에서, 이전/현재 ZIP에 같은 모델과 일반
마우스 클릭을 사용했다. native 픽셀 하나를 화면 2px로 맞추고 실제 입력 지도 좌표의
일치를 검사했다. 경로 주입·`G`·직접 저장 주입은 없다.

| 600 U자 방법 | 가시 F1@2px 전 → 후 | 가시 p95 전 → 후 |
| --- | --- | --- |
| Ink | 1.000 → 1.000 | 0.558 → 0.558px |
| USGS guide | 0.957 → 1.000 | 2.750 → 0.558px |
| 한국 guide | 0.957 → 1.000 | 2.750 → 0.558px |

수정 후 두 guide의 실제 저장 좌표는 기존 Ink와 완전히 같다. 600 U자의 Ink,
700 세 방법과 대각선 세 방법도 이전 저장 좌표를 그대로 유지했다. 기존 7건 ×
세 방법의 21쌍도 같았다. 잘못된 이웃 선 두 쌍 × 세 방법은 자동 후보가 없으며
기존 fallback 좌표가 같았다. 음성 쌍에는 가시선 정확도 점수를 부여하지 않았다.
잘못된 클릭에서 기존 Ink fallback 자체가 이웃 선을 막는다는 뜻은 아니다.

총 12건 × 세 방법의 36쌍 중 변경은 두 guide의 600 U자뿐이다. 모든 실제 입력이
같고 저장·한 번의 Undo를 통과했다. 네 QGIS 비교 프로세스는 종료 0이었으며
임시 profile도 제거했다. 대각선의 교정된 바깥 쌍과 모든 기존 참조·unknown 영역을
수정하지 않았다.

참조 준비의 비용도 측정했다. 준비된 evidence·격자·모델 prior를 재사용한 Mac의
별도 Python 3.12·NumPy 2.5.4·SciPy 1.18.1에서 두 번 준비 실행 후 다섯 번씩
전후를 교대로 측정했다. 이 한 320px window의 tree 준비 중앙값은 USGS
62.64→96.79ms, 한국 62.37→96.74ms였다. 모델 추론·입력 읽기는 제외했으며 실제
QGIS 전체 지연이나 일반 성능 보장이 아니다. Ink에는 새 참조가 없었다.

시험·최종 ZIP·소스·모델·원본 진단·실제 QGIS 결과 해시는
[기계 판독 기록](data/learned-u-followup-2026-10-16.json)에 결속한다.
일반 pytest는 1,040 통과·155 skip·171 subtest 통과였다. 최종 ZIP의 Python
44개는 현재 소스와 실제 비교 ZIP의 코드에 바이트 단위로 같았다. 이 ZIP을 푼 실제
QGIS import와 163개 시험은 skip 없이 통과했다. clean-profile 9개 흐름도
경고 0·종료 0으로 통과하고 임시 profile을 제거했다.

추가 회귀 시험 17개는 검출·취소·0%·회피·부분 보조·실패 fallback과 커서 처리에서
추가 graph build가 없는지 확인했다. 실제 QGIS 추가 시험은 기존 Ink와 평활화 후
미리보기 및 저장 좌표가 일치하고, 이웃 선 접촉 검사와 한 번의 Undo를 통과했다.
새 최종 ZIP SHA-256은
`14890ade7fad9850c49ebd41a96071d4df852f95004e8d719fb204781e173004`이며,
실제 전후 비교 ZIP은 `31e1fcd0e89ddd87a89cac75f8476ee426d86a8a2e60d1149bd8a1bca62504ac`다.
차이는 offline README이며 Python 코드는 모두 같다. 공식 artifact와 이전 ZIP은 보존했다.

숨은 선의 정답은 없으며, 기존 agent 검수 개발 사례의 반복 비교는 독립·맹검
성능 검증이 아니다. 사람 승인 양성 주석은 0건이다. 학습·Calibration·Holdout은
열거나 수정하지 않았으며 모델 가중치도 바꾸지 않았다.

```bash
python scripts/qgis_grid_comparison.py \
  --zip /path/to/ai_vectorizer-unreleased-learned600u.zip \
  --baseline-zip /path/to/ai_vectorizer-unreleased-diagonal600.zip \
  --fixtures learning/data/korea-label-gap-visual-cases-v4.json \
  --dataset /path/to/korea/dataset \
  --old-model /path/to/usgs-model --new-model /path/to/korea-model \
  --learned-dependencies /path/to/qgis-learned-deps \
  --grid-in-both --before-avoidance 0 --compare-occluded-labels \
  --output /path/to/comparison/result.json
```

기존 7건 비교는 fixtures를 `korea-contour-visual-cases-v3.json`으로 바꾼다.
