# 자동 숫자 공백 연결 점검 — 2026-10-15

`main`의 `379c3c4` 동작을 기준으로, Ink v2에서 `G`를 누르지 않고 일부 실제 숫자
공백을 연결하는 개발 변경을 비교했다. 모델·학습 데이터·가중치·공식 버전은 변경하지
않았다. QGIS 3.44.8, macOS M4 CPU, 실제 NumPy 중심선 backend와 일회용 profile을
사용했다. 사용자 QGIS profile과 동결된 0.1.6 artifact는 변경하지 않았다.

## 사용자 지적으로 철회한 대각선 600

초기 후보는 대각선 `600`의 왼쪽 `(123,269)`에서 오른쪽 `(153,261)`로 이어졌다.
이 쌍을 같은 등고선이라고 적은 v2 참조는 잘못이었다. 사용자가 출력 이미지를 보고
이웃 등고선으로 넘어갔다고 지적했으며, 원본을 다시 확대한 결과 왼쪽의 이른바
끝점 `(122,256)`은 끝나지 않고 6 왼쪽을 돌아 위쪽의 `(134,237)` 부근까지 이어졌다.
문자 영역의 여유 마스크가 멀쩡한 선을 잘라 인공 끝점을 만들었다.

따라서 이 대각선 쌍의 성공 판정과 가시선 정확도 점수는 철회한다. 숨은 올바른
연결은 이번 작업에서 확정하지 않았다. v2 원본과 실패 출력은 감사 자료로 보존하고,
비교 도구는 그 긍정 사례 ID를 기본 거부한다. 동일 클릭은 v3에서 **다른 등고선인
음성 사례**로만 재생한다. 이 재분류는 출력을 본 뒤의 사용자 피드백임을 명시하며,
출력 전 동결된 정답이나 독립 평가로 취급하지 않는다.

- 원래 v2 SHA-256: `0404df954b2c5b43715173a9c61016234cacef4bcae472d599616b589d8dca17`
- 사용자 스크린샷 SHA-256: `b1b9f5a3cea20799f2a00cfef76385b210b7cb4f976af99140e348635d87295b`
- 현재 [v3 사례](data/korea-label-gap-visual-cases-v3.json) SHA-256:
  `5c4291437afbd8bd10c079e0d914c48c1cf6cfc5728d14708b5246309f22ee0d`
- [철회 기록](data/korea-label-gap-case-audit-v1.json)

## 제품 경로

`label_occlusion.py`가 Byte RGB/gray의 작은 내부 공간 두 개 이상을 가진 인쇄 획
집단을 찾는다. OCR·외부 실행 파일·추가 모델·리사이즈는 없다. 검출은 문자 정체성이나
등고선 소속의 증명이 아니며, 작은 내부 공간이 없는 글자와 붙은 획은 놓칠 수 있다.

`occluded_connection.py`는 독립 검출 영역을 제외한 별도 관측 그래프에서 선 조각을
확인한다. 양쪽 모두 **마스크 적용 전 원본 중심선의 차수 1 끝점**이어야 한다.
인공 끝점과 짧아서 방향을 추정하지 못한 조각도 순서·경쟁·장벽 검사에서는 남긴다.
방향, 실제 인쇄 굵기, 끝점 순서와 상호 유일성, 곡선 간 교차와 옆 실선 접촉을
검사한다. 굵기만으로 같은 등고선이라고 판단하지 않는다.

곡선 하나와 그 양쪽의 관측 경로만 허용한다. 원본·LineEvidence는 변경하지 않는다.
커서의 최근접 원본 선을 먼저 고정하고, 제거된 가까운 선 대신 먼 선을 골라서는
안 된다. 가까운 서로 다른 선이 동률인 경우도 거부한다. 마스크 밖에서 온전히
연결되는 관측 경로는 부분 보조율에서도 우선 보존한다. 옛 픽셀 지름길이 숫자를
피해 이미 옆 선으로 넘어간 경우도 있으므로, 옛 경로와 문자 마스크의 교차만으로
새 후보 적용을 제한하지 않는다.

보조율을 한 번 적용한 **실제 표시 곡선**도 옆 관측선 접촉을 검사한다. 이 곡선은
다시 이동평균하지 않으며, 초록 preview와 저장 좌표를 일치시킨다. 수동·색상·학습
회피 비용, 기존 색상 공백 연결, 0%, 지원되지 않는 영상과 선택 기능 실패는 기존
경로를 유지한다. 모든 검출·그래프 준비는 기존 generation/cache/anchor 수명에
결속된 worker에서 하고, hover에서는 완료된 배열과 기하만 읽는다.

## 실제 QGIS 비교

격자 회피·관측 연결·기존 짧은 공백 기능을 양쪽 모두 켜고, 새 자동 문자 공백만
OFF/ON으로 비교했다. Ink, 기존 USGS guide, 한국 guide의 세 조건에서 같은 문맥과
동일한 화면 클릭을 사용했다. 학습 회피 비용은 보정 기록이 없어 0을 유지했다.
학습 모델은 계속 기본 OFF다. `G`나 수동 공백 제안은 호출하지 않았다.

두 양성 사례는 기존 agent 원본 관찰의 개발 진단이며 사람 승인 수는 0이다.
가려진 곡선은 정답을 알 수 없으므로 unknown 사각형을 빼고 **보이는 부분만**
측정했다. 결과는 다음과 같다.

| 사례 | 방법 | 가시 F1@2px 전 → 후 | 가시 p95 거리 전 → 후 |
| --- | --- | --- | --- |
| 700 U | Ink | 0.544 → 1.000 | 7.032 → 2.000px |
| 700 U | USGS guide | 0.397 → 1.000 | 11.416 → 2.000px |
| 700 U | 한국 guide | 0.540 → 1.000 | 6.962 → 2.000px |
| 600 U | Ink | 1.000 → 0.958 | 0.545 → 2.715px |
| 600 U | USGS guide | 0.367 → 0.958 | 9.045 → 2.715px |
| 600 U | 한국 guide | 0.356 → 0.958 | 9.034 → 2.715px |

두 U자는 세 방법 모두 `automatic-label-gap`으로 일반 클릭 채택·저장·한 번의
Undo를 통과했다. **600 U의 Ink 가시 점수는 낮아졌다.** 이 결과를 숨기거나 모든
조건의 향상으로 요약하지 않는다. 보이는 부분과 겹치는 추론 곡선도 현재의 동결된
unknown 경계 밖이면 오차로 센다. 원래 곡선·독립 정답이 없으므로 숨은 굽이의
우열을 이 점수로 판단할 수 없다. 다른 화면 실행의 baseline 결과는 커서 양자화에
영향받았으며, 여기서는 같은 실행 안의 짝 비교만 적었다.

대각선 600 잘못된 쌍과 600 옆 선 쌍은 세 방법 모두 새 연결 후보가 없고 기존 Ink
저장 좌표가 동일하다. **기존 Ink 자체의 잘못된 경로까지 해결했다는 뜻은 아니다.**
이 두 음성 쌍에는 같은 등고선 정확도 점수를 부여하지 않는다.

기존 [원본 재감사 7건](data/korea-contour-visual-cases-v3.json)은 세 방법 모두
새 기능 OFF/ON의 저장 좌표가 완전히 같았다(21쌍). 그중 깊은 U자 두 건도 유지했다.
Calibration·Holdout을 열거나 이 사례들로 모델을 재학습하지 않았다.

## 검증과 산출물

최종 검증 및 source/package 해시는
[기계 판독 기록](data/automatic-label-gap-check-2026-10-15.json)에 결속한다.
일반 pytest는 948 통과·153 skip·171 subtest 통과, 별도 실제 ONNX 환경은 20 통과였다.
최종 ZIP을 풀어 43개 Python 파일의 source byte 일치를 확인하고, 실제 QGIS import와
161개 시험을 skip 없이 통과했다. 일회용 GUI profile의 9개 smoke는 경고 0·종료 0,
7건/숫자 4건의 비교도 종료 0이었고 profile을 제거했다.
최종 ZIP SHA-256은 `3257ec688cb4ba0ce02e5bca1c5a3579b08d2b9750c888c6ba775a939b84f2aa`다.
GUI 비교 당시 ZIP은 문서 갱신 전이었으며 최종 ZIP과 43개 Python byte가 모두 같다.
실제 QGIS 시험은 RGB 검출부터 worker, 일반 클릭, 저장·Undo까지 실행하며, 취소·
빠른 클릭·레이어 제거·래스터 교체·dispose·0%·정상 경로 우선과 부분 보조율 접촉을
확인한다. 합성 회전 시험을 실제 대각선 600 해결의 근거로 사용하지 않는다.

QGIS 합성 192² 장면에서 8회 warm-up 뒤 50회 측정한 전체 경로 조회 p95는
3.17–3.20ms, 도구 preview는 4.02–4.10ms였다. 영상 처리 없는 기하 검사 비용을
포함하며, 전체 한국 지도나 실제 숫자 사례의 지연 성능으로 일반화하지 않는다.

재현 시 로컬 dataset/model 경로를 지정한다.

```bash
python scripts/qgis_grid_comparison.py \
  --zip /path/to/ai_vectorizer-unreleased-auto-labels.zip \
  --fixtures learning/data/korea-label-gap-visual-cases-v3.json \
  --dataset /path/to/korea/dataset \
  --old-model /path/to/usgs-model --new-model /path/to/korea-model \
  --learned-dependencies /path/to/qgis-learned-deps \
  --grid-in-both --before-avoidance 0 --compare-occluded-labels \
  --trace-diagnostics-dir /path/to/local-cache-evidence \
  --output /path/to/local-comparison/result.json
```

7건 보존 비교는 fixtures를 `korea-contour-visual-cases-v3.json`으로 바꾼다.
원본 source/model/cache/PNG와 provisional 실패 결과는 기존 작업 artifact의
`work/auto-labels/`에 보존했다. 재배포 ZIP은 공식 artifact와 별도의 Unreleased
파일이며 새 Python 모듈과 갱신된 오프라인 안내를 포함한다.
