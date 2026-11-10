# Colab 실험 성과 전달과 커밋 구성 — 2026-10-24

## 사람들에게 전달할 핵심

> ArchaeoTrace는 Colab GPU 학습에서 로컬 ONNX 추론과 QGIS 저장 경로 비교까지
> 실험 절차를 완주했습니다. 일반 글자 공백용 재학습 모델은 17개 개발 진단 중
> 양성 13개의 경로를 개선하지 못해 기존 Korea 모델을 유지했습니다. 후속 평가셋과
> 선택적 로그로 O 공백의 단어 그룹 검출 누락을 확인했으며, 다음은 그 검출 조건을
> 좁게 보완하고 기존 경로·음성 사례를 보존하는지 확인하는 작업입니다.

README 첫 화면은 이 결론과 증거 링크를 전달한다. 세 seed/loss/해시/측정 조건은
상세 실행 기록으로 보낸다. 실제 지도 그림에는 **학습 모델을 쓰지 않은 Ink 커널
시범**임을 함께 적는다. QGIS 파일 API 렌더를 GUI 조작 화면처럼 소개하지 않는다.

공개할 수 있는 주장은 다음과 같다.

- 3 seed 학습·재개·선택과 FP32 ONNX 독립 검증 완료.
- 같은 제품 코드·입력·클릭·CPU에서 모델 채택 여부를 비교하고 기존 모델 유지 결정.
- 선택적 경로 로그와 개발용 6쌍을 추가해 수정할 병목을 구체적으로 확인.
- 공개 실제 지도에서 가시 등고선 3구간을 커널/API로 저장하고 파일 무결성 확인.

사람 승인 주석은 0개이고 공식 Calibration/Holdout은 평가하지 않았다.
'일반 글자 공백 정확도 개선', '17건 모두 동일', '모든 잘못된 연결 거부',
'전체 지도 자동 완성', '전체 QGIS 속도 향상'은 현재 결과가 뒷받침하지 않는다.
원본·모델·상세 실행 자료는 저장소 밖에 있어 공개 저장소만으로 모든 수치의
완전 재실행이 가능하다고도 설명하지 않는다.

## main에서 세 커밋으로 나누기

이 문서는 10-24 작성한 커밋 구성이다. 11-11 원격 main의 README 축약과 배포 기록
수정을 먼저 통합한 뒤 이 구성으로 반영한다. 아래 명령은 파일 분리 기준이다.
학습 구현·노트북은 이미 `59ad23d`까지의 main 이력에 있다. 아래는 그 이후 남은
결과 보고·진단·전달 방식의 변경이다. 버전·tag·Release를 새로 만들지 않는다.

저장소 루트에서 먼저 확인한다.

```bash
git branch --show-current
git status --short
git diff --check
```

브랜치는 `main`이어야 한다. 첫 커밋은 실험의 완료·한계·모델 채택 판단을 기록한다.

```bash
git add learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md \
  learning/data/korea-general-text-model-comparison-2026-10-19.json
git diff --cached --stat
git commit -m "docs: record Colab text-gap retraining and QGIS comparison"
```

두 번째는 실패를 관측하는 기능·평가 fixture·검증을 함께 묶는다. 검출·경로 선택
규칙과 모델 기본값은 유지하며 진단은 기본 OFF다. OFF에서도 작은 helper 비용은 있다.
비교 스크립트의 선택적 manifest SHA 검증과 불확실 사례 채점 구분도 이 묶음에 속한다.

```bash
git add ai_vectorizer/core/livewire.py \
  ai_vectorizer/core/occluded_connection.py \
  ai_vectorizer/core/text_occlusion.py \
  ai_vectorizer/tools/smart_trace_tool.py \
  scripts/qgis_clean_profile_smoke.py scripts/qgis_grid_comparison.py \
  scripts/summarize_gap_diagnostics.py \
  tests/test_grid_comparison.py tests/test_learned_qgis.py \
  tests/test_occluded_connection.py tests/test_text_occlusion.py \
  tests/test_gap_diagnostics_summary.py tests/test_livewire_diagnostics.py \
  learning/GENERAL_TEXT_FAILURE_DIAGNOSIS_2026-10-19.md \
  learning/data/korea-general-text-failure-cases-v1.json \
  learning/data/korea-general-text-failure-cases-v2.json \
  learning/data/korea-general-text-failure-extra-cases-v1.json \
  learning/data/korea-general-text-failure-raw-assessment-v1.json \
  learning/data/korea-general-text-failure-diagnosis-2026-10-19.json
git diff --cached --stat
git commit -m "feat: add opt-in gap diagnostics and frozen failure cases"
```

세 번째는 한국어·영어 소개와 출처를 갖춘 작은 지도 시범을 공개한다.
`.gitignore`는 해당 PNG 하나만 명시적으로 허용한다.

```bash
git add .gitignore README.md README.en.md CHANGELOG.md learning/README.md \
  docs/EXPERIMENT_PUBLISHING.md docs/REAL_MAP_PILOT_2026-10-19.md \
  docs/images/mupungjang-ink-pilot-detail.png
git diff --cached --stat
git commit -m "docs: clarify verified outcomes and show a scoped real-map pilot"
```

## 검사와 푸시

2026-10-24 재확인: 일반 Python **1215 passed, 160 skipped, 171 subtests passed**.
별도 임시 경로의 package 생성·`--check` 통과, 제품 ZIP SHA는
`8fe6814901b241dde814cfa0f84ee7b7613a577d00e646d1f00e9bb725afa4dc`로
10-19 진단 검사와 같다. `git diff --check`도 통과했다.
**QGIS runtime 178개와 기존 102개 경로 동일성은 10-19 보존 기록**이며 오늘
GUI에서 재실행한 결과는 아니다. 그 소스·원자료 SHA를 다시 대조했다.
그 이후 코드·패키지 변경이 생기면 CONTRIBUTING의 해당 test tier를 다시 실행한다.

모델 가중치·원본 스캔·Drive/Colab 개인 문서 ID·인증 정보·충돌 IPS는 stage하지 않는다.
이번 공개 대상은 정제된 요약 JSON, 출처 기록, 코드/fixture/테스트와 작은 그림이다.

```bash
git status --short
git fetch origin
git log --oneline main..origin/main
git log --oneline origin/main..main
```

첫 log가 비어 있고 두 번째에 의도한 세 커밋만 보이는지 확인한 뒤 푸시한다.
원격 main에 추가 커밋이 보이면 먼저 내용을 확인해 통합하며 force push하지 않는다.

```bash
git push origin main
```

푸시 후 GitHub Actions 결과를 해당 commit SHA로 확인한다. 이 작업은 Unreleased
소스·실험 결과의 공개이며, QGIS 공식 플러그인 업데이트나 모델 승격은 별도다.
