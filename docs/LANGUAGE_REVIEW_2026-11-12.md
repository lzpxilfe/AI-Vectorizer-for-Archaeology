# Language and presentation update

2026-11-12 · Unpublished 0.1.7 candidate

The later [QGIS submission](QGIS_SUBMISSION_0.1.7.md) includes this work and a
scoped-enum compatibility fix. This document retains the language candidate's
original archive and test evidence.

ArchaeoTrace now starts in Korean for a Korean QGIS locale and English for other
locales. A language chosen in the panel remains in effect on subsequent launches.
Switching languages also updates locally authored model status messages and an
open DEM dialog without resetting its paths.

Both user guides are included in the installation ZIP. The English guide covers
tracing, Undo, saving, DEM inputs, optional models and troubleshooting. The UI and
README use fewer decorative symbols and simpler wording. Headers and primary
buttons follow the QGIS theme. Model names, experimental labels and default-off
settings remain visible.

Routing, elevation handling, raster reads, model weights and training data were
not changed in this update. The preceding code review and its exact artifact
remain documented in the [English review](RELEASE_REVIEW_2026-11-12.en.md) and
[Korean review](RELEASE_REVIEW_2026-11-12.md).

## Validation and artifact

The language candidate is validated separately from the preceding v3 ZIP.

- Archive: `ai_vectorizer-0.1.7-en-review-2026-11-12.zip`
- SHA-256: `39c042074d4c2eb21f5dc39f12a3c1d37115028563c0a7d58e5f0e45e8f2b546`
- Pure Python: 1,246 passed, 208 skipped, 171 subtests passed.
- QGIS 3.44.8: 226 tests passed, no failures, errors or skips; the isolated
  plugin loader also passed all seven scenarios.
- Both guides are present in the ZIP, and the package matches the source.
- Remote CI: all 12 jobs passed for commit
  `2b986430b36bfb8808a43567976df190814aec55`, including QGIS 3.22.16, 3.44.13
  and 4.2.1, security/dependency checks and identical Linux/Windows packages.
  [CI run](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43113650204)
- Size: 2,097,953 bytes, 53 files. The routing and raster/data-read modules are
  byte-identical to the preceding reviewed build. No new map accuracy or speed
  result is claimed for these language changes.

The [validation record](evidence/language-review-2026-11-12-validation.json)
binds the archive, source, guides, test logs and CI. Previous ZIPs remain unchanged.

## 한국어

첫 실행 시 QGIS가 한국어이면 한국어, 그 외에는 영어로 시작합니다. 패널에서 직접
고른 언어는 유지됩니다. 언어를 바꿨을 때 모델 상태에 한국어가 남는 문제도 수정했습니다.
열려 있는 DEM 창의 경로는 그대로 두고 안내 문구를 바꿉니다.

영어 사용 설명서를 설치 ZIP에 넣고 영어 검토 기록을 추가했습니다. UI와 README의
장식용 이모지와 과장된 표현을 줄였으며 제목·시작 버튼은 QGIS 테마를 따릅니다.
추적 알고리즘, 고도 처리, 모델과 학습 데이터는 이번 작업에서 바꾸지 않았습니다.

영어 설명서를 포함한 새 ZIP으로 실제 QGIS 226개 검사와 원격 CI 12개 작업을
통과했습니다. 이전 후보 ZIP은 그대로 보존합니다.
