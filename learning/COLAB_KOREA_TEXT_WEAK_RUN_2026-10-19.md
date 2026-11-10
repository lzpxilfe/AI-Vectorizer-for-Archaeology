# Colab 한국 지도 일반 글자 공백 약한 정답 학습·QGIS 비교 — 2026-10-19

`word_gaps`를 추가한 세 seed 학습, FP32 ONNX 내보내기와 검증, 기존 Korea
모델과의 동일 조건 QGIS 17개 진단 사례 비교를 완료했다. **양성 13개에서 새
모델의 경로·점수 개선은 관찰되지 않았다. 기존 Korea 모델을 교체하지 않으며
학습 보조는 실험용·기본 OFF로 유지한다.** 양성 13개와 일반 글자 음성 2개는
저장 좌표까지 같았고, 기존 숫자 공백 음성 2개는 일반 Ink 경로 좌표가 달랐다.
따라서 17개 모두 동일하다는 결론은 아니다.

기계 판독용 결과는
[`data/korea-general-text-model-comparison-2026-10-19.json`](data/korea-general-text-model-comparison-2026-10-19.json)에
기록한다. 원본 raster, 체크포인트, 모델, 상세 실행 JSON·NPZ·오버레이는 git 밖에
보존한다. 이 기록은 배포 승인이나 사람이 검수한 정확도 평가가 아니다.

## 학습과 선택

| 항목 | 확인 결과 |
| --- | --- |
| 실행 환경 | Colab Tesla T4, Python 3.12.13, torch 2.14.1, torchvision 0.29.1, segmentation-models-pytorch 0.5.0 |
| 동결 데이터 | USGS 16도엽·144크롭 + AMS Korea L751 16도엽·149크롭 |
| 학습에 사용 | train 162크롭(USGS 72 + Korea 90), validation 48크롭(24 + 24) |
| 열지 않은 분할 | 공식 Calibration 36크롭(24 + 12), Holdout 47크롭(24 + 23) |
| 증강 | `augment=true`, `ink_recolor=true`, `label_gaps=true`, `word_gaps=true` |
| 공통 설정 | learning rate 0.0001, 최대 60 epoch, patience 8, batch 2 × accumulation 8, topology weight 0.2, AMP 사용 |
| 정답·검수 수준 | `automated_weak_labels`, `allow_weak_labels=true`, `human_review_required=true`, 사람 승인 0크롭 |
| 코드 | `59ad23d041137e1886b594dd983d9de862f98a01` |
| 학습 소스 SHA-256 | `9071c099d5c3defb70ac47b9d92530db9786234fff7151eb4f51edd379087ea0` |
| 데이터 manifest SHA-256 | `b2fd2a087f75de4e5e1d31b44f13dfdb1a8af3963e18e68f81f9cb4a91e32249` |

새 일반 글자 진단 크롭은 학습 데이터에 추가하지 않았다. Korean validation은
기존 kr-09·kr-10 자료를 포함하므로 아래 진단은 선택과 개발 과정의 영향을 받은
자료다. 일반화 정확도나 독립 Holdout 성능으로 해석하지 않는다.

| Seed | 종료 이유 | 완료 epoch | 최종 optimizer steps | 최저 weak-label validation loss | 최저 loss epoch |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3407 | `max_epochs` | 60 | 653 | 0.8653252273797989 | 60 |
| **3408** | `max_epochs` | **60** | **655** | **0.8430383851130804** | **59** |
| 3409 | `early_stopping` | 49 | 535 | 1.0264879042903583 | 41 |

3409는 epoch 41 이후 개선 없는 8개 epoch로 patience를 소진했다. 세 후보 모두
`selected-model.json`에 들어 있으며, validation loss 최소값과 seed 오름차순
동률 규칙만으로 3408을 선택했다. 선택 당시
`calibration_accessed=false`, `holdout_accessed=false`를 확인했다.
선택된 **best 체크포인트는 epoch 59·645 steps**다. 이는 학습 종료 기록인
60 epoch·655 steps와 다른 시점이다. 모델 manifest의 `epochs_completed=59`는
이 선택 체크포인트를 가리킨다.

| 선택·산출물 commitment | SHA-256 |
| --- | --- |
| 3408 best 체크포인트 | `00f57fa4fb11a89701691cdefa1caec9eccec99d8a0703e540efb064f39db90e` |
| 3408 학습 설정 | `3f70809518e7ee8ecc855d02c6dabae05bd4e174d1360a3cfa9d73e615d7a75e` |
| 선택 JSON 파일 | `cf3b3ca0118f15cc49d468a6561f04318f2f66196ee1e0f2dc8029939117e447` |
| 선택 JSON 정규화 내용 | `ce9e9fc9a519262621f9c7640fb63b719a972c6fc54d3edf2b2a87d5d0c9cd0d` |

## ONNX 번들 검증

새 번들 ID는 `ai-weak-contour-guide-korea-text-v1`이다. 기존
`ai-weak-contour-guide-korea-v1`과 `ai-weak-pilot` 결과는 보존했다.

| 항목 | 결과 |
| --- | --- |
| 모델 파일 | `model.onnx`, 49,848,570 bytes |
| 새 모델 SHA-256 | `a1f27dfd8421c44defaaed521b5421c18e1358abcad9499ffc987dc33990e4d1` |
| 비교한 기존 Korea 모델 SHA-256 | `ae5fbf01e8fb073d686f81ec4f2f23367bd66386eddbfde8b314777e02e70398` |
| 새 manifest 파일 SHA-256 | `68017088f93b74e18271b4f40c6f2f0d6abe004bdaab0ae9a316559c01ba2023` |
| 새 번들 SHA-256 | `ab979237b831a13bf2b8d5413599d7793b80e5d82c15fac6c32af5041555d164` |
| export 소스 SHA-256 | `c8c79a79c4653236a2ec4956895ec125e10cef7d162c695c63e245216cf38d8c` |
| 계약 | FP32 `rgb [1,3,512,512]` → `logits [1,2,512,512]`, RGB/255, ImageNet 정규화는 그래프 내부 |
| 파일·그래프 검사 | `inspect_bundle.ready=true`, 정확한 크기·SHA, 표준 ONNX 연산, 외부 tensor 없음, FP32 가중치 |
| 보조 비용 | support 0.8, learned avoidance 0.0 |
| 범위 | `GPL-2.0-only`, `research_ready=false`, `publication_ranking_eligible=false`, 사람 검토 필요 |

내보내기와 독립 재검증 모두 `learning/export.py`의 고정 random·paper 입력을
사용하고 sigmoid 이후 확률 절대오차를 비교했다. ONNX 1.23.2·ONNX Runtime
1.31.0, CPU, intra-op 최대 4 threads·inter-op 1의 동일 설정이다.

| 검증 장소 | Random 최대오차 | Paper 최대오차 | 허용값 |
| --- | ---: | ---: | ---: |
| Colab CPU export manifest | 3.7522656578525826e-6 | 2.1913821305896697e-6 | 1e-4 |
| 로컬 CPU 독립 재검증 | 1.039898846777465e-5 | 2.1902125442285758e-5 | 1e-4 |

두 환경 모두 통과했다. Colab의 전체 최대값은 3.752e-6이며, 2.191e-6은 paper
입력 값이다. 로컬 재검증은 선택 best.pt의 실제 바이트·설정·소스 commitment와
세 후보 선택 문서를 함께 확인했다.

## 기존 Korea 모델과 동일 조건 QGIS 비교

현재 제품 ZIP `ai_vectorizer-unreleased-general-text-final.zip`의 SHA-256은
`1966cecdee2733c6d57cfaff702f5f13a275da7e6504b67c0e546c7f9064d359`다.
두 모델을 같은 ZIP·QGIS 3.44.8-Solothurn·원본 native RGB·정수 화면 클릭·CPU
조건에서 비교했다. ORT 1.31.0, `CPUExecutionProvider`, intra/inter-op 각각 1,
`ORT_SEQUENTIAL`, support 0.8·avoidance 0.0이다. after 단계는 격자 회피,
관측 연결, 숫자·일반 글자 자동 공백 검출을 켜고 observation gate와 route guard는
끄는 동일 정책을 사용했다.

도구의 역사적 이름 때문에 **`usgs_after`가 기존 Korea 모델**, **`korea_after`가
새 일반 글자 모델**이다. 이 실행에서 `usgs`는 USGS 전용 모델을 뜻하지 않는다.
아래는 두 모델의 **after 결과끼리** 비교한 표다. before→after 차이는 자동 공백
처리 등 제품 동작의 차이를 포함하므로 재학습 효과로 합산하지 않는다.

F1은 2px 허용 기준이다. 일반 곡선은 검토한 가시 선을, 공백 양성은 가시 양쪽
팔만 채점했다. 음성 쌍은 정확도 기준선이 없어 F1·p95를 N/A로 표시했다.

| 사례 ID | 기존 F1 | 새 F1 | 기존 p95(px) | 새 p95(px) | 저장 좌표 |
| --- | ---: | ---: | ---: | ---: | --- |
| `kr10-text05-blue-a-gap` | 1.000 | 1.000 | 0.314 | 0.314 | 동일 |
| `kr10-text05-blue-a-wrong-right-neighbor` | N/A | N/A | N/A | N/A | 동일 |
| `yongdam-text02-black-myeon-gap` | 1.000 | 1.000 | 1.000 | 1.000 | 동일 |
| `yongdam-text02-black-myeon-wrong-right-neighbor` | N/A | N/A | N/A | N/A | 동일 |
| `kr-10-crop-04-clear-w` | 1.000 | 1.000 | 0.433 | 0.433 | 동일 |
| `kr-09-crop-08-curving` | 1.000 | 1.000 | 0.496 | 0.496 | 동일 |
| `kr-09-crop-05-grid-crossing` | 1.000 | 1.000 | 1.146 | 1.146 | 동일 |
| `kr-10-crop-02-deep-u` | 1.000 | 1.000 | 0.969 | 0.969 | 동일 |
| `kr-10-crop-06-dense-s` | 1.000 | 1.000 | 1.147 | 1.147 | 동일 |
| `kr-09-crop-02-index-u` | 0.991 | 0.991 | 1.982 | 1.982 | 동일 |
| `kr-10-crop-01-clear-w-corrected-raw-v1` | 1.000 | 1.000 | 0.858 | 0.858 | 동일 |
| `kr-10-crop-01-label-700-u` | 1.000 | 1.000 | 2.000 | 2.000 | 동일 |
| `kr-10-crop-01-label-600-u` | 1.000 | 1.000 | 0.558 | 0.558 | 동일 |
| `kr-10-crop-12-label-600-diagonal-outer-raw-v1` | 1.000 | 1.000 | 0.471 | 0.471 | 동일 |
| `kr-10-crop-12-label-600-diagonal-wrong-neighbor` | N/A | N/A | N/A | N/A | 변경, 최대거리 1.442–1.447px |
| `kr-10-crop-01-label-600-wrong-neighbor` | N/A | N/A | N/A | N/A | 변경, 최대거리 0.566–0.571px |
| `kr10-second06-index-300` | 1.000 | 1.000 | 0.696 | 0.696 | 동일 |

양성 13개의 저장 정점과 점수가 모두 같아 관찰된 개선·퇴행은 각각 0개다.
일반 글자 음성 2개도 저장 좌표가 같았다. 기존 숫자 음성 2개는 Ink fallback
좌표가 변했지만 음성에 정확도 정답이 없으므로 이를 개선·퇴행 점수로 만들지
않는다. 양성·일반 글자 사례에 새 모델만의 추가 우회는 확인되지 않았다.
음성 경로 차이는 원래 선을 0.01px 이하 간격으로 샘플링하고 상대 연속 선분까지
최단거리를 구한 대칭 최대거리다. 샘플링으로 구한 하한에 0.005px를 더한 상한을
함께 표시했다. 두 음성 모두 시작·끝점과 전체 bounding box는 같았다.

음성 4개 모두 두 모델의 숫자 연결 후보와 일반 글자 연결 후보가 `null`이고
`display_accepted=false`이며, 추론 공백 연결을 저장하지 않았다. 그러나 사용자가
지정한 잘못된 이웃 클릭까지 **일반 Ink 경로는 저장된다**. 이 결과는 추론 연결
후보 제외 검사이며, 서로 다른 등고선의 의미를 알아 모든 경로를 거부한다는
보장이 아니다. 面 양성은 두 모델 모두 같은 `automatic-text-gap` 경로다.
해당 사례의 before F1 0.975→after 1.0은 제품의 자동 글자 공백 처리 효과이며
새 모델의 우위가 아니다. 가려진 곡선의 정확한 모양은 정답으로 채점하지 않았다.

별도 가시 이웃 선 기준선이 있는 5개 사례의 연속 선분 간 최소거리는 다음과 같다.
두 모델의 값은 같고 모두 1px 이내 접촉이 없었다. 기준선이 없는 나머지 사례나
모든 이웃 등고선의 비접촉을 보장하는 수치로 확대하지 않는다.

| 사례 | 최소거리(px), 기존=새 모델 |
| --- | ---: |
| 파란 글자 양성 | 4.243 |
| 파란 글자 음성 | 4.353 |
| 검은 面 양성 | 5.821 |
| 검은 面 음성 | 5.000 |
| 새 300 숫자 양성 | 3.578 |

클릭의 map·screen 좌표, NPZ의 원본 RGB·edges·centerline·center_score·tangent
x/y·coherence·root·origin·target을 비교해 동일 입력을 확인했다. 저장 좌표와
전체 오버레이도 검토했다. 이 NPZ는 학습 확률 출력을 직렬화하지 않으므로,
관측 배열이나 최종 경로가 같다는 사실이 두 모델의 예측 확률까지 같다는 뜻은
아니다. 각 비교의 상세 좌표·정점 수·진단은 외부 result와 audit에 남겼다.
별도 CPU 측정의 동일 RGB 입력에서는 11개 고유 크롭 모두 기존·새 모델 출력
해시가 달랐다. 모델을 잘못 재사용해서 모든 양성 경로가 같아진 것으로 해석할
근거는 없다.

## CPU 지연과 메모리

Apple M4에서 실제 진단 17개가 공유하는 입력을 파일 SHA로 중복 제거한
11개 native 512×512 RGB 크롭을 측정했다. 이미지/model마다 새 subprocess를
순차 실행하고 짝수 크롭은 기존→새, 홀수는 새→기존 순서를 사용했다.
첫 추론은 별도 측정 후 1회 추가 warm-up을 거쳐 25회씩 측정했다. 모델당
275회이며 pooled p95는 정렬한 275개 중 262번째 값이다.

| 실제 크롭 standalone 측정 | 기존 Korea | 새 일반 글자 |
| --- | ---: | ---: |
| prepared p50 | 54.417ms | 54.820ms |
| prepared pooled p95 | 57.682792ms | 58.197959ms |
| 크롭별 p95의 최대값 | 59.354542ms | 60.137500ms |
| peak RSS 증가량 최대 | 536,625,152 bytes / 511.765625MiB | 536,150,016 bytes / 511.312500MiB |
| 절대 peak RSS 최대 | 577,896,448 bytes / 551.125000MiB | 577,585,152 bytes / 550.828125MiB |

prepared 시간은 uint8→FP32 변환, 추론, sigmoid와 prior 생성을 포함한다.
이미지 디코딩과 전체 QGIS 클릭→미리보기·저장은 포함하지 않는다. 관측된 pooled
p95 차이는 새 모델 +0.893%로 속도 개선 근거가 없으며, 통계적 유의성이나 사용자
작업 속도 차이를 주장하지 않는다. RSS 증가량은 각 프로세스 high-water에서
시작 기준을 뺀 값으로 절대 RSS와 구분한다.

별도의 QGIS clean-profile lifecycle은 실제 지도 대신 합성 단색 입력을 쓴다.
두 번들 모두 native learned preview의 저장 좌표 유지, 한 번의 Undo, 0% 보조,
빠른 클릭과 cache 준비 중 클릭 등 9개 시나리오를 통과했고 임시 profile을 제거했다.

| 합성 QGIS lifecycle 측정 | 기존 Korea | 새 일반 글자 |
| --- | ---: | ---: |
| prepared tile p95, 25회 | 56.596207ms | 56.462500ms |
| peak RSS 증가량 | 458,981,376 bytes / 437.718750MiB | 461,160,448 bytes / 439.796875MiB |

이 합성 입력의 p95나 RAM을 실제 지도 정확도·전체 QGIS 작업 성능 또는 paired
Holdout RAM으로 바꾸어 설명하지 않는다.

## 근거와 재현

외부 실행 폴더의 다음 파일을 보존한다. 공개 저장소에는 local 사용자 이름,
Drive 문서 URL, 원본 지도나 모델 가중치를 넣지 않는다.

- `seed-3407-result-verified.json`, `seed-3408-result-verified.json`,
  `seed-3409-result-verified.json`: 종료 이유·이력·step·설정·실제 체크포인트 검증.
- `selection-verified.json`, `training-checkpoints/selected-model-2026-10-19.json`:
  세 후보의 실제 바이트 commitment와 validation 전용 선택.
- `new-onnx-verification.json`: 다운로드 해시, 번들 검사, 독립 FP32 parity와 런타임.
- `general-text-qgis-audit.json`, `regression-qgis-audit.json`: 좌표·캐시·연결 후보·접촉 감사.
- `model-text-vs-old-korea-{blue,black,old7,old5,old300}/`: 각 `result.json`,
  `scores.json`, 전체 비교 PNG와 `cache/*.npz`.
- `real-crop-cpu-comparison.json`, `benchmark_real_crop_cpu.py`: 실제 크롭 지연·RSS 원자료와 실행 코드.
- `model-old-korea-qgis-lifecycle.json`, `model-text-qgis-lifecycle.json`: 실제 QGIS lifecycle 기록.
- `outputs/learned-model-korea-general-text/ai-weak-contour-guide-korea-text-v1/`:
  새 `model.onnx`·`manifest.json` 번들. 기존 Korea 번들도 별도로 유지한다.

학습 설정과 러너는
[`colab_train_korea_text_weak.ipynb`](colab_train_korea_text_weak.ipynb),
절차와 데이터 경계는 [`README.general-text.ko.md`](README.general-text.ko.md)를
따른다. QGIS 비교는 외부 산출물 경로를 환경 변수에 지정한 뒤 아래 형태로
재현한다. `$RUN_ARTIFACTS`는 위 외부 실행 폴더, `$PRODUCT_ZIP`는 기록된 동일
SHA의 ZIP, `$OLD_KOREA_BUNDLE`·`$NEW_TEXT_BUNDLE`은 검증한 번들 폴더,
`$LEARNED_DEPS`는 격리된 ONNX 의존성 폴더다. QGIS Python에 pip 설치하지 않는다.

```bash
python scripts/qgis_grid_comparison.py \
  --zip "$PRODUCT_ZIP" \
  --fixtures learning/data/korea-general-text-gap-cases-v1.json \
  --dataset "$RUN_ARTIFACTS/dataset" \
  --old-model "$OLD_KOREA_BUNDLE" --new-model "$NEW_TEXT_BUNDLE" \
  --learned-dependencies "$LEARNED_DEPS" \
  --grid-in-both --before-avoidance 0 --after-avoidance 0 \
  --compare-occluded-labels \
  --trace-diagnostics-dir "$RUN_ARTIFACTS/recheck-blue/cache" \
  --output "$RUN_ARTIFACTS/recheck-blue/result.json"
```

다른 네 cohort는 기계 판독용 결과의 fixture·dataset 조합으로 같은 명령을
실행하고 `usgs_after`와 `korea_after`를 비교한다. 이미 검증한 출력에 덮어쓰지
않는다. parity는 별도 학습 Python 환경에서 `learning.export._parity_inputs()`와
`PARITY_TOLERANCE=1e-4`, 선택 best.pt로 재검증한다.

기존 실행 기록의 철회된 약한 정답 기준선은 복권하지 않는다. 현재 결과는
약한 정답 학습과 agent가 검토한 개발·validation 진단에 한정된다. 사람 검수,
독립 Calibration, 검토된 Holdout 비교와 일반 글자 유형별 평가가 남아 있으며,
실제 경로 개선 근거가 생기기 전 새 모델을 기본 모델로 교체하지 않는다.

## 저장소 검증

이 절은 **후속 진단 로그 구현 전의 실험 결과 기록 단계**에 수행한 검사다.
이후 진단 코드 변경과 178개 QGIS runtime 검사는
별도의 후속 실패 진단 기록에 구분해 둔다.

결과 문서와 한국어·영어 README·Unreleased 기록을 갱신했다. 제품 Python 소스,
버전과 기본 모델은 바꾸지 않았다. `python -m pytest -q`는 1182 passed,
159 skipped, 171 subtests passed였다. 일반 Python의 QGIS skip과 별도로 위 실제
QGIS 17개 진단 및 두 모델의 9개 lifecycle 시나리오씩을 실행했다.
`package_release.py`의 별도 임시 ZIP 생성과 `--check`, `git diff --check`도
통과했다. 문서 변경 후 생성한 제품 ZIP은 비교에 사용한 ZIP과 SHA가 같았다.
