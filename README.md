# 🏛️ ArchaeoTrace

[English](README.en.md)

고지도 등고선을 사람이 검수하며 벡터화하고, 고도선과 표고점에서 DEM·hillshade까지
만드는 로컬 우선 오픈소스 QGIS 플러그인입니다. 지도 자료를 외부 추론 서버에 보내지
않고도 연구자와 현장 실무자가 전체 흐름을 확인하고 다시 실행할 수 있게 만드는 것이
목표입니다.

![QGIS target 3.22–4.99](https://img.shields.io/badge/QGIS_target-3.22%E2%80%934.99-3c8c3c.svg)
![Source Python 3.8+](https://img.shields.io/badge/source_Python-3.8%2B-3776ab.svg)
![Experimental](https://img.shields.io/badge/status-experimental-f28c28.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## Current source status

- [공식 QGIS plugin 저장소](https://plugins.qgis.org/plugins/ai_vectorizer/version/0.1.5/)에
  experimental `0.1.5`가 2026-08-26 공개됐습니다. 현재 source는 Ink v2와
  Smart Recovery와 선택적 학습 등고선 보조를 포함한 `0.1.6` experimental 후보입니다. 아직 QGIS 저장소,
  GitHub Release 또는 tag로 게시하지 않았습니다.
- 공식 QGIS `0.1.5` ZIP은 1,483,635 bytes, SHA-256
  `24f1def6acd63d483ea6bf7c20b944f56507ead52190667ec4f35562fca6c964`입니다.
  repository의 `dist/ai_vectorizer-0.1.5.zip`은 그보다 이전
  commit `89b9f20`에서 만든 로컬 후보(SHA-256 `d2925198…`)이며 공식
  다운로드와 같은 artifact가 아닙니다.
- 현재 `main`의 Ink v2·Smart Recovery와 후속 수정은 `0.1.6` 후보로 묶었습니다.
  이후 개발 중에는 metadata 숫자나 기존 tag를 다시 바꾸거나, 같은 version으로
  QGIS artifact를 재게시하지 않습니다. Git history의
  `0.1.5–0.1.7`과 미공개 worktree의 `0.1.8` 표시도 추가 QGIS 릴리스가
  아니었으며 이력은 rewrite하지 않습니다.
- metadata 대상은 QGIS `3.22–4.99`, source 계약은 Python `3.8+`입니다. 로컬에서는
  Python 3.8/3.10/3.12와 macOS QGIS 3.44.8을 확인했습니다. 후속
  exact `0.1.6` 후보 commit `f48111a`의 원격 CI에서는 QGIS
  3.22.16/3.44.13/4.2.1 package import·runtime safety와 Linux/Windows 결정적 ZIP이
  [통과했습니다](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/33344520364).
  후보 commit·ZIP·CI는
  [`0.1.6 release-readiness 기록`](docs/RELEASE_READINESS_0.1.6.md)에 결속했습니다.
- 학습 기능을 통합한 `main`의 commit `3bf2b5a2c9e793f54286bfc633351bcb4aa2a095`는
  [CI 12개 작업](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/38437848923)을
  모두 통과했습니다. 이는 구현 검증이며 실제 지도 학습이나 정확도 검증을 뜻하지 않습니다.
- 과거 미공개 개발판 `0.1.7–0.1.8`을 설치했다면 QGIS가 `0.1.6`을 자동 update로
  인식하지 않을 수 있습니다. 기존 plugin을 제거한 뒤 `0.1.6` ZIP을 설치하세요.
  QGIS profile의 검증된 model 파일은 plugin 밖에 보존됩니다.

## Main features

- 🧪 [학습 등고선 파일럿](learning/README.md): USGS 144개 크롭 준비·QGIS 주석·
  Colab 학습/재개·CPU 평가와 기본 OFF 로컬 ONNX 보조. 사람 승인 주석은 0개이며,
  별도 약한 라벨 실험은 USGS와 한국 고지도 크롭으로 세 seed 학습·ONNX 내보내기·
  QGIS 비교까지 마쳤습니다.
- ✏️ 추가 model 없이 직접 입력하는 `Freehand`
- 🖋️ 9·15·31 source-pixel 다중 스케일과 RGB/명도 증거를 한 번 계산해 제한된
  Live-Wire에 전달하는 기본 `Ink Centerline`
- 원본 Byte Ink에서 긴 가로·세로 격자를 자동 검출하고 따라가는 비용을 높입니다.
  등고선이 격자를 가로지르는 이동은 허용합니다.
- 🛟 Ink가 불확실한 구간에만 검증된 EfficientSAM-Ti를 soft corridor로 쓰는
  `Smart Recovery (Experimental)`; 기본 OFF, 명시적 설치, 실패 시 같은 Ink 경로 유지
- 🎚️ `0%` 정확한 cursor부터 `100%` 전체 보조 경로까지 좌표를 실제로 혼합하는
  assist slider
- 🟧 Ink 추적 중 `Alt`+두 번 클릭으로 문자·기호·얼룩을 부드럽게 피하는
  session-only 회피 영역
- 🎨 `C`로 원하는 유색 등고선을 지정하고 `Shift+C`로 해제하는 session-only
  선 색상 보조
- 🪜 일부 숫자 공백의 실제 선 끝을 자동으로 잇는 Ink preview와, 양끝을 직접
  지정하는 `⤴ 라벨 공백 연결` (`G`); 초록 경로를 검수한 뒤 클릭해 채택
- 📐 접힌 `Advanced / Legacy methods`에 보존한 `LSD`, `HED`, `MobileSAM`,
  `SAM (ViT-B)`, `Legacy Canny`
- 👁️ 클릭했을 때 채택될 경로를 보여 주는 초록색 preview와 anchor 기반 교정
- ↩️ QGIS edit buffer, constraint와 한 번의 Undo를 존중하는 새 선·기존 선 연장
- ⛰️ contour elevation과 선택적 `Spot Heights` 저장
- 🏔️ 저장된 입력에서 실험적 선형 TIN DEM과 GDAL hillshade 생성
- 🌏 한국어·English UI

기존 추적·DEM 기능의 경계와 module별 책임은
[`Features and architecture`](docs/FEATURES_AND_ARCHITECTURE.md), 학습 파일럿의
데이터·학습·평가 계약은 [`학습 안내`](learning/README.md)에 있습니다.

## How it works

```text
Raster
  → source-grid 다중 스케일 Ink score + tangent/coherence
  → (선택) 로컬 학습 모델의 등고선 지지·문자/도로 회피 점수
  → bounded Live-Wire Ink champion
  → (선택) 저신뢰 구간만 EfficientSAM soft corridor challenger
  → 안전 조건을 모두 통과한 경우에만 challenger 채택
  → 사용자가 확인하는 green preview와 anchors
  → QGIS edit buffer + Undo
  → elevation contour + optional spot height
  → linear TIN DEM + hillshade
```

검출 결과나 mask는 최종 선이 아닙니다. 클릭으로 채택한 구간만 저장 후보가 되고,
최종 확정은 QGIS의 `Save Layer Edits`에서 이루어집니다.

## Tracing modes and dependencies

| UI option | 역할 | 추가 runtime | 상태 |
| --- | --- | --- | --- |
| `✏️ Freehand` | 사용자가 직접 선을 입력 | 추가 pip·model 없음 | baseline |
| `🖋 Ink Centerline` | 다중 스케일·색상 선 증거 + bounded Live-Wire | 추가 pip 없음; SciPy/scikit-image는 선택 | default |
| `Learned contour guide` | Ink 경로에 등고선 지지·비등고선 회피 점수 결합 | SciPy + ONNX + ONNX Runtime + 명시적으로 가져온 로컬 모델 | experimental, default OFF |
| `🛟 Smart Recovery` | Ink 저신뢰 구간의 EfficientSAM corridor challenger | `onnxruntime` + 명시적으로 설치한 고정-hash model | experimental, default OFF |
| `📐 LSD` | OpenCV 선분 검출 + 공용 Live-Wire | OpenCV; SciPy는 선택 | advanced/legacy |
| `🧠 HED` | Caffe HED edge map + 공용 Live-Wire | OpenCV + 약 56.1 MiB model; SciPy는 선택 | advanced/legacy |
| `🎯 MobileSAM` | point mask + edge/skeleton/A* | OpenCV + PyTorch + backend + 약 38.8 MiB weights | advanced/legacy |
| `🧩 SAM (ViT-B)` | point mask + edge/skeleton/A* | OpenCV + PyTorch + backend + 약 357.7 MiB checkpoint | advanced/legacy |
| `🔧 Legacy Canny` | gradient edge + Live-Wire | OpenCV·SciPy는 선택 | advanced/legacy |

QGIS Python의 NumPy는 plugin 공통 전제입니다. SciPy가 없으면 모든 non-SAM edge
mode(Ink/LSD/HED/Canny)는 방향 인식 Live-Wire 대신 제한 범위의 NumPy nearby-edge
snap으로 돌아갑니다. 탐색 범위를 제한해도
글자·기호·인접선에서 오추적할 수 있으므로 초록색 경로를 확인하고 anchor나
Freehand로 교정하세요. Smart Recovery는 SAM mask를 선으로 저장하거나 Ink와 OR하지
않습니다. 끝점·우회·강한 Ink 보존·평행선 전환 검사를 통과한 경로만 채택합니다.
assist `0%`에서는 model과 evidence 계산을 모두 생략합니다.

## Experimental: 학습 등고선 보조

`Advanced: 기존 추적 모델`에서 `로컬 모델 가져오기`를 눌러 `model.onnx`와
검증·출처 `manifest.json`이 있는 폴더를 선택한 뒤 `학습 등고선 보조`를 켭니다.
기본은 OFF이며 플러그인은 이 모델을 자동 다운로드하지 않습니다.
[`requirements-learned-guide.txt`](requirements-learned-guide.txt)의 선택 패키지는
QGIS Python에 설치하고, PyTorch 학습 패키지는 별도 Python 3.12 환경에 설치합니다.

학습 보조는 SciPy, Ink 모드, `0%`보다 큰 보조 강도와 정렬된 로컬 3-band Byte RGB
GeoTIFF·PNG·JPEG가 필요합니다. 원본 좌표에 고정된 512×512 타일에서 중앙
384×384 결과를 사용해 화면을 이동해도 같은 픽셀의 문맥을 유지합니다. 이 방식은
전체 이미지 추론과 같은 결과를 보장하지 않습니다. 모델 없음·손상·런타임 없음·
잘못된 출력·지원되지 않는 래스터는 Ink로 복귀합니다. 늦게 도착한 결과는 이미
표시된 미리보기, 대기 클릭이나 수동 제안을 바꾸지 않습니다.

USGS 파일럿은 크롭 생성 전에 도엽 가족을 분할하고 원본·권리·크롭 위치·주석
수정 이력을 해시로 기록합니다. 16개 도엽의 144개 크롭을 원본과 픽셀 단위로
대조했으며, 현재 승인된 주석은 **0개**입니다.

| 분할 | 도엽 | 크롭 | 용도 |
| --- | ---: | ---: | --- |
| Train | 6 | 72 | 가중치 학습·증강 |
| Validation | 2 | 24 | 체크포인트·seed 선택 |
| Calibration | 4 | 24 | 경로 비용 6개 조합 비교·동결 |
| Holdout | 4 | 24 | 동결된 모델·설정의 최종 평가 |

`학습 크롭 주석 편집`에서 준비된 PNG와 작업 폴더를 선택합니다. QGIS에 추가된
`contours`, `other_ink`, `other_ink_regions`, `reviewed`, `ignore` 레이어에서 초안을
수정하고 검수 영역의 모든 보이는 등고선을 표시하세요. 판단할 수 없는 영역은
`ignore`로 남깁니다. 레이어 편집을 저장한 뒤 `주석 검수·내보내기`로 새 JSON을
만들고, CLI로 새 데이터 매니페스트에 연결합니다. 평가용 크롭에서는 검수한 열린
선 하나를 선택하고 난이도를 지정합니다. 단독 검수는 실험용으로 기록하며 공식
성능 검증이나 한국 지도 일반화 근거로 사용하지 않습니다.

원본 이미지·가중치는 Git에 포함하지 않습니다. 데이터 준비 명령과 주석 연결,
[`Colab 노트북`](learning/colab_train.ipynb), 학습·내보내기·평가 방법은
[`학습 안내`](learning/README.md)에 있습니다. 모델은 ImageNet ResNet18 인코더와
작은 U-Net 디코더의 두 출력으로 등고선 지지와 문자·도로 회피를 학습합니다.
세 seed 중 Validation 결과로 후보 하나를 선택하고, Drive 체크포인트로 재개합니다.
Calibration에서 설정을 동결한 뒤 Holdout을 평가합니다. **사람 검수 파일럿의 Colab 학습,
Holdout 성능 검증과 한국 지도 검증은 아직 완료하지 않았습니다.** 합성 진단 모델은
학습·재개·ONNX·QGIS 연결 확인용입니다. Unsloth 고도 숫자·범례 판독은 후속 단계입니다.

사람 검수 전에는 별도 데이터·실험 폴더에서 명시적 `--allow-weak-labels`로
기계 생성 라벨을 학습할 수 있습니다. [`실험용 Colab 노트북`](learning/colab_train_weak_labels.ipynb)은
`machine_labeled` 상태와 `human_review_required=true`를 체크포인트·ONNX에 기록합니다.
이 모델의 Validation 결과는 기계 라벨과의 일치도이며 실제 등고선 정확도 근거가
아닙니다. 사람 검수 Calibration/Holdout 평가와 합격 모델 승격은 허용하지 않습니다.
실행 방법은 [`약한 라벨 실험 안내`](learning/README.weak-labels.ko.md)에 있습니다.

2026-10-14에는 Colab `2026.07` 런타임(Python 3.12.13, Tesla T4)에서 동결된
144개 크롭 중 Train 72개·Validation 24개로 세 seed를 모두 학습했습니다. Validation
손실만으로 seed 3408(60 epoch, 약한 라벨 손실 1.0018)을 골라 47.5 MiB FP32 ONNX로
내보냈고, PyTorch와의 확률 차이는 최대 `1.7e-6`이었습니다. M4 Mac CPU에서 준비된
타일 추론 p95는 69 ms, 추가 RAM은 254 MiB였고, QGIS 3.44.8에서 미리보기 좌표 저장·
한 번의 Undo·0% 보조 시 추론 없음을 확인했습니다.

이 모델은 USGS의 갈색 등고선 관례를 배웠습니다. 같은 선을 검정·파랑으로 그리면
등고선이 아니라 회피 대상으로 판단하므로, 검정·파랑 등고선이 많은 한국 지도에는
아래 한국 재학습 모델을 쓰세요. Validation 크롭 5개의 QGIS 비교에서는 Ink가 이웃 등고선으로
건너간 1개 사례를 바로잡았고 나머지는 비슷하거나 판단할 수 없었습니다. 기계 기준선과
5개 사례만 쓴 방향 확인이며, 사람 검수·Calibration·Holdout 평가는 아직 남아 있습니다.
해시와 측정 세부는 [`실행 기록`](learning/COLAB_WEAK_RUN_2026-10-14.md)에 있습니다.

### 한국 고지도 약한 라벨 재학습 (2026-10-15)

두 번째 실험은 USGS 144개 크롭에 검정 인쇄 AMS 한국 1:50,000(L751) 16개 도엽의
149개 크롭을 더했습니다. 한국 라벨도 사람 검수 없이 획의 기하로 자동 생성했습니다.
가늘고 길게 휘어지는 검정 획은 등고선 후보, 파란 덧인쇄·직선 격자·글자·채운 기호는
비등고선 잉크, 애매한 곳은 손실에서 제외합니다. 학습에는 두 증강을 켰습니다.
`ink_recolor`는 등고선 잉크를 검정·회색·파랑·갈색으로 바꿔 칠해 색만으로 등고선을
판단하지 못하게 하고, `label_gaps`는 등고선 위에 고도 숫자를 써서 선을 끊되 정답은
이어진 채로 두어 숫자를 비등고선으로 보고 끊긴 선을 잇도록 가르칩니다.

[`한국 실험 노트북`](learning/colab_train_korea_weak.ipynb)으로 Train 162개·Validation
48개를 Colab T4에서 세 seed 모두 60 epoch 학습했고, Validation 손실만으로 seed
3408(0.860)을 골라 `ai-weak-contour-guide-korea-v1` FP32 ONNX(47.5 MiB)로
내보냈습니다. Calibration·Holdout은 열지 않았습니다. M4 Mac CPU에서 준비된 타일 추론
p95는 56 ms, 추가 RAM은 253 MiB였습니다.

기존 모델이 회피하던 합성 파랑·검정 선을 새 모델은 등고선으로 지지합니다(지지 점수
0.88·0.89). QGIS에서 합성 파란 선 이탈은 37.7 px에서 0.19 px로 줄었습니다. 실제 한국
크롭에서 등고선을 더 잘 따라간다는 근거는 아직 없습니다. 처음 보고한 같은 클릭 비교는
기계 라벨로 만든 기준선이 옆 등고선으로 건너뛰고 일부 클릭이 격자선 위에 있어 철회했습니다.
기준선이 한 등고선을 따르는 유일한 사례에서는 새 모델이 Ink와 같은 선을 그렸고, 기존
모델은 등고선을 가로질렀습니다. 촘촘한 등고선과 격자가 겹친 사례에서는 세 방식 모두
실패했습니다. 이후 개발용 클릭 쌍을 원본으로 재검토했으며 사람 검수 정답, 파란 등고선 도엽,
Calibration·Holdout 평가가 남아 있으며 모델은 계속 기본 OFF 실험 기능입니다. 세부는
[`한국 실행 기록`](learning/COLAB_KOREA_WEAK_RUN_2026-10-15.md)에 있습니다.

## 격자 회피와 등고선 연속성

원본 해상도의 Byte Ink 캐시에서는 긴 검정·저채도 직선이 가로·세로 각각 2개 이상
확인되면 격자 후보로 검출합니다. 작은 스캔 기울기를 허용하며 Live-Wire가 격자를
따라가는 비용을 높입니다. Recovery에도 유한 비용을 전달하므로 격자를 지우거나
교차하는 등고선을 막지 않습니다. 색이 진한 파란 격자, 크게 회전했거나 짧고 끊긴
격자는 놓칠 수 있고, 두꺼운 격자의 가장자리를 따라갈 수도 있습니다. 판정은 읽은
캐시 문맥에 따라 달라질 수 있습니다.

Ink v2의 원본 중심선이 두 클릭 사이에 이어져 있으면 그 연결을 우선 따라갑니다.
백지를 가로지르는 짧은 옆 선 대신 깊은 U자도 실제 관측선을 따라갑니다. 계산은
백그라운드에서 한 번 준비하며, 끊김·너무 긴 우회·회피 힌트·기존 공백 연결은
기존 픽셀 비용 경로로 처리합니다. 실제 분기나 서로 붙은 선의 정체성까지 보장하지
않으므로 초록색 경로를 검수하세요. 숫자 공백은 아래 `Label-gap bridge`로 검수합니다.
기존 Ink 경로로 되돌리는 보호와 관측 주변 지지 제한 후보는 계속 기본 OFF입니다.

경로 비용 선택 기록이 없는 학습 모델은 **등고선 지지만 사용**합니다. 문자·도로로
잘못 판단한 실제 등고선을 벌점화하던 기본 비용을 적용하지 않으며, 상태에도
`등고선 지지만 사용 · 문자 회피 설정 없음`을 표시합니다. 선택된 비용과 평가 해시가
있는 모델은 그 설정을 따릅니다. 수동 회피 영역과 선 색상 지정은 계속 적용됩니다.
문자·도로를 잘 피하던 모델의 이점도 잃을 수 있으므로 이 변경을 모든 모델의
정확도 향상으로 해석하지 않습니다.

원본만 다시 확대 감사한 결과 이전 7건 중 한 시작 클릭이 이웃 등고선 위에 있었습니다.
그 행과 포함 합계를 철회하고 기존 파일은 보존했습니다. 별도로 동결한 올바른 클릭
사례를 넣은 새 7건에서는 기존 성공 5건을 유지하고 두 U자 모두 같은 등고선을
따라갑니다. Ink와 한국 모델의 U자 F1@2px는 각각 1.000·0.991이며 p95는
0.966·1.982px입니다. 사람 승인·독립 성능 평가·재학습은 없었습니다.
[원본 감사·U자 비교 기록](learning/U_CONTOUR_CHECK_2026-10-15.md)과
[과거 회피 비용 기록](learning/UNCALIBRATED_AVOIDANCE_CHECK_2026-10-15.md)을 구분합니다.

## Experimental: 선 색상 지정

Ink 추적을 시작한 뒤 교차점·글자·배경을 피한 유색 등고선 위에 커서를 두고 `C`를
누르세요. `선 색상 지정` 버튼도 마지막 지도 커서 위치를 사용하며, 상태에 선택한
RGB가 표시됩니다. `Shift+C` 또는 `색상 해제`로 제거합니다. 준비된 native Byte RGB
Ink v2 지도, SciPy와 `0%`보다 큰 assist가 필요합니다. 회색·배경이나 여러 색이
겹쳐 모호한 지점은 지정하지 않습니다.

다른 색의 Ink에 유한한 비용을 더하는 선택적 soft guidance입니다. 회피 영역과는
픽셀별 최댓값으로 합쳐 Ink와 Recovery에 똑같이 전달하며, 같은 색의 글자·기호는
여전히 구분하지 못하므로 초록 경로를 검수하세요. pan 뒤 새 cache에도 색상을 다시
적용하지만 추적 종료·취소·reset 때 지웁니다. 색상은 저장·업로드되지 않습니다.
실제 지도 정확도나 작업시간 개선을 검증한 기능은 아닙니다.

## Experimental: label-gap bridge

Ink v2는 이제 일부 숫자 공백을 **`G` 없이** 연결합니다. 원본 Byte 영상에서 작은
글자 내부 공간이 두 개 이상 모인 영역을 찾고, 양쪽의 **원본에서도 실제 끝점인 선**이
서로 유일할 때 실선 방향을 따르는 곡선을 준비합니다. 평소처럼 등고선을 클릭하고
반대편으로 커서를 옮겨 초록 preview를 검수한 다음 클릭해 채택하세요. OCR이나
추가 모델 설치는 필요 없으며 SciPy와 0%보다 큰 보조율이 필요합니다.

문자 마스크의 여유 영역에 걸친 연속선은 보존하고, 잘라 만든 가짜 끝점은 사용하지 않습니다. 마스크 밖에서
온전히 이어지는 기존 경로를 먼저 보존하며, 경로당 추론 공백은 하나입니다. 경쟁하는
짧은 조각도 옆 선의 증거로 남기고, 커서와 가까운 다른 선·곡선 교차·보조율 적용 후
옆 선 접촉을 검사합니다. 수동·색상·학습 회피 비용이 있거나 연결이 모호하면 기존
Ink를 유지합니다. 기존 경로가 두 실제 끝점과 보이는 선을 이미 잘 따르면 그 경로와
평활화를 보존합니다. 내부 공간이 적은 글자, 붙은 문자 획과 실제 분기는 여전히 어렵습니다.

자동 연결이 나오지 않는 실제 공백은 `G`로 양끝을 지정해 검수할 수 있습니다.

```text
공백 직전 마지막 실선에 anchor 확정 → 반대편 첫 실선 끝에 hover → G → 초록 preview 검토 → 같은 끝점 클릭
```

| 상황 | 동작 |
| --- | --- |
| 양쪽에 공백 바깥쪽 Ink 지지가 있고 방향이 일치 | assist 강도를 반영한 제한된 곡선 preview |
| 글자 획이 3px 방향장을 가림 | 두 명시적 endpoint의 바깥쪽 12px 중심선만 다시 확인 |
| 공백 안에서 U자로 크게 꺾임 | 양쪽 실선의 부호 있는 방향으로 Bézier 후보를 만들고 옆 선에 닿으면 굽이를 줄여 다시 검사 |
| 대상선이 없거나 평행선·문자만 보임, 방향이 퍼짐 | **연결하지 않고 기존 Ink preview 유지** |

끝점을 클릭해야 채택하며 `Esc` 또는 다른 위치 클릭은 proposal만 취소합니다. hover는
미리보기를 지우지 않습니다. `G`는 지정한 끝점을 움직이지 않으며, 가까운 다른 끝점을
건너뛰거나 긴 옆 선을 가로지르는 후보는 거부합니다. 숨은 곡선은 추정이므로 원래 선을
복원했다는 뜻은 아닙니다. 실제 숫자 사례와 기존 사례의 비교·남은 실패는
[라벨 공백 비교 기록](learning/LABEL_GAP_CHECK_2026-10-15.md)에 남깁니다.

실제 QGIS에서 `700`·`600` U자와 새로 원본을 감사한 대각선 `600`의 **바깥 선 쌍**을
일반 클릭으로 연결하고 저장·Undo했습니다. 대각선은 보이는 굽이를 끝까지 따른 뒤
실제 공백만 연결하며, 세 방법 모두 가시 F1@2px가 1.000이었습니다. 기존에 잘못
짝지었던 안쪽 선 쌍에는 자동 연결을 제안하지 않으며 기존 Ink fallback은 유지합니다.
`600` U자는 Ink와 두 학습 guide 모두
가시 F1이 1.000입니다. 두 guide의 남았던 0.957은 학습 지지가 짧은 경로를 싸게 만들어
좋은 Ink 경로를 비교에서 빠뜨린 결과였습니다. 검증된 실제 공백에 한해 원본 경로도
검사하고, 같은 끝점·가시선과 표시 충돌 검사를 통과하면 보존합니다.
기존 7건 × 세 방법의 21개 저장 좌표 쌍도 동일합니다. 숨은 선의 정답·일반 정확도
검증은 아니며, 고정된 실제 입력 좌표와 남은 한계는
[학습 보조 U자 후속 비교](learning/LEARNED_U_FOLLOWUP_2026-10-16.md)에 기록합니다.

OpenCV 선택 기능의 선언 범위는 `OpenCV 4.8–4.11`입니다.

```bash
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"
```

backend별 설치, model 크기·SHA-256, 수동 경로와 운영체제별 안내는 ZIP에 포함되는
[`plugin user guide`](ai_vectorizer/README.md)를 확인하세요.

## Quick start

1. QGIS `Plugins > Manage and Install Plugins > Install from ZIP`에서 배포 ZIP을
   설치하고 ArchaeoTrace를 활성화합니다.
2. georeferenced raster와 2D line output layer를 선택합니다.
3. 기본 `Ink Centerline` 또는 `Freehand`와 assist 강도를 고릅니다. 필요할 때만
   Smart Recovery 또는 가져온 학습 보조 모델을 켭니다.
4. 초록색 preview와 `Ink` / `Recovering` / `Enhanced` / `Ink fallback` 상태를
   확인합니다. 기존 방식은 접힌 `Advanced / Legacy methods`에 있습니다.
5. 클릭·드래그로 anchor를 정하고 `Enter` 또는 우클릭으로 edit buffer에 넣습니다.
   시작점 근처에서 닫으면 elevation을 입력할 수 있습니다.
6. `Save Layer Edits`로 contour를 저장한 뒤 `Step 4 > DEM 생성…`에서 pixel size와
   output을 확인합니다.

## Data, network and safety boundary

- raster crop, vector geometry와 local inference 결과를 원격 inference 서버로 보내지
  않으며 기본 telemetry가 없습니다.
- QGIS plugin의 network는 사용자가 HED/SAM model 또는 Recovery model 설치를 직접 실행할 때만
  사용합니다. Recovery는 자동 다운로드하지 않으며, 설치 뒤에는 content-addressed
  cache의 크기·SHA-256과 ONNX session을 background task에서 준비합니다. 준비 중에는
  Ink만 사용하며 Recovery 입력은 native Byte raster로 제한합니다. 손상된 regular
  cache object는 명시적인 `Repair Recovery Model`에서만 격리·재검증하고 unsafe
  object는 자동 변경하지 않습니다. SAM Check/Status는
  유효한 local checkpoint가 있으면 size와 SHA-256을 offline 확인하고, 파일이 없을
  때만 고정 source의 availability를 조회합니다. EfficientSAM benchmark도 명시적인
  `model fetch`만 network를 사용합니다. 지도 crop과 사용자 geometry는 업로드하지
  않습니다.
- 학습 등고선 모델은 로컬 폴더에서만 가져옵니다. 별도 데이터 준비 도구는 명시적인
  명령으로 USGS 원본을 내려받고, Colab 학습은 사용자가 자신의 Drive에 올린 자료와
  명시적으로 초기화한 ImageNet 가중치를 사용합니다. 이는 QGIS 로컬 추론과 분리됩니다.
- `SAM Status Report`에는 현재 작업 경로, QGIS/Python 환경값과 model 경로가 포함될
  수 있고 clipboard에도 복사됩니다. 다른 사람에게 보낼 때 내용을 먼저 확인하고
  local path나 민감한 환경값을 지우세요.
- model은 고정 URL, byte size와 SHA-256을 확인한 뒤 temporary file, `fsync`, atomic
  publication과 rollback을 거칩니다. symlink인 저장 위치와 최종 파일은 거부합니다.
- trace 결과는 QGIS edit buffer에 남아 있으며 사용자가 저장하기 전에는 원본
  dataset에 commit되지 않습니다.

보안 제보와 안전한 진단 공유 방법은 [`SECURITY.md`](SECURITY.md)에 있습니다.

## Terrain limitations

DEM 입력은 저장된 2D contour, 숫자형 finite elevation, 서로 다른 두 고도값, TIN을
만들 수 있는 비공선 vertex와 미터 단위 projected CRS가 필요합니다. 결과는 기본
2,500만 cell까지 제한되고 staged file을 검증한 뒤 DEM·hillshade 쌍으로 게시됩니다.

선형 TIN은 입력 범위 밖과 희소하거나 잘못된 contour에 취약합니다. 현재 버전은
topology QA, uncertainty/NoData layer와 provenance sidecar를 아직 만들지 않습니다.
따라서 결과는 연구 결론을 자동 확정하는 지형 정답이 아니라 검토할 가설입니다.

## Benchmark evidence

`benchmarks/`는 `ink-livewire-v1`, `ink-livewire-v2`,
`efficientsam-ti-onnx-v1`, `ink-v2-effsam-recovery-v1`의 최종 ordered
centerline, 실행 환경, 입력·출력 SHA-256, 시간·RAM과 topology 지표를 기록합니다.
8개 도엽·48개 crop의 권리·주석·분할을 강제하는 공개 데이터셋에서 첫 USGS HTMC
도엽과 무손실 crop 6개를 실제로 staged했습니다. 나머지 7개 도엽과 모든 독립 검수가
아직 채워지지 않았으므로
`publication_ranking_eligible=false`입니다. 현재 자료는 실제 지도 정확도나 다른
도구보다 낫다는 근거가 아닙니다. 명령과 evidence 형식은
[`benchmarks/README.md`](benchmarks/README.md)를 참고하세요.

`learning/`의 16개 USGS 도엽·144개 크롭은 이 기존 8×6 회귀 자료와 별도입니다.
학습·재개·ONNX 출력 비교와 QGIS 동작은 합성 자료로 검증했고, 별도 약한 라벨
실험은 실제 USGS·한국 고지도 크롭으로 Colab GPU 학습·ONNX 내보내기·QGIS 동작을 확인했습니다.
실제 지도의 F1@2px·선간 거리·끊김·과잉 분기 개선이나 도입 기준 통과를 주장하지 않습니다.

## Repository guide

| 문서·경로 | 내용 |
| --- | --- |
| [`ai_vectorizer/`](ai_vectorizer/) | QGIS plugin source와 ZIP 사용자 guide |
| [`docs/FEATURES_AND_ARCHITECTURE.md`](docs/FEATURES_AND_ARCHITECTURE.md) | 실제 기능, 구현 흐름, 안전 경계와 module map |
| [`docs/INK_V2_SMART_RECOVERY.md`](docs/INK_V2_SMART_RECOVERY.md) | Ink v2 증거, Recovery gate·fallback과 공개 benchmark 계약 |
| [`ROADMAP.md`](ROADMAP.md) | 구현됨·다음 단계·의도적인 비목표 |
| [`docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md`](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md) | 오픈소스 원칙과 작업 gate |
| [`docs/RELEASE_READINESS_0.1.6.md`](docs/RELEASE_READINESS_0.1.6.md) | 현재 후보의 실행 검증, artifact identity와 잔여 위험 |
| [`docs/RELEASE_READINESS_0.1.5.md`](docs/RELEASE_READINESS_0.1.5.md) | 공식 0.1.5와 동결 후보의 역사적 기록 |
| [`benchmarks/`](benchmarks/) | 격리 worker, manifest, 지표와 synthetic fixture |
| [`learning/`](learning/README.md) | USGS 파일럿·주석 계약·Colab 학습/재개·ONNX export·Calibration/Holdout |
| [`AGENTS.md`](AGENTS.md) | 단일 `main` 브랜치 작업 규칙 |
| [`tests/`](tests/) | pure core, packaging와 QGIS safety 회귀 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 개발 환경, test tier, 문서·data 기여 규칙 |
| [`SECURITY.md`](SECURITY.md) | 취약점·data-loss 제보와 진단 redaction |

## Version and release discipline

저장소는 `main` 하나를 유지하며 일반 작업과 커밋도 `main`에서 진행합니다.
별도 개발 브랜치나 PR을 자동으로 만들지 않습니다. 작업 규칙은 [`AGENTS.md`](AGENTS.md)에 있습니다.

`ai_vectorizer/metadata.txt`가 배포 버전의 source of truth입니다. 일상 개발은
`Unreleased` changelog에서 진행하고 metadata 숫자는 release 준비가 시작될 때만 한
번 바꿉니다. CI는 metadata, citation, 문서와 artifact 이름이 같은지 검사합니다.
과거 Git commit과 tag는 지우거나 다시 쓰지 않습니다. 자세한 기록은
[`CHANGELOG.md`](CHANGELOG.md)와 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

개발 중인 현재 source는 공개 artifact 이름과 분리된 임시 ZIP으로 검증하세요.
이 명령은 repository-local `0.1.5`·`0.1.6` 후보 ZIP이나 release tree를
건드리지 않습니다.

```bash
current_source_dir="$(mktemp -d)"
current_source_zip="$current_source_dir/ai_vectorizer-unreleased.zip"
python3 scripts/package_release.py --output "$current_source_zip"
python3 scripts/package_release.py --check --output "$current_source_zip"
```

metadata에서 파생된 repository-local 후보 ZIP은 기록된 동결
SHA-256과 현재 source build가 다르면 기본 명령으로 덮어쓸 수 없습니다. 이
보호 해시는 공식 QGIS 다운로드의 identity가 아니니 두 artifact를 혼동하지
마세요. 새 release를 승인할 때만 metadata와 같은 버전을
`--approve-release-overwrite VERSION`에 명시해야 합니다.
후속 `Unreleased` 구현에서 새 tag, GitHub Release 또는 QGIS upload를 만들지
않습니다. `0.1.6`을 실제 게시하는 동작도 별도 승인과 동일 ZIP 검증이 필요합니다.

## Contributing and citation

bug report, 번역, 문서, 재배포 가능한 sample map과 code contribution을 환영합니다.
실제 데이터는 출처·license·개인정보를 확인하고, 성능 표현에는 재현 절차와 실패
사례를 함께 남겨 주세요. 시작 방법은 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

연구·교육·실무에서 사용했다면 저장소의 [`CITATION.cff`](CITATION.cff)를
이용해 인용할 수 있습니다.

## License

GNU General Public License v2.0. [`LICENSE`](LICENSE)
