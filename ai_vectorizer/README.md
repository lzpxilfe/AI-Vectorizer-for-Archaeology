# 🏛️ ArchaeoTrace

로컬 컴퓨터에서 고지도 등고선을 사람이 검수하며 벡터화하고, 고도선으로 DEM과
hillshade까지 만드는 QGIS 플러그인입니다. 지도와 추적 결과를 원격 추론 서버로
보내지 않습니다.

![QGIS release metadata 3.22+](https://img.shields.io/badge/QGIS_release_metadata-3.22%2B-3c8c3c.svg)
![Source Python 3.8+](https://img.shields.io/badge/source_Python-3.8%2B-3776ab.svg)
![Experimental](https://img.shields.io/badge/status-experimental-f28c28.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 🚧 Current Source Status

- 공식 QGIS 저장소에 experimental `0.1.5`가 2026-08-26 공개됐습니다.
  현재 checkout과 설치 후보는 metadata `0.1.6`, `experimental=True`이며 아래
  Ink v2·Smart Recovery를 포함합니다. `0.1.6`은 아직 QGIS 저장소나 GitHub
  Release에 게시되지 않은 검증 후보입니다.
- 공식 `0.1.5` artifact와 Ink v1 fallback은 보존합니다. 현재 `0.1.6` UI와 사용법은
  이 문서가 source of truth이며, 정확한 후보 ZIP·commit·CI 범위는 repository의
  `docs/RELEASE_READINESS_0.1.6.md`에 기록합니다.
- QGIS `3.22–4.99`와 Python `3.8+` source 호환성을 대상으로 합니다. 로컬 검증은
  Python 3.8/3.10/3.12와 macOS QGIS 3.44.8에서 수행했습니다. 원격 CI의 QGIS
  3.22.16/3.44.13/4.2.1 package import·runtime safety와 Linux/Windows 결정적 ZIP도
  통과했습니다.
- 기본 UI는 `Freehand`, 다중 스케일 `Ink Centerline`, 기본 OFF의
  `Smart Recovery (Experimental)`입니다. LSD, HED, MobileSAM, SAM (ViT-B),
  Legacy Canny와 기존 model index 0–5는 접힌 Advanced 영역에 보존됩니다.
- Smart Recovery의 EfficientSAM-Ti ONNX는 Ink가 약한 구간의 soft corridor로만
  사용하며 model 미설치·오류·취소 시 Ink 경로를 그대로 유지합니다.
- 지도와 추적 처리는 로컬에서 수행됩니다. model 다운로드는 network를 사용합니다.
  SAM Check/Status는 유효한 local checkpoint가 있으면 size와 SHA-256을 offline
  확인하고, 파일이 없을 때만 고정 source의 availability를 조회합니다.
- 과거 미공개 개발판 `0.1.7–0.1.8`을 설치했다면 QGIS가 `0.1.6`을 자동 update로
  보지 않을 수 있습니다. 기존 plugin을 제거한 뒤 `0.1.6` ZIP을 설치하세요.
  profile에 저장된 검증 model은 plugin 폴더 밖에 유지됩니다.

## 🎯 What You Can Do

- ✏️ `Freehand` 모드로 순수 수동 디지타이징
- 🖋️ 9·15·31 source-pixel과 RGB/명도에서 만든 연속 선 증거를 기준점마다 한 번
  계산하는 방향 인식 Live-Wire
- 🛟 검증된 local EfficientSAM으로 저신뢰 Ink 구간만 보완하고 안전하게 나아진
  challenger만 채택하는 Smart Recovery
- 🎚️ `0%` 정확한 커서부터 `100%` 완전 보조 경로까지 실제 좌표 비율로 혼합
- 👁️ 클릭 한 번으로 채택될 정확한 경로를 같은 초록색 선으로 실시간 표시
- 👁️ `Ink Centerline`/`LSD`/`HED`/`Legacy Canny` 검출 미리보기와 SAM 계열의 인터랙티브 초록색 경로 미리보기
- ⛰️ 등고선 고도값 입력 및 `Spot Heights` 포인트 저장
- 🏔️ 고도 등고선을 선형 TIN `DEM`/`hillshade` GeoTIFF로 변환
- 📄 `Verify Selected SAM Model` / `SAM Status Report`로 모델 상태 점검
- 🌏 한국어 / English UI 지원

## 🧠 Tracing Modes

| UI option | 역할 | 필요한 런타임 | 상태 |
| --- | --- | --- | --- |
| `✏️ Freehand` | 사용자가 직접 선을 입력 | QGIS `NumPy`; 추가 pip·model 없음 | baseline |
| `🖋 Ink Centerline` | 다중 스케일·색상 연속 증거와 방향 인식 Live-Wire | QGIS `NumPy`; `SciPy`/`scikit-image`는 선택 최적화 | 기본 방식 |
| `🛟 Smart Recovery` | 저신뢰 Ink 구간의 EfficientSAM corridor challenger | `onnxruntime>=1.17,<2` + 약 39.4 MiB split ONNX | 실험적·기본 OFF |
| `📐 LSD` | 선분 검출 결과를 Live-Wire에 결합 | `OpenCV`; `SciPy`는 Live-Wire 선택 최적화 | Advanced / Legacy |
| `🧠 HED` | 학습된 edge map으로 추적 보조 | `OpenCV 4.8–4.11` + 약 `56.1 MiB` model; `SciPy` 선택 | Advanced / Legacy |
| `🎯 MobileSAM` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `MobileSAM` + 약 `38.8 MiB` weights | Advanced / Legacy |
| `🧩 SAM (ViT-B)` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `segment_anything` + 약 `357.7 MiB` checkpoint | Advanced / Legacy |
| `🔧 Legacy Canny` | 기존 그래디언트 경계 검출과 Live-Wire | `NumPy`; `OpenCV`/`SciPy` 선택 | Advanced / Legacy |

> `0%`는 edge 감지와 model 실행을 건너뛰고 cursor를 그대로 사용합니다. `1~99%`는
> 같은 완전 보조 경로와 cursor 경로를 좌표 비율로 혼합하고, `100%`는 전체 보조
> 경로입니다. 탐색 창을 최근 anchor 주변으로 제한해도 인접 contour, 글자나 기호로
> 오추적할 수 있습니다. 초록색 경로를 확인하고 anchor나 Freehand로 교정하세요.
> 실제 역사 지도 기준 dataset 전에는 model 간 정확도 순위를 주장하지 않습니다.

> Smart Recovery는 SAM mask를 최종 선으로 쓰거나 Ink와 이진 OR하지 않습니다. 시작·끝점,
> 기존 탐색창·우회 한계, 강한 Ink 보존, 평행선·분기 전환과 개선량을 검사합니다.
> 실제 재배포 가능 고지도 holdout 결과 전에는 기본 승격이나 정확도 우위를 주장하지
> 않습니다. 모델 파일은 플러그인 ZIP에 포함되지 않습니다.

## 📦 Installation

### 1. Install the plugin

1. QGIS에서 `Plugins > Manage and Install Plugins > Install from ZIP`을 엽니다.
2. 배포 ZIP을 선택해 설치합니다.
3. QGIS를 재시작한 뒤 `ArchaeoTrace`를 활성화합니다.

### 2. Install only the dependencies you need

`Freehand`와 기본 `Ink Centerline`은 QGIS ZIP 설치 직후 추가 pip 없이 동작합니다.
`SciPy`가 없으면 모든 non-SAM edge mode(Ink/LSD/HED/Canny)는 방향 인식 Live-Wire
대신 제한 창의 NumPy nearby-edge snap을 사용합니다. 아래 선택 패키지는
시스템 Python이 아니라 QGIS가 사용하는 Python에 설치해야 합니다.
OpenCV는 검증한 4.8–4.11 범위로 제한하지만 이 제한만으로 QGIS의 공유
NumPy ABI가 고정되지는 않습니다. 설치 전에 pip의 변경 계획을 확인하세요.

QGIS 3.22/Python 3.8은 기본 ZIP 경로의 source·무의존성 계약 대상입니다. `0.1.6`
후보는 QGIS 3.22.16/3.44.13/4.2.1에서 exact package import·runtime-safety CI를
통과해야 하며 결과는 repository의 release-readiness 기록에 결속합니다. 이 검증은
이미 게시된 공식 0.1.5 ZIP을 소급해 증명하지 않습니다. Python 3.8은 EOL이며 최신 보안
수정이 적용된 Pillow/pytest
의존성 계열을 설치할 수 없습니다. 이 릴리스의 선택적 SciPy/scikit-image,
OpenCV, SAM pip 스택과 `requirements-dev.txt`는 보안 유지 대상 Python
3.10+ 환경에서 사용하세요. 가능하면 최신 지원 QGIS/Python을 사용하고,
공유 QGIS 환경을 변경하기 전에 프로필과 환경을 백업하세요.

```bash
# LSD / HED / SAM용 OpenCV (Ink Centerline과 Legacy Canny에는 불필요)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"

# Optional: non-SAM edge mode의 방향 인식 Live-Wire와 더 빠른 morphology/thinning
<QGIS_PYTHON> -m pip install scipy scikit-image

# Optional: Smart Recovery local CPU runtime (plugin이 자동 실행하지 않음)
<QGIS_PYTHON> -m pip install "onnxruntime>=1.17,<2"

# Optional: MobileSAM/SAM download and update checks
<QGIS_PYTHON> -m pip install requests

# MobileSAM (immutable upstream commit)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"

# SAM (ViT-B, immutable upstream commit)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
```

저장소 checkout의 격리된 Python 3.10+ 개발 환경에서는 기본
`requirements.txt`, Smart Recovery용 `requirements-smart-recovery.txt`, OpenCV용
`requirements-opencv.txt`, 백엔드별
`requirements-sam-mobile.txt` 또는 `requirements-sam-full.txt`를 사용할 수
있습니다. `requirements-dev.txt`는 Python 3.10+ 테스트 전용입니다.

macOS QGIS.app 예시:

```bash
"/Applications/QGIS.app/Contents/MacOS/python3.12" -m pip install "opencv-python-headless>=4.8,<4.12"
```

### 3. Download model weights in the plugin

- `Smart Recovery`: 먼저 위 ONNX Runtime을 QGIS Python에 설치합니다. QGIS를 다시
  시작한 뒤 `Smart Recovery (Experimental)`을 켜고 `Install Recovery Model`을
  명시적으로 누릅니다. encoder 24,799,761 bytes와 decoder 16,565,728 bytes를 고정
  URL에서 받아 SHA-256을 검증한 후에만 활성화됩니다. 자동 다운로드나 지도 업로드는
  없습니다. 파일 검증과 ONNX session 준비는 background task에서 진행되어 그동안도
  Ink tracing을 사용할 수 있습니다. Recovery는 native Byte raster에서만 실행하고
  그 밖의 정수형에서는 Ink v2를 유지합니다. 상태는 `Ink`, `Recovering`, `Enhanced`,
  `Ink fallback`으로 표시됩니다. 검증된 regular model 파일이 손상됐으면 버튼이
  `Repair Recovery Model`로 바뀌며, 명시적으로 눌렀을 때만 기존 파일을 격리하고
  다시 받습니다. 실패·취소 시 미완료 파일은 격리본을 복원하고, 이미 hash 검증된
  replacement는 유지하며 symlink·directory·Windows junction/reparse point는 변경하지
  않습니다.
- `HED`: `Step 3`에서 HED를 선택한 뒤 `Download HED`를 사용합니다.
- `MobileSAM` / `SAM`: model을 선택하고 `Verify Selected SAM Model`로 local 상태를
  확인한 뒤 필요한 경우 해당 `Download` 버튼을 사용합니다. local checkpoint가
  없을 때만 고정 source availability 조회가 일어납니다.
- MobileSAM/SAM 진단이 필요하면 `SAM Status Report`를 생성합니다. report에는 현재
  작업 경로, QGIS/Python 환경값과 model 경로가 포함될 수 있고 clipboard에도
  복사됩니다. 공유 전에 내용을 검토하고 민감한 local path·환경값을 지우세요.

<details>
<summary>📥 Manual model paths</summary>

브라우저로 직접 받아야 하는 경우 아래 파일과 경로를 사용하면 됩니다.

- `HED` network definition: `https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/deploy.prototxt`
- `HED` weights: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- `MobileSAM` weights: `https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt`
- `SAM (ViT-B)` weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

HED 수동 다운로드도 정확히 검증해야 합니다: definition은 8,186 bytes / SHA-256
`378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a`, weights는
58,876,104 bytes / SHA-256 `4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7`입니다.

복사 경로:

- `HED` definition: `<QGIS_PROFILE>/ai_vectorizer/models/hed_deploy.prototxt`
- `HED` weights: `<QGIS_PROFILE>/ai_vectorizer/models/hed_pretrained_bsds.caffemodel`
- `MobileSAM`: `<QGIS_PROFILE>/ai_vectorizer/models/mobile_sam.pt`
- `SAM (ViT-B)`: `<QGIS_PROFILE>/ai_vectorizer/models/sam_vit_b_01ec64.pth`

HED and SAM assets are stored outside the plugin directory so reinstalling or
upgrading the ZIP does not require downloading them again. Exact verified
plugin-local HED assets from an older install are migrated automatically on
first use.

</details>

## 🗺️ Quick Workflow

1. 래스터 지도를 선택합니다.
2. 출력 라인 레이어를 새로 만들거나 기존 SHP를 고릅니다.
3. 원하는 모델을 선택합니다.
4. 기본 Ink의 초록색 경로와 Recovery 상태를 확인합니다. 기존 LSD/HED/SAM/Canny가
   필요하면 `Advanced: legacy tracing models`를 펼칩니다.
5. 클릭/드래그로 등고선을 추적합니다.
6. `Enter` 또는 우클릭으로 QGIS 편집 버퍼에 추가합니다. 한 번의 Undo로 추가/연장을 되돌릴 수 있습니다.
7. 시작점 근처를 다시 클릭하면 폐합 후 고도값을 입력할 수 있습니다.
8. QGIS의 `Save Layer Edits`로 변경을 확정한 뒤 `Step 4 > DEM 생성…`에서 격자 크기와 출력 경로를 확인합니다.

## 🏔️ Terrain Reconstruction

`elevation` 등고선 + 선택적 표고점 → QGIS 선형 TIN 보간 → GeoTIFF DEM → GDAL hillshade

- 비어 있지 않은 geometry, finite 숫자형 고도, 서로 다른 두 개 이상의 고도값, 비공선 vertex 3개 이상, 미터 단위 투영 CRS가 필요합니다.
- 편집을 저장한 후 실행하세요. 임시 `Spot Heights`도 사용할 수 있지만 point geometry와 등고선과 동일한 CRS가 필요하며, 재현을 위해 파일 레이어로 저장하는 것을 권장합니다.
- QGIS Processing에서 `qgis:tininterpolation`, `gdal:translate`, `native:rasterlayerstatistics`, `gdal:hillshade` 알고리즘이 사용 가능해야 합니다.
- 출력은 기본 2,500만 셀까지 허용하고, 기존 파일은 확인 후에만 덮어씁니다.
- TIN의 범위 밖·입력 누락 구간은 추정 한계로 남습니다. 결과는 자동 확정값이 아니라 검토할 수 있는 지형 가설입니다.

## ⌨️ Shortcuts

- 트레이싱 중 `Ctrl+Z` / `Backspace`: 마지막 체크포인트로 되돌리기
- 저장 후 `Ctrl+Z`: QGIS 편집 스택에서 방금 추가·연장한 작업 되돌리기
- `Esc` / `Delete`: 현재 트레이싱 취소
- `Enter` / 우클릭: 현재 선 저장

## 🧯 Troubleshooting

- `ModuleNotFoundError: No module named 'cv2'`
  QGIS Python에 `opencv-python-headless`가 설치되지 않은 상태입니다. 시스템 Python이 아니라 QGIS Python에 설치해야 합니다.
- `HED model is invalid or failed to load`
  손상된 모델일 수 있습니다. 플러그인 UI에서 다시 다운로드하세요.
- MobileSAM/SAM 최신 확인 또는 다운로드가 실패함
  네트워크 또는 `requests` 미설치 문제일 수 있습니다. 필요 시 수동 다운로드 경로를 사용하세요. HED 다운로드는 `requests`와 무관합니다.
- 엣지 미리보기에 아무것도 보이지 않음
  래스터 범위 안으로 확대하고, 다른 모델로도 비교해보세요.
- 선택 model 기능이 안 되는 환경임
  유효한 raster와 2D line output이 있으면 추가 pip·model 없이 `Freehand`를 사용할
  수 있습니다.
- DEM 실행 전 차단됨
  편집·고도·비공선 vertex·투영 CRS·Processing provider를 확인하세요. 덮어쓸 출력이 QGIS에 로드되어 있다면 먼저 제거해야 합니다.

## 🧩 Repository Note

- 저장소의 [GitHub README](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology)는
  구현 구조, 개발 계획과 release 검증 문서로 연결합니다.
- 설치된 플러그인 폴더에서는 일반 사용자 기준으로 별도 패키징 스크립트가 필요하지 않습니다.

## 🇬🇧 English Summary

- ArchaeoTrace is a local-first QGIS plugin for tracing elevation contours on historical maps and building reviewable terrain hypotheses.
- The QGIS repository published experimental `0.1.5` on 2026-08-26. This guide describes the unpublished experimental `0.1.6` candidate with Ink v2 and Smart Recovery.
- Cursor movement performs only a predecessor lookup after one asynchronous tree build per accepted anchor; the green line is the exact one-click result.
- Assist is literal from 0% (exact cursor, no model work), through coordinate blending, to 100% (the full Live-Wire route).
- `Freehand` needs no additional pip package, external model, or OpenCV, but uses NumPy from the QGIS Python environment.
- Default `Ink Centerline` works after ZIP installation using QGIS' NumPy. SciPy enables the bounded direction-aware Live-Wire for every non-SAM edge mode; without it those modes use the local NumPy nearby-edge snap.
- Trace additions and extensions stay in QGIS' edit buffer; one Undo reverts each operation, and `Save Layer Edits` commits it.
- `MobileSAM` and `SAM` also require `PyTorch` plus their backend packages and model weights.
- Smart Recovery is an experimental, default-OFF EfficientSAM-Ti corridor prior.
  It never replaces a failed run with another backend and always preserves the Ink champion.
- The DEM workflow requires saved elevation contours in a projected metre CRS; its output is a reviewable hypothesis, not an archaeological ground truth.

## 📚 Citation

Use the repository's
[CITATION.cff](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CITATION.cff)
as the citation source of truth.

## 📄 License

GNU General Public License v2.0
