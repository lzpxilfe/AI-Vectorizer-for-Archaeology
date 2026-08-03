# 🏛️ ArchaeoTrace

로컬 컴퓨터에서 고지도 등고선을 벡터화하고, 검수한 고도선으로 DEM과 hillshade까지 만드는 QGIS 플러그인입니다.

![QGIS release metadata 3.22+](https://img.shields.io/badge/QGIS_release_metadata-3.22%2B-3c8c3c.svg)
![Source Python 3.10+](https://img.shields.io/badge/source_Python-3.10%2B-3776ab.svg)
![Metadata 0.1.6](https://img.shields.io/badge/metadata-0.1.6-f28c28.svg)
![Development M1.2](https://img.shields.io/badge/development-M1.2-5b5bd6.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 🚧 Current Source Status

- 현재 플러그인 metadata 버전은 `0.1.6`입니다.
- QGIS `3.22+`와 Python `3.10+`를 대상으로 하며 QGIS 3.40.5 / Python 3.12에서 실기동 검증했습니다.
- UI 추적 방식은 `Freehand`, `Canny`, `LSD`, `HED`, `MobileSAM`, `SAM (ViT-B)`입니다.
- EfficientSAM-Ti ONNX는 비교용 benchmark 전용이며 UI 모델이나 제품 기본값이 아닙니다.
- 지도와 추적 처리는 로컬에서 수행되며, 네트워크는 사용자가 다운로드·업데이트 확인·SAM 상태 리포트 같은 모델 관리 작업을 시작할 때만 필요합니다. SAM 상태 리포트도 원격 모델 정보를 조회합니다.

## 🎯 What You Can Do

- ✏️ `Freehand` 모드로 순수 수동 디지타이징
- 🧲 기준점마다 한 번 계산하고 커서 이동 때 즉시 조회하는 방향 인식 Live-Wire
- 🎚️ `0%` 정확한 커서부터 `100%` 완전 보조 경로까지 실제 좌표 비율로 혼합
- 👁️ 클릭 한 번으로 채택될 정확한 경로를 같은 초록색 선으로 실시간 표시
- 👁️ `Canny`/`LSD`/`HED` 엣지 미리보기와 SAM 계열의 인터랙티브 초록색 경로 미리보기
- ⛰️ 등고선 고도값 입력 및 `Spot Heights` 포인트 저장
- 🏔️ 고도 등고선을 선형 TIN `DEM`/`hillshade` GeoTIFF로 변환
- 📄 `Check Selected SAM Model` / `SAM Status Report`로 모델 상태 점검
- 🌏 한국어 / English UI 지원

## 🧠 Tracing Modes

| UI option | 역할 | 필요한 런타임 | 상태 |
| --- | --- | --- | --- |
| `✏️ Freehand` | 사용자가 직접 선을 입력 | 없음 | 항상 사용 가능 |
| `🔧 Canny` | 짧은 끊김을 잇고 진행 방향을 유지하는 실시간 Live-Wire | `NumPy` + `SciPy`, `OpenCV` 선택 | 기본 방식 |
| `📐 LSD` | 선분 검출 결과를 방향 인식 Live-Wire에 결합 | `OpenCV` + `SciPy` | OpenCV 4/5 지원 |
| `🧠 HED` | 학습된 엣지 지도로 추적 보조 | `OpenCV` + 약 `56MB` 모델 | UI에서 다운로드 가능 |
| `🎯 MobileSAM` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `MobileSAM` + 약 `39MB` weights | 선택 설치 |
| `🧩 SAM (ViT-B)` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `segment_anything` + 약 `358MB` checkpoint | 선택 설치 |

> `0%`는 엣지 감지와 모델 실행을 건너뛰고 커서를 그대로 사용합니다. `1~99%`는 같은 완전 보조 경로와 커서 경로를 좌표 비율로 혼합하고, `100%`는 방향 인식 Live-Wire 전체 경로입니다. 탐색 창은 최근 기준점 주변으로 제한되어 인접 등고선이나 글자로 멀리 도망가지 않습니다. 실제 고지도 기준 데이터셋이 마련되기 전까지 모델 간 정확도 순위는 주장하지 않습니다.

> 현재 개발 소스의 EfficientSAM-Ti ONNX 경로는 합성·실데이터 비교를 위한 격리 benchmark 후보입니다. 고지도 정확도 근거가 쌓이기 전에는 제품 기본 모델이나 UI 선택지로 바꾸지 않습니다.
> 이 후보의 모델 파일은 플러그인에 포함되지 않으며, benchmark는 고정 SHA-256 캐시·CPU provider readback·동일 prompt 및 반복 mask/logit 증거를 검증합니다.

## 📦 Installation

### 1. Install the plugin

1. QGIS에서 `Plugins > Manage and Install Plugins > Install from ZIP`을 엽니다.
2. 배포 ZIP을 선택해 설치합니다.
3. QGIS를 재시작한 뒤 `ArchaeoTrace`를 활성화합니다.

### 2. Install only the dependencies you need

아래 패키지는 시스템 Python이 아니라 QGIS가 사용하는 Python에 설치해야 합니다.

```bash
# LSD / HED / edge preview (Canny Live-Wire에는 OpenCV 불필요)
<QGIS_PYTHON> -m pip install opencv-python-headless

# Direction-aware Live-Wire and better thinning quality
<QGIS_PYTHON> -m pip install scikit-image

# Optional: MobileSAM/SAM download and update checks
<QGIS_PYTHON> -m pip install requests

# MobileSAM
<QGIS_PYTHON> -m pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git

# SAM (ViT-B)
<QGIS_PYTHON> -m pip install torch torchvision git+https://github.com/facebookresearch/segment-anything.git
```

macOS QGIS.app 예시:

```bash
"/Applications/QGIS.app/Contents/MacOS/python3.12" -m pip install opencv-python-headless
```

### 3. Download model weights in the plugin

- `HED`: `Step 3`에서 HED를 선택한 뒤 `Download HED`를 사용합니다.
- `MobileSAM` / `SAM`: 모델을 선택하고 `Check Selected SAM Model`로 원격 상태를 확인한 뒤 해당 `Download` 버튼을 사용합니다.
- MobileSAM/SAM 진단이 필요하면 `SAM Status Report`를 생성합니다. 이 작업은 원격 모델 정보도 조회합니다.

<details>
<summary>📥 Manual model paths</summary>

브라우저로 직접 받아야 하는 경우 아래 파일과 경로를 사용하면 됩니다.

- `HED` network definition: `https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt`
- `HED` weights: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- `MobileSAM` weights: `https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt`
- `SAM (ViT-B)` weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

복사 경로:

- `HED` definition: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/core/models/hed_deploy.prototxt`
- `HED` weights: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/core/models/hed_pretrained_bsds.caffemodel`
- `MobileSAM`: `<QGIS_PROFILE>/ai_vectorizer/models/mobile_sam.pt`
- `SAM (ViT-B)`: `<QGIS_PROFILE>/ai_vectorizer/models/sam_vit_b_01ec64.pth`

SAM weights are stored outside the plugin directory so reinstalling or upgrading
the ZIP does not require downloading them again. Existing plugin-local weights
are migrated automatically on first use.

</details>

## 🗺️ Quick Workflow

1. 래스터 지도를 선택합니다.
2. 출력 라인 레이어를 새로 만들거나 기존 SHP를 고릅니다.
3. 원하는 모델을 선택합니다.
4. `Canny`/`LSD`/`HED`는 `Preview AI-Detected Edges`로 확인합니다. MobileSAM/SAM은 트레이싱을 시작한 뒤 초록색 경로 미리보기를 확인합니다.
5. 클릭/드래그로 등고선을 추적합니다.
6. `Enter` 또는 우클릭으로 저장합니다.
7. 시작점 근처를 다시 클릭하면 폐합 후 고도값을 입력할 수 있습니다.
8. 편집을 저장한 뒤 `Step 4 > DEM 생성…`에서 격자 크기와 출력 경로를 확인합니다.

## 🏔️ Terrain Reconstruction

`elevation` 등고선 + 선택적 표고점 → QGIS 선형 TIN 보간 → GeoTIFF DEM → GDAL hillshade

- 비어 있지 않은 geometry, finite 숫자형 고도, 서로 다른 두 개 이상의 고도값, 비공선 vertex 3개 이상, 미터 단위 투영 CRS가 필요합니다.
- 편집을 저장한 후 실행하세요. 임시 `Spot Heights`도 사용할 수 있지만 point geometry와 등고선과 동일한 CRS가 필요하며, 재현을 위해 파일 레이어로 저장하는 것을 권장합니다.
- QGIS Processing에서 `qgis:tininterpolation`, `gdal:translate`, `native:rasterlayerstatistics`, `gdal:hillshade` 알고리즘이 사용 가능해야 합니다.
- 출력은 기본 2,500만 셀까지 허용하고, 기존 파일은 확인 후에만 덮어씁니다.
- TIN의 범위 밖·입력 누락 구간은 추정 한계로 남습니다. 결과는 자동 확정값이 아니라 검토할 수 있는 지형 가설입니다.

## ⌨️ Shortcuts

- `Ctrl+Z` / `Backspace`: 마지막 체크포인트로 되돌리기
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
- AI 기능이 당장 안 되는 환경임
  `Freehand` 모드는 계속 사용할 수 있습니다.
- DEM 실행 전 차단됨
  편집·고도·비공선 vertex·투영 CRS·Processing provider를 확인하세요. 덮어쓸 출력이 QGIS에 로드되어 있다면 먼저 제거해야 합니다.

## 🧩 Repository Note

- 저장소 루트의 `README.md`에는 GitHub 배포와 릴리스 패키징 안내가 포함되어 있습니다.
- 설치된 플러그인 폴더에서는 일반 사용자 기준으로 별도 패키징 스크립트가 필요하지 않습니다.

## 🇬🇧 English Summary

- ArchaeoTrace is a local-first QGIS plugin for tracing elevation contours on historical maps and building reviewable terrain hypotheses.
- The plugin metadata version is `0.1.6`; this update replaces per-target A* with an anchor-rooted, direction-aware Live-Wire tree.
- Cursor movement performs only a predecessor lookup after one asynchronous tree build per accepted anchor; the green line is the exact one-click result.
- Assist is literal from 0% (exact cursor, no model work), through coordinate blending, to 100% (the full Live-Wire route).
- `Freehand` needs no external model or OpenCV, but uses NumPy from the QGIS Python environment.
- Default `Canny` uses NumPy edge evidence plus SciPy Live-Wire without OpenCV; a local NumPy snap remains the fallback if SciPy is absent.
- `MobileSAM` and `SAM` also require `PyTorch` plus their backend packages and model weights.
- EfficientSAM-Ti ONNX is benchmark-only and is not a product default or a UI option.
- The DEM workflow requires saved elevation contours in a projected metre CRS; its output is a reviewable hypothesis, not an archaeological ground truth.

## 📚 Citation

```bibtex
@software{ArchaeoTrace2026,
  author = {lzpxilfe},
  title = {ArchaeoTrace: AI-assisted contour digitizing QGIS plugin for historical maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology},
  version = {0.1.6}
}
```

## 📄 License

GNU General Public License v2.0
