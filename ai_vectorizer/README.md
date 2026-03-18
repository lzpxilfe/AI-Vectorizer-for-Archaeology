# 🏛️ ArchaeoTrace v0.1.3

AI-assisted contour digitizing plugin for QGIS.  
고지도와 지형도의 등고선 벡터화를 더 빠르고 안정적으로 도와주는 QGIS 플러그인입니다.

![QGIS 3.22+](https://img.shields.io/badge/QGIS-3.22+-3c8c3c.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab.svg)
![Version 0.1.3](https://img.shields.io/badge/version-0.1.3-f28c28.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## ✨ 0.1.3 Highlights

- 🛡️ `OpenCV (cv2)`가 없어도 플러그인 자체는 로드되며, 필요한 기능만 안내 메시지와 함께 차단됩니다.
- ✅ `HED`는 파일 존재 여부가 아니라 실제 로드와 forward 검증이 성공해야 준비 상태로 표시됩니다.
- 📦 `HED` / `SAM` 가중치 다운로드는 임시 파일 검증 후 교체되어 손상 파일이 남지 않습니다.
- 🧮 래스터 dtype 해석과 엣지 미리보기 해상도 계산이 더 안전해졌습니다.
- 🔁 릴리스 폴더와 ZIP은 스크립트로 재생성해 배포 드리프트를 줄였습니다.

## 🎯 What You Can Do

- ✏️ `Freehand` 모드로 순수 수동 디지타이징
- 🧲 엣지를 따라가는 스마트 트레이싱
- 👁️ `Preview AI-Detected Edges`로 현재 모델이 보는 윤곽선 확인
- ⛰️ 등고선 고도값 입력 및 `Spot Heights` 포인트 저장
- 📄 `Check Selected SAM Model` / `SAM Status Report`로 모델 상태 점검
- 🌏 한국어 / English UI 지원

## 🧠 Model Lineup

| Mode | Speed | Quality | Needs | Notes |
| --- | --- | --- | --- | --- |
| `✏️ Freehand` | Fastest | Manual | None | AI 없이 바로 사용 가능 |
| `🔧 Canny` | Fastest | Good baseline | `OpenCV` | 가장 빠른 기본 추적 |
| `📐 LSD` | Fast | Good | `OpenCV` | 선분 기반 감지 |
| `🧠 HED` | Medium | Smooth | `OpenCV` + `~56MB` model | 인앱 다운로드 가능 |
| `🎯 MobileSAM` | Slow | High | `OpenCV` + `PyTorch` + `MobileSAM` + `~39MB` weights | 경량 세그멘테이션 |
| `🧩 SAM (ViT-B)` | Slowest | Highest | `OpenCV` + `PyTorch` + `segment_anything` + `~358MB` checkpoint | 기본 Full SAM 구성 |

> `Freehand`는 추가 패키지 없이도 사용할 수 있습니다.  
> `Canny / LSD / HED / SAM` 계열 AI 기능은 QGIS Python 환경에 `OpenCV`가 필요합니다.

## 📦 Installation

### 1. Install the plugin

1. QGIS에서 `Plugins > Manage and Install Plugins > Install from ZIP`을 엽니다.
2. 배포 ZIP을 선택해 설치합니다.
3. QGIS를 재시작한 뒤 `ArchaeoTrace`를 활성화합니다.

### 2. Install only the dependencies you need

아래 패키지는 시스템 Python이 아니라 QGIS가 사용하는 Python에 설치해야 합니다.

```bash
# Canny / LSD / HED / edge preview / AI tracing
<QGIS_PYTHON> -m pip install opencv-python-headless

# Optional: better thinning quality
<QGIS_PYTHON> -m pip install scikit-image

# Optional: in-app model download / update checks
<QGIS_PYTHON> -m pip install requests

# MobileSAM
<QGIS_PYTHON> -m pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git

# SAM (default: ViT-B)
<QGIS_PYTHON> -m pip install torch torchvision git+https://github.com/facebookresearch/segment-anything.git
```

macOS QGIS.app 예시:

```bash
"/Applications/QGIS.app/Contents/MacOS/python3.12" -m pip install opencv-python-headless
```

### 3. Download model weights in the plugin

1. `Step 3`에서 `HED`, `MobileSAM`, 또는 `SAM`을 선택합니다.
2. 필요하면 `Check Selected SAM Model`로 최신 여부를 확인합니다.
3. `Download` 버튼으로 가중치를 받습니다.
4. 문제가 있으면 `SAM Status Report`로 JSON 진단 리포트를 생성합니다.

<details>
<summary>📥 Manual model paths</summary>

브라우저로 직접 받아야 하는 경우 아래 파일과 경로를 사용하면 됩니다.

- `HED` weights: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- `MobileSAM` weights: `https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt`
- `SAM (ViT-B)` weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

복사 경로:

- `HED`: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/core/models/hed_pretrained_bsds.caffemodel`
- `MobileSAM`: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/mobile_sam.pt`
- `SAM (ViT-B)`: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/sam_vit_b_01ec64.pth`

</details>

## 🗺️ Quick Workflow

1. 래스터 지도를 선택합니다.
2. 출력 라인 레이어를 새로 만들거나 기존 SHP를 고릅니다.
3. 원하는 모델을 선택합니다.
4. 필요하면 `Preview AI-Detected Edges`로 결과를 미리 봅니다.
5. 클릭/드래그로 등고선을 추적합니다.
6. `Enter` 또는 우클릭으로 저장합니다.
7. 시작점 근처를 다시 클릭하면 폐합 후 고도값을 입력할 수 있습니다.

## ⌨️ Shortcuts

- `Ctrl+Z` / `Backspace`: 마지막 체크포인트로 되돌리기
- `Esc` / `Delete`: 현재 트레이싱 취소
- `Enter` / 우클릭: 현재 선 저장

## 🧯 Troubleshooting

- `ModuleNotFoundError: No module named 'cv2'`
  QGIS Python에 `opencv-python-headless`가 설치되지 않은 상태입니다. 시스템 Python이 아니라 QGIS Python에 설치해야 합니다.
- `HED model is invalid or failed to load`
  손상된 모델일 수 있습니다. 플러그인 UI에서 다시 다운로드하세요.
- 모델 최신 확인 / 다운로드가 실패함
  네트워크 또는 `requests` 미설치 문제일 수 있습니다. 필요 시 수동 다운로드 경로를 사용하세요.
- 엣지 미리보기에 아무것도 보이지 않음
  래스터 범위 안으로 확대하고, 다른 모델로도 비교해보세요.
- AI 기능이 당장 안 되는 환경임
  `Freehand` 모드는 계속 사용할 수 있습니다.

## 🧩 Repository Note

- 저장소 루트의 `README.md`에는 GitHub 배포와 릴리스 패키징 안내가 포함되어 있습니다.
- 설치된 플러그인 폴더에서는 일반 사용자 기준으로 별도 패키징 스크립트가 필요하지 않습니다.

## 🇬🇧 English Summary

- ArchaeoTrace is a QGIS plugin for contour digitizing on historical maps.
- `v0.1.3` focuses on safer runtime behavior, validated HED loading, atomic model downloads, and cleaner release packaging.
- `Freehand` works without extra packages.
- `Canny / LSD / HED / SAM` features require `OpenCV` inside the QGIS Python environment.
- `MobileSAM` and `SAM` also require `PyTorch` plus their backend packages and model weights.

## 📚 Citation

```bibtex
@software{ArchaeoTrace2026,
  author = {lzpxilfe},
  title = {ArchaeoTrace: AI-assisted contour digitizing QGIS plugin for historical maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology},
  version = {0.1.3}
}
```

## 📄 License

GNU General Public License v2.0
