# 🏛️ ArchaeoTrace

로컬 컴퓨터에서 고지도 등고선을 벡터화하고, 검수한 고도선으로 DEM과 hillshade까지 만드는 QGIS 플러그인입니다.

![QGIS release metadata 3.22+](https://img.shields.io/badge/QGIS_release_metadata-3.22%2B-3c8c3c.svg)
![Source Python 3.10+](https://img.shields.io/badge/source_Python-3.10%2B-3776ab.svg)
![Metadata 0.1.5](https://img.shields.io/badge/metadata-0.1.5-f28c28.svg)
![Development M1.2](https://img.shields.io/badge/development-M1.2-5b5bd6.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 🚧 Current Source Status

- 현재 플러그인 metadata 버전은 `0.1.4`입니다. 현재 개발 소스에는 그 이후의 실험 기능이 포함되어 있으며 별도 GitHub release artifact로 배포된 상태는 아닙니다.
- `0.1.4` 릴리스 metadata는 QGIS `3.22+`를 선언하지만, 현재 post-`0.1.4` 소스의 신규 모듈은 Python `3.10+`가 필요합니다. 자동화된 코어/benchmark 검증은 Python `3.12`에서 수행했습니다.
- QGIS UI는 `Freehand`, `Canny`, `LSD`, `HED`, `MobileSAM`, `SAM (ViT-B)` 추적과 고도 입력을 제공합니다.
- 저장한 고도 등고선과 선택적 표고점으로 선형 TIN DEM 및 GDAL hillshade를 생성할 수 있습니다.
- 개발용 M1.2 benchmark는 공식 EfficientSAM-Ti split ONNX를 고정 해시·CPU-only 조건에서 비교하지만, EfficientSAM은 아직 UI 모델이나 제품 기본값이 아닙니다.
- 지도와 추적 처리는 로컬에서 수행됩니다. 네트워크는 사용자가 모델 다운로드·업데이트 확인·SAM 상태 리포트 같은 모델 관리 작업을 직접 실행할 때만 필요합니다. SAM 상태 리포트도 원격 모델 정보를 조회합니다.

## 🎯 What You Can Do

- ✏️ `Freehand` 모드로 순수 수동 디지타이징
- 🧲 엣지를 따라가는 스마트 트레이싱
- 👁️ `Canny`/`LSD`/`HED` 엣지 미리보기와 SAM 계열의 인터랙티브 초록색 경로 미리보기
- ⛰️ 등고선 고도값 입력 및 `Spot Heights` 포인트 저장
- 🏔️ 고도 등고선을 선형 TIN `DEM`/`hillshade` GeoTIFF로 변환
- 📄 `Check Selected SAM Model` / `SAM Status Report`로 모델 상태 점검
- 🌏 한국어 / English UI 지원

## 🧠 Tracing Modes

| UI option | 역할 | 필요한 런타임 | 상태 |
| --- | --- | --- | --- |
| `✏️ Freehand` | 사용자가 직접 선을 입력 | 없음 | 항상 사용 가능 |
| `🔧 Canny` | Canny 엣지 비용지도를 따라 A* 추적 | `OpenCV` | 기본 edge 방식 |
| `📐 LSD` | 선분 검출 결과를 비용지도에 결합 | `OpenCV` | OpenCV 4/5 지원 |
| `🧠 HED` | 학습된 엣지 지도로 추적 보조 | `OpenCV` + 약 `56MB` 모델 | UI에서 다운로드 가능 |
| `🎯 MobileSAM` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `MobileSAM` + 약 `39MB` weights | 선택 설치 |
| `🧩 SAM (ViT-B)` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `segment_anything` + 약 `358MB` checkpoint | 선택 설치 |

> `Freehand`는 별도 모델이나 OpenCV 없이 동작하지만 QGIS Python에 포함된 NumPy를 사용합니다. 나머지 추적 방식은 선택 의존성이 필요합니다. 실제 고지도 기준 데이터셋이 마련되기 전까지 모델 간 정확도 순위는 주장하지 않습니다.

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

현재 개발 버전은 다음 수직 파이프라인을 제공합니다.

`elevation` 등고선 + 선택적 표고점 → QGIS 선형 TIN 보간 → GeoTIFF DEM → GDAL hillshade

실행 조건:

- 등고선은 숫자형 고도 필드와 서로 다른 두 개 이상의 고도값을 가져야 합니다.
- 입력 geometry는 비어 있지 않고 vertex를 포함해야 하며, 고도는 finite 숫자여야 합니다. TIN을 만들 수 있는 비공선 vertex가 합계 3개 이상 필요합니다.
- 입력 레이어는 미터 단위의 투영 좌표계를 사용해야 합니다. 경위도 레이어는 먼저 지역에 맞는 CRS로 재투영하세요.
- 편집 중인 내용은 먼저 저장해야 합니다.
- 표고점은 선택 입력입니다. point geometry와 등고선과 동일한 CRS가 필요합니다. 임시 `Spot Heights`도 현재 세션에서 쓸 수 있지만, 재현을 위해 파일 레이어로 저장하는 것을 권장합니다.
- QGIS Processing에서 `qgis:tininterpolation`, `gdal:translate`, `native:rasterlayerstatistics`, `gdal:hillshade` 알고리즘이 사용 가능해야 합니다.
- 안전을 위해 출력은 기본 2,500만 셀까지 허용하고, 기존 파일은 확인 후에만 덮어씁니다.

선형 TIN은 피처의 범위 밖을 신뢰성 있게 복원하지 못하며, 등고선 간격·누락·오표기가 결과에 그대로 반영됩니다. 생성물은 고고학적 사실의 자동 확정값이 아니라 검토할 수 있는 지형 가설입니다.

중기 개발 계획과 합격 기준은 [`ROADMAP.md`](ROADMAP.md)에 기록합니다.

## 🧪 Reproducible Contour Benchmark (developer)

M1.2의 `benchmarks/` 하네스는 모델별 최종 ordered centerline을 같은 조건에서 평가합니다. 입력·기준선·예측 파일의 SHA-256, CPU 실행 여부, 실제 backend/fallback, 반복 시간과 peak RSS를 기록하고 JSON 및 CSV 보고서를 만듭니다. 각 실행은 불변 `runs/` 디렉터리에 저장되고 `benchmark_latest.json` 포인터 하나만 원자적으로 교체되므로 중단된 실행과 이전 CSV가 섞이지 않습니다.

현재 포함된 자료는 합성 계약 fixture뿐입니다. 실제 고지도 데이터셋과 제품 MobileSAM/SAM adapter 비교가 완료되기 전에는 이 하네스로 기본 모델이나 정확도 순위를 결정할 수 없습니다.

```bash
python3 -m benchmarks validate benchmarks/data/synthetic-smoke/manifest.json
python3 -m benchmarks evaluate benchmarks/data/synthetic-smoke/manifest.json \
  --output work/benchmark-results
```

`SmartTraceTool`의 실제 A*와 평활화는 이제 QGIS 독립 공용 커널이며, Canny/LSD worker도 같은 `EdgeDetector → 비용지도 → 커널 → 최종 ordered centerline` 경로를 사용합니다. 실제 격리 smoke 실행은 다음처럼 만듭니다.

```bash
python3 -m venv work/benchmark-runtime
work/benchmark-runtime/bin/python -m pip install \
  'opencv-python-headless>=4.8.0' 'scikit-image>=0.21.0'
work/benchmark-runtime/bin/python -m benchmarks generate \
  benchmarks/data/runtime-template/manifest.json \
  --output work/runtime-smoke \
  --python-executable work/benchmark-runtime/bin/python
python3 -m benchmarks evaluate work/runtime-smoke/manifest.json \
  --output work/runtime-smoke-report --require-eligible
```

EfficientSAM-Ti는 아직 제품 기본 모델이 아니라 M1 비교 후보입니다. 공식 split ONNX 두 파일은 패키지에 넣지 않으며, 아래의 명시적 `fetch`에서만 네트워크를 사용합니다. `status`, `verify`, `generate`는 오프라인으로 content-addressed cache의 고정 크기와 SHA-256을 다시 확인합니다.

```bash
work/benchmark-runtime/bin/python -m pip install 'onnxruntime>=1.17.0'

work/benchmark-runtime/bin/python -m benchmarks model fetch \
  --model-cache work/model-cache
work/benchmark-runtime/bin/python -m benchmarks model verify \
  --model-cache work/model-cache

work/benchmark-runtime/bin/python -m benchmarks generate \
  benchmarks/data/efficientsam-runtime-template/manifest.json \
  --output work/efficientsam-runtime-smoke \
  --python-executable work/benchmark-runtime/bin/python \
  --model-cache work/model-cache
python3 -m benchmarks evaluate \
  work/efficientsam-runtime-smoke/manifest.json \
  --output work/efficientsam-runtime-report --require-eligible
```

worker는 sample×method마다 새 프로세스로 실행되고 실제 provider, 모델·패키지·thread 상태, 실행 소스 파일, 입력/설정/출력의 SHA-256, 반복 시간과 peak RSS를 직접 기록합니다. EfficientSAM 경로는 encoder/decoder의 ORT 설정과 OpenCV 상태를 실제 세션에서 다시 읽고, 동일 prompt의 의미 해시·float32 tensor 해시와 반복별 IoU/선택 index/logit·mask 해시를 첫 측정 산출물에 결속합니다. 생성기는 이 증거를 요청과 대조하고 로컬 model-cache 경로가 든 private IPC 파일을 제거한 뒤 기존 디렉터리를 교체하지 않는 원자적 게시만 허용합니다. 포함된 9×9/1024² 자료는 연결 상태만 확인하는 합성 smoke fixture이므로 Canny/LSD/SAM의 실제 고지도 성능 순위를 뜻하지 않습니다. 형식과 EfficientSAM-Ti ONNX 계약은 [`benchmarks/README.md`](benchmarks/README.md)와 [`benchmarks/ADAPTER_CONTRACT.md`](benchmarks/ADAPTER_CONTRACT.md)에 있습니다.

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
  네트워크 또는 `requests` 미설치 문제일 수 있습니다. 필요 시 수동 다운로드 경로를 사용하세요. HED 다운로드는 Python 표준 라이브러리를 사용하므로 `requests`와 무관합니다.
- 엣지 미리보기에 아무것도 보이지 않음
  래스터 범위 안으로 확대하고, 다른 모델로도 비교해보세요.
- AI 기능이 당장 안 되는 환경임
  `Freehand` 모드는 계속 사용할 수 있습니다.
- DEM 실행 전 차단됨
  등고선 편집을 저장하고, 숫자형 고도 필드·서로 다른 두 고도값·비공선 vertex·미터 단위 투영 CRS·Processing provider·출력 격자 크기를 확인하세요. 기존 출력이 QGIS에 로드되어 있다면 안전한 덮어쓰기를 위해 먼저 제거해야 합니다.

## 🧭 Repository Map

- `ai_vectorizer/`: QGIS 플러그인 UI, 추적 도구, DEM 파이프라인
- `benchmarks/`: 격리 worker, checksummed manifest, 평가 지표와 합성 fixture
- `tests/`: QGIS 비의존 코어·benchmark 계약 테스트
- `ROADMAP.md`: 로컬 지형 복원 단계와 합격 기준

## 📁 Release Packaging

- 현재 개발 소스는 metadata `0.1.4` 이후의 기능을 포함합니다. 새 ZIP을 공개하기 전 `metadata.txt`와 릴리스 문서의 버전을 다음 릴리스로 올려야 합니다.
- 릴리스 폴더는 직접 수정하지 않고 루트 소스에서 다시 생성하는 흐름을 권장합니다.
- 빌드: `python3 scripts/package_release.py`
- 동기화 확인: `python3 scripts/package_release.py --check`

## 🇬🇧 English Summary

- ArchaeoTrace is a local-first QGIS plugin for tracing elevation contours on historical maps and building reviewable terrain hypotheses.
- The plugin metadata version is `0.1.5`; this update makes Human-led Assist the responsive default and keeps SAM weights outside the plugin directory.
- `Freehand` needs no external model or OpenCV, but uses NumPy from the QGIS Python environment.
- Default mouse-led `Canny` Human Assist uses a NumPy local-edge fallback when OpenCV is absent; LSD/HED/SAM still require OpenCV.
- `MobileSAM` and `SAM` also require `PyTorch` plus their backend packages and model weights.
- EfficientSAM-Ti ONNX is benchmark-only and is not a product default or a UI option.
- The source-tree DEM workflow requires saved elevation contours in a projected metre CRS; its output is a reviewable hypothesis, not an archaeological ground truth.

## 📚 Citation
[![Cite this repository](https://img.shields.io/badge/Cite_this-repository-2ea44f?logo=github)](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology)
[![Star this repository](https://img.shields.io/github/stars/lzpxilfe/AI-Vectorizer-for-Archaeology?style=social)](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology)

인용 메타데이터는 [CITATION.cff](CITATION.cff)에 보관합니다.


```bibtex
@software{ArchaeoTrace2026,
  author = {lzpxilfe},
  title = {ArchaeoTrace: AI-assisted contour digitizing QGIS plugin for historical maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology},
  version = {0.1.5}
}
```

## 📄 License

GNU General Public License v2.0
