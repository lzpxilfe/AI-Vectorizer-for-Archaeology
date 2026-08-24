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

- plugin metadata는 `0.1.5` 공개 후보이며 `experimental=True`입니다.
- [공식 QGIS plugin 저장소](https://plugins.qgis.org/plugins/ai_vectorizer/)의
  `0.1.4` 다음 배포판을 `0.1.5`로 정리했습니다. Git
  history의 `0.1.5–0.1.7`과 미공개 worktree의 `0.1.8`은 QGIS 저장소에 게시되지
  않은 개발 metadata였으며 이력은 rewrite하지 않습니다.
- metadata 대상은 QGIS `3.22–4.99`, source 계약은 Python `3.8+`입니다. 로컬에서는
  Python 3.8/3.10/3.12와 macOS QGIS 3.44.8을 확인했습니다. QGIS 3.22/3.44/4.2
  package matrix는 CI에 구성되어 있지만 이 후보의 원격 run은 아직 실행하지
  않았습니다. 정확한 범위는
  [`release-readiness 기록`](docs/RELEASE_READINESS_0.1.5.md)을 확인하세요.
- 개발판 `0.1.6–0.1.8`을 source나 ZIP으로 직접 설치했다면 낮은 버전 번호가 자동
  update로 인식되지 않을 수 있습니다. 기존 plugin을 제거한 뒤 검증된 `0.1.5` ZIP을
  다시 설치하세요. QGIS profile의 검증된 model 파일은 plugin 밖에 보존됩니다.

## Main features

- ✏️ 추가 model 없이 직접 입력하는 `Freehand`
- 🖋️ 검은 인쇄 획을 한 픽셀 중심선으로 만들고 제한된 Live-Wire와 결합하는 기본
  `Ink Centerline`
- 🎚️ `0%` 정확한 cursor부터 `100%` 전체 보조 경로까지 좌표를 실제로 혼합하는
  assist slider
- 📐 선택 가능한 `LSD`, `HED`, `MobileSAM`, `SAM (ViT-B)`, `Legacy Canny`
- 👁️ 클릭했을 때 채택될 경로를 보여 주는 초록색 preview와 anchor 기반 교정
- ↩️ QGIS edit buffer, constraint와 한 번의 Undo를 존중하는 새 선·기존 선 연장
- ⛰️ contour elevation과 선택적 `Spot Heights` 저장
- 🏔️ 저장된 입력에서 실험적 선형 TIN DEM과 GDAL hillshade 생성
- 🌏 한국어·English UI

구현 완료·실험·미구현 기능의 정확한 경계와 module별 책임은
[`Features and architecture`](docs/FEATURES_AND_ARCHITECTURE.md)에 있습니다.

## How it works

```text
Raster
  → Ink/LSD/HED/Canny edge 또는 MobileSAM/SAM point mask
  → bounded Live-Wire 또는 SAM용 strict A*
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
| `🖋 Ink Centerline` | 검은 획 중심선 + bounded Live-Wire | 추가 pip 없음; SciPy는 선택 | default |
| `📐 LSD` | OpenCV 선분 검출 + 공용 Live-Wire | OpenCV; SciPy는 선택 | optional |
| `🧠 HED` | Caffe HED edge map + 공용 Live-Wire | OpenCV + 약 56.1 MiB model; SciPy는 선택 | optional |
| `🎯 MobileSAM` | point mask + edge/skeleton/A* | OpenCV + PyTorch + backend + 약 38.8 MiB weights | optional |
| `🧩 SAM (ViT-B)` | point mask + edge/skeleton/A* | OpenCV + PyTorch + backend + 약 357.7 MiB checkpoint | optional |
| `🔧 Legacy Canny` | gradient edge + Live-Wire | OpenCV·SciPy는 선택 | compatibility |

QGIS Python의 NumPy는 plugin 공통 전제입니다. SciPy가 없으면 모든 non-SAM edge
mode(Ink/LSD/HED/Canny)는 방향 인식 Live-Wire 대신 제한 범위의 NumPy nearby-edge
snap으로 돌아갑니다. 탐색 범위를 제한해도
글자·기호·인접선에서 오추적할 수 있으므로 초록색 경로를 확인하고 anchor나
Freehand로 교정하세요. SAM의 `Auto Path`는 선택 기능이며 assist `0%`에서는 model과
edge 계산을 생략합니다.

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
3. 기본 `Ink Centerline` 또는 원하는 mode와 assist 강도를 고릅니다.
4. edge mode는 `Trace Preview`로 확인합니다. SAM mode는 추적 중 interactive green
   preview를 확인합니다.
5. 클릭·드래그로 anchor를 정하고 `Enter` 또는 우클릭으로 edit buffer에 넣습니다.
   시작점 근처에서 닫으면 elevation을 입력할 수 있습니다.
6. `Save Layer Edits`로 contour를 저장한 뒤 `Step 4 > DEM 생성…`에서 pixel size와
   output을 확인합니다.

## Data, network and safety boundary

- raster crop, vector geometry와 local inference 결과를 원격 inference 서버로 보내지
  않으며 기본 telemetry가 없습니다.
- network는 사용자가 HED/SAM model을 다운로드할 때 사용합니다. SAM Check/Status는
  유효한 local checkpoint가 있으면 size와 SHA-256을 offline 확인하고, 파일이 없을
  때만 고정 source의 availability를 조회합니다. EfficientSAM benchmark도 명시적인
  `model fetch`만 network를 사용합니다.
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

`benchmarks/`는 최종 ordered centerline, 실행 환경, 입력·출력 SHA-256, 시간·RAM과
topology 지표를 기록하는 개발자용 harness입니다. 현재 포함된 자료는 synthetic
contract fixture뿐이며 실제 역사 지도 정확도나 다른 도구보다 낫다는 근거가 아닙니다.
명령과 evidence 형식은 [`benchmarks/README.md`](benchmarks/README.md)를 참고하세요.

## Repository guide

| 문서·경로 | 내용 |
| --- | --- |
| [`ai_vectorizer/`](ai_vectorizer/) | QGIS plugin source와 ZIP 사용자 guide |
| [`docs/FEATURES_AND_ARCHITECTURE.md`](docs/FEATURES_AND_ARCHITECTURE.md) | 실제 기능, 구현 흐름, 안전 경계와 module map |
| [`ROADMAP.md`](ROADMAP.md) | 구현됨·다음 단계·의도적인 비목표 |
| [`docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md`](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md) | 오픈소스 원칙과 작업 gate |
| [`docs/RELEASE_READINESS_0.1.5.md`](docs/RELEASE_READINESS_0.1.5.md) | 실행 검증, artifact identity와 잔여 위험 |
| [`benchmarks/`](benchmarks/) | 격리 worker, manifest, 지표와 synthetic fixture |
| [`tests/`](tests/) | pure core, packaging와 QGIS safety 회귀 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 개발 환경, test tier, 문서·data 기여 규칙 |
| [`SECURITY.md`](SECURITY.md) | 취약점·data-loss 제보와 진단 redaction |

## Version and release discipline

`ai_vectorizer/metadata.txt`가 배포 버전의 source of truth입니다. 일상 개발은
`Unreleased` changelog에서 진행하고 metadata 숫자는 release 준비가 시작될 때만 한
번 바꿉니다. CI는 metadata, citation, 문서와 artifact 이름이 같은지 검사합니다.
과거 Git commit과 tag는 지우거나 다시 쓰지 않습니다. 자세한 기록은
[`CHANGELOG.md`](CHANGELOG.md)와 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

release tree를 직접 수정하지 말고 source에서 다시 만드세요.

```bash
python3 scripts/package_release.py
python3 scripts/package_release.py --check
```

현재 감사에서는 commit, push, tag, GitHub Release나 QGIS upload를 수행하지
않았습니다.

## Contributing and citation

bug report, 번역, 문서, 재배포 가능한 sample map과 code contribution을 환영합니다.
실제 데이터는 출처·license·개인정보를 확인하고, 성능 표현에는 재현 절차와 실패
사례를 함께 남겨 주세요. 시작 방법은 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

연구·교육·실무에서 사용했다면 저장소의 [`CITATION.cff`](CITATION.cff)를
이용해 인용할 수 있습니다.

## License

GNU General Public License v2.0. [`LICENSE`](LICENSE)
