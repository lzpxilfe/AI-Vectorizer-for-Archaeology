# ArchaeoTrace features and architecture

이 문서는 ArchaeoTrace `0.1.5`에 실제로 들어 있는 기능, 각 기능이 구현된 방식,
현재 안전 경계와 아직 구현되지 않은 계획을 한곳에 정리합니다. ArchaeoTrace는
고지도 등고선을 사용자가 검수하며 벡터화하고, 고도 의미를 부여한 뒤 검토 가능한
DEM과 hillshade까지 만드는 오픈소스 QGIS 플러그인입니다. 계정이나 원격 추론 서비스는
필요하지 않으며 지도와 추적 계산은 로컬에서 처리합니다.

## Status at a glance

| 영역 | `0.1.5` 상태 | 설명 |
| --- | --- | --- |
| 수동·반자동 선 추적 | 구현 | Freehand, 기본 Ink Centerline, LSD, HED, MobileSAM, SAM, Legacy Canny |
| QGIS 편집 통합 | 구현 | 새 피처, 기존 피처 연장, 고도 필드·값 변경을 편집 버퍼와 한 번의 Undo로 관리 |
| 표고 데이터 | 구현 | 등고선 숫자 고도와 선택적 Spot Heights 저장 |
| 지형 산출 | 실험적 구현 | 저장된 입력으로 선형 TIN DEM과 GDAL hillshade 생성 |
| 모델 무결성 | 구현 | 고정 URL·크기·SHA-256 검증, 임시 파일, 원자 게시, 실패 시 복원 |
| 재현 benchmark | 개발자용 구현 | 격리 worker, 엄격한 manifest, 실행·입력·출력 해시와 최종 centerline 지표 |
| EfficientSAM-Ti | benchmark 전용 | 고정 split ONNX를 CPU-only로 비교하며 UI 추적 옵션은 아님 |
| 위상 QA·불확실성·DEM provenance | 미구현 | 교차·중복·고도 이상 검출, NoData/불확실성, sidecar manifest는 로드맵 항목 |

`구현`은 기능 경로가 코드와 테스트에 존재한다는 뜻입니다. 실제 고지도 정확도나
다른 도구보다 빠르다는 뜻은 아닙니다. 저장소의 합성 fixture는 wiring과 형식 계약을
검증할 뿐 제품 성능 근거로 사용하지 않습니다.

## End-to-end data flow

```text
QgsRasterLayer
  → 크기·자료형 제한이 있는 uint8 raster cache
  → EdgeDetector 또는 point-prompt mask
  → bounded Live-Wire / 공용 A* trace kernel
  → 클릭 결과와 동일한 초록색 preview
  → QGIS vector layer edit buffer + elevation
  → 저장된 contour + 선택적 spot height
  → QGIS linear TIN → GeoTIFF DEM → GDAL hillshade
```

AI나 엣지 검출 결과가 곧바로 최종 피처가 되지는 않습니다. 사용자가 초록색 경로를
확인하고 기준점을 채택해야 편집 버퍼에 들어가며, 최종 저장도 QGIS의
`Save Layer Edits`에서 결정합니다.

## Tracing modes

### Freehand

래스터 모델을 실행하지 않고 사용자의 클릭과 드래그를 그대로 기록합니다. 추가
pip 패키지나 model이 없거나 자동 경로가 맞지 않는 상황에서 사용할 수 있는 기준
경로입니다. QGIS와 plugin 공통 NumPy, 유효한 raster·2D line output은 필요합니다.

### Ink Centerline — default

`core/edge_detector.py`가 어두운 인쇄 획의 국소 black top-hat 반응을 정리하고
morphology와 thinning으로 한 픽셀 중심선을 만듭니다. Canny의 양쪽 경계 대신
등고선 획의 가운데를 비용 지도로 사용하는 것이 목적입니다.

SciPy가 있으면 `core/livewire.py`가 최근 기준점 주위의 제한된 창에서 방향·선
거리·우회 비용을 포함한 단일 최단경로 트리를 백그라운드로 만듭니다. 커서 이동은
그 트리의 predecessor를 역추적하므로 매번 전체 A*를 다시 실행하지 않습니다.
SciPy가 없으면 모든 non-SAM edge mode(Ink/LSD/HED/Canny)가 제한 창의 NumPy
nearby-edge snap으로 돌아가며, 기본 ZIP 설치만으로도 Ink와 Canny를 사용할 수
있습니다. `0%` 보조는 엣지와 모델 작업을 건너뛰고 정확한 커서 좌표를 사용하고,
`100%`는 전체 보조 경로를 사용하며 중간값은 두 경로의 실제 좌표를 혼합합니다.

### LSD, HED and Legacy Canny

- `LSD`는 OpenCV Line Segment Detector 결과를 공용 추적 경로에 결합합니다.
- `HED`는 검증된 deploy definition과 Caffe weights로 비보정 edge map을 만들고
  공용 경로 탐색에 사용합니다.
- `Legacy Canny`는 기존 그래디언트 경계 기반 동작을 비교·호환용으로 유지합니다.

세 모드 모두 검출 결과를 자동 저장하지 않고 사용자가 채택할 경로의 비용으로
사용합니다. LSD와 HED는 검증 범위의 OpenCV가 필요하고, Legacy Canny는 NumPy
fallback을 가집니다. SciPy가 있으면 Ink와 같은 방향 인식 Live-Wire를 사용하고,
없으면 공통 bounded nearby-edge snap으로 전환합니다.

### MobileSAM and SAM (ViT-B)

`core/sam_engine.py`가 point prompt로 mask를 예측하고,
`core/sam_trace_kernel.py`가 mask 정리·면적 제한·skeleton snap·edge 보조 비용을
거쳐 공용 `core/trace_kernel.py`의 순서 있는 centerline으로 바꿉니다. PyTorch,
각 backend 패키지와 별도 weights가 필요한 선택 기능입니다. mask는 최종 polygon이나
고고학적 사실로 직접 저장되지 않습니다.

### EfficientSAM-Ti

`core/efficientsam_onnx.py`와 benchmark worker에는 고정 해시의 split ONNX
encoder/decoder를 CPU-only로 검증하고 실행하는 경로가 있습니다. 실행 provider,
session 설정, prompt tensor와 반복 출력 증거를 기록하지만 현재 플러그인 UI나 기본
모델에는 연결하지 않았습니다. 실제 고지도 데이터 평가 없이 승격하지 않습니다.

## Editing and data safety

`tools/smart_trace_tool.py`는 QGIS map tool 수명주기와 편집 작업을 관리합니다.

- 새 피처와 기존 contour 연장은 QGIS edit buffer에 남습니다.
- geometry, 새 elevation field와 값 변경을 하나의 `beginEditCommand` /
  `endEditCommand` 단위로 묶습니다.
- constraint, geometry 또는 attribute 저장이 실패하면 `destroyEditCommand`로 함께
  되돌립니다.
- 성공한 한 작업은 QGIS Undo 한 번으로 되돌릴 수 있습니다.
- 레이어 제거, 출력 source 교체, CRS 변경, dialog 종료와 plugin unload 때 활성
  trace, background task, timer, rubber band와 임시 preview를 정리합니다.
- 현재 프로젝트에 로드된 dataset 경로를 새 출력으로 선택하면 path alias와
  symlink까지 비교해 writer 호출 전에 차단합니다.
- canvas, raster, output layer CRS가 다르면 저장 전에 좌표를 명시적으로 변환합니다.

QGIS 객체 변경은 main thread에서 수행합니다. 긴 Live-Wire 트리와 DEM 처리는
`QgsTask`를 사용하지만 HED/SAM 다운로드·모델 준비와 일부 raster 준비는 아직 UI
thread에서 길어질 수 있어 추가 분리 대상입니다.

## Raster and allocation boundaries

`core/raster_utils.py`는 provider block을 NumPy 배열로 만들기 전에 pixel 수와
자료형별 byte 수를 검사합니다. 현재 단일 읽기는 최대 2,500만 pixel과 64 MiB
payload로 제한하며, nodata와 정수·실수 자료형을 확인한 뒤 8-bit 작업 영상으로
정규화합니다. 추적 cache와 SAM/EfficientSAM 입력에도 별도 dimension과 iteration
상한이 있어 손상되거나 과대한 입력을 계산 전에 거부합니다.

## Model storage and network boundary

지도 crop이나 벡터 geometry를 원격 추론 서버로 전송하지 않습니다. 네트워크는
사용자가 모델 다운로드를 직접 실행할 때 사용합니다. SAM Check/Status는 유효한
local checkpoint가 있으면 size와 SHA-256을 offline 확인하고, 파일이 없을 때만
고정 source의 availability를 조회합니다. 상태 리포트에는 현재 작업 경로,
QGIS/Python 환경값과 model 경로가 포함될 수 있고 clipboard에도 복사되므로 공유 전에
내용을 검토하고 민감한 local path·환경값을 지워야 합니다.

HED와 SAM 계열 artifact는 plugin 설치 폴더가 아닌 QGIS profile 아래의 지속 저장소에
둡니다. 구현은 다음 계약을 적용합니다.

1. 허용한 HTTPS URL과 redirect 정책을 확인합니다.
2. 다운로드 byte 상한, 기대 크기와 SHA-256을 검사합니다.
3. 같은 안전한 디렉터리의 임시 파일에 쓰고 `fsync`합니다.
4. symlink인 디렉터리나 최종 파일을 거부합니다.
5. 검증된 파일만 원자적으로 게시하고, 교체 실패나 사후 검증 실패 시 기존 파일을
   복원합니다.
6. plugin-local 구버전 asset은 정확히 검증된 경우에만 profile 저장소로 옮깁니다.

`core/model_store.py`는 EfficientSAM benchmark artifact에 같은 원칙의
content-addressed bundle 계약을 제공합니다. HED와 PyTorch SAM 저장 구현은 아직
완전히 하나의 store로 통합되지 않았습니다.

## Terrain reconstruction

`ui/dem_dialog.py`, `core/dem_spec.py`, `core/dem_pipeline.py`가 다음 흐름을
담당합니다.

```text
saved contour LineString + numeric elevation
  + optional saved point height
  → qgis:tininterpolation
  → gdal:translate
  → native:rasterlayerstatistics
  → gdal:hillshade
  → staged DEM/hillshade publication
```

입력은 미터 단위 투영 CRS, finite 고도, 서로 다른 두 개 이상의 고도값과 TIN을
만들 수 있는 비공선 vertex를 가져야 합니다. 편집 중인 입력은 먼저 저장해야 합니다.
격자는 기본 최대 2,500만 셀입니다. 기존 출력, sidecar와 QGIS에 로드된 대상은 시작
전과 최종 게시 직전에 다시 검사하고, 처리 중 대상이 생기거나 바뀌면 결과 게시를
중단하고 원본을 복원합니다.

선형 TIN은 입력 범위 밖과 희소·오류 contour에 취약합니다. 따라서 현재 DEM은
검토 가능한 지형 가설이며 uncertainty layer나 provenance sidecar가 이미 구현된
것처럼 해석하면 안 됩니다.

## Reproducible benchmark path

`benchmarks/`는 추적 backend가 최종적으로 만든 ordered centerline을 같은 형식으로
평가합니다.

- strict JSON manifest와 입력·기준선·예측·설정·소스 SHA-256
- sample×method마다 새 worker process와 고정 CPU/thread 조건
- 실제 backend, fallback, provider, 반복 시간과 peak RSS 기록
- centerline F1/Dice, 거리, 연결성, 과잉 분기와 누락 junction을 분리한 지표
- 기존 결과를 덮지 않는 immutable run과 검증 뒤 no-replace 게시

포함된 synthetic smoke fixture는 코드 경로와 증거 형식만 검사합니다. 실제 지도
품질 판단에는 재배포 가능한 역사 지도 crop, 독립 기준선과 사전 고정 평가 절차가
추가로 필요합니다.

## Source map

| 경로 | 책임 |
| --- | --- |
| `ai_vectorizer/plugin.py` | plugin 등록·dock·toolbar·unload 수명주기 |
| `ai_vectorizer/ui/main_dialog.py` | tracing UI, layer/model 선택, preview와 출력 wiring |
| `ai_vectorizer/ui/dem_dialog.py` | DEM 입력·격자·출력 UI와 task 상태 |
| `ai_vectorizer/tools/smart_trace_tool.py` | map event, trace session, preview, QGIS edit transaction |
| `ai_vectorizer/core/edge_detector.py` | Ink/LSD/HED/Canny edge·centerline과 HED artifact |
| `ai_vectorizer/core/livewire.py` | 제한 창의 방향 인식 최단경로 트리와 assist 혼합 |
| `ai_vectorizer/core/trace_kernel.py` | QGIS 독립 A*, smoothing, ordered centerline |
| `ai_vectorizer/core/sam_trace_kernel.py` | SAM mask 후처리와 공용 kernel 연결 |
| `ai_vectorizer/core/sam_engine.py` | MobileSAM/SAM backend와 checkpoint 관리 |
| `ai_vectorizer/core/raster_utils.py` | bounded provider read와 uint8 정규화 |
| `ai_vectorizer/core/dem_spec.py` | 격자·경로·보간 계약과 원자 output 게시 |
| `ai_vectorizer/core/dem_pipeline.py` | QGIS Processing 검증·실행·결과 결속 |
| `ai_vectorizer/core/model_store.py` | 검증된 benchmark model bundle 저장 |
| `benchmarks/` | 격리 실행, evidence manifest와 평가 지표 |
| `scripts/package_release.py` | allowlist 기반 release tree와 결정적 ZIP 생성 |

현재 `smart_trace_tool.py`, `main_dialog.py`, `edge_detector.py`는 UI, 상태, IO와
계산 책임이 큰 파일입니다. 다음 구조 개선은 동작을 바꾸기 전에 session/controller,
layer edit service, model artifact store와 output guard를 분리하고 취소·model switch·
layer removal·unload 상태 전이를 테스트로 고정하는 방향입니다. 제품 경로에서 쓰이지
않는 `core/path_finder.py`와 `core/vectorizer.py`는 외부 import 호환성을 확인한 뒤
제거하거나 명시적 adapter로 정리해야 합니다.

## Where to change and how to verify

- 추적 수학이나 smoothing 변경: `core/trace_kernel.py`, `core/livewire.py`,
  `core/sam_trace_kernel.py`와 해당 `tests/test_*`를 먼저 수정합니다.
- QGIS 편집·수명주기 변경: `tools/smart_trace_tool.py`와
  `tests/test_qgis_safety.py`, `tests/test_qgis_runtime_safety.py`를 함께 봅니다.
- DEM 변경: `core/dem_spec.py`, `core/dem_pipeline.py`, `ui/dem_dialog.py`와
  `tests/test_dem_spec.py` 및 실제 QGIS runtime을 확인합니다.
- 모델·download 변경: 고정 출처·크기·SHA-256, rollback과 offline 재검증 테스트를
  반드시 함께 갱신합니다.
- 배포 파일 변경: `scripts/package_release.py`로 두 번 빌드해 byte identity를
  확인하고, 생성 ZIP 자체를 QGIS에서 import합니다.

개발 순서와 아직 없는 기능의 합격 기준은 [`../ROADMAP.md`](../ROADMAP.md), 현재
릴리스 검증 증거와 잔여 위험은
[`RELEASE_READINESS_0.1.5.md`](RELEASE_READINESS_0.1.5.md), 오픈소스 개발 원칙과
작업 gate는 [`OPEN_SOURCE_DEVELOPMENT_PLAN.md`](OPEN_SOURCE_DEVELOPMENT_PLAN.md)에
있습니다.
