# ArchaeoTrace features and architecture

이 문서는 ArchaeoTrace의 공식 experimental `0.1.5` 기준선과 현재
experimental `0.1.6` 후보에 들어 있는
기능, 각 기능이 구현된 방식,
현재 안전 경계와 아직 구현되지 않은 계획을 한곳에 정리합니다. ArchaeoTrace는
고지도 등고선을 사용자가 검수하며 벡터화하고, 고도 의미를 부여한 뒤 검토 가능한
DEM과 hillshade까지 만드는 오픈소스 QGIS 플러그인입니다. 계정이나 원격 추론 서비스는
필요하지 않으며 지도와 추적 계산은 로컬에서 처리합니다.
현재 0.1.6 후보 ZIP의 정확한 identity는
[`RELEASE_READINESS_0.1.6.md`](RELEASE_READINESS_0.1.6.md), 공식 QGIS 0.1.5
download와 repository-local `d292…` 사전 후보는
[`RELEASE_READINESS_0.1.5.md`](RELEASE_READINESS_0.1.5.md)에 분리해 기록합니다.

## Status at a glance

| 영역 | 현재 source 상태 | 설명 |
| --- | --- | --- |
| 수동·반자동 선 추적 | 구현 | Freehand, 다중 스케일 Ink Centerline; 기존 방법은 Advanced / Legacy에 보존 |
| Smart Recovery | 실험적·기본 OFF | 검증된 EfficientSAM-Ti를 Ink 저신뢰 구간의 corridor prior로만 사용하고 실패 시 Ink 유지 |
| QGIS 편집 통합 | 구현 | 새 피처, 기존 피처 연장, 고도 필드·값 변경을 편집 버퍼와 한 번의 Undo로 관리 |
| 표고 데이터 | 구현 | 등고선 숫자 고도와 선택적 Spot Heights 저장 |
| 지형 산출 | 실험적 구현 | 저장된 입력으로 선형 TIN DEM과 GDAL hillshade 생성 |
| 모델 무결성 | 구현 | 고정 URL·크기·SHA-256 검증, 임시 파일, 원자 게시, 실패 시 복원 |
| 재현 benchmark | 개발자용 구현 | 4개 독립 method ID, worker v1/v2 호환, 공개 8×6 dataset 계약과 최종 centerline 지표 |
| 공개 역사 지도 dataset | 미완성 gate | USGS 1개 도엽·6개 draft crop staged; 나머지 7개 도엽·독립 검수 전에는 ranking 불가 |
| 위상 QA·불확실성·DEM provenance | 미구현 | 교차·중복·고도 이상 검출, NoData/불확실성, sidecar manifest는 로드맵 항목 |

`구현`은 기능 경로가 코드와 테스트에 존재한다는 뜻입니다. 실제 고지도 정확도나
다른 도구보다 빠르다는 뜻은 아닙니다. 저장소의 합성 fixture는 wiring과 형식 계약을
검증할 뿐 제품 성능 근거로 사용하지 않습니다.

## End-to-end data flow

```text
QgsRasterLayer
  → 크기·자료형 제한이 있는 uint8 raster cache
  → source-grid 다중 스케일 LineEvidence
  → bounded Live-Wire Ink champion
  → (선택) 저신뢰 판정 → EfficientSAM corridor prior → strict challenger
  → 안전 arbiter가 champion/challenger 중 선택
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

`core/edge_detector.py`의 기존 단일 15px black top-hat 경로는
`detect_edges()`의 `ink-livewire-v1` 호환 기준선으로 유지됩니다. 새
`detect_ink_evidence()`는 9·15·31 source-pixel black top-hat을 RGB 각 채널과
명도에서 계산합니다. 원본 격자에 고정된 tile과 halo에서 강건하게 정규화하므로 같은
source 위치가 pan/zoom과 tile 경계 때문에 임의로 달라지지 않게 설계했습니다.
128px tile core를 완성한 뒤 16px response halo와 최대 filter 반경 15px을 합친
31px source context를 읽고, 각 core의 threshold·skeleton·방향장을 그 halo에서
독립 계산합니다. 현재 제품 v2는 native integer raster에서만 이 고정 source-DN
계약을 활성화합니다. float raster 또는 halo 포함 범위가 1000×1000 source pixel을
넘는 축척에서는 잘못된 block stretch/cache/source 단위 혼합 대신 Ink v1과 구체적인
fallback 사유를 표시합니다. fallback용 cache는 확장 v2 block과 분리해 0.1.5와 같은
visible extent·resampling·8-bit 변환으로 읽고, v2 성공 때만 확장 source-grid
transform을 게시합니다.

`core/line_evidence.py`의 QGIS 독립 `LineEvidence`는 같은 크기의 연속
`center_score`, `tangent_x`, `tangent_y`, `coherence`, `scale_px`와 호환용 이진
`centerline`을 가집니다. float 값은 finite와 범위를 검증합니다. 작은 성분과 spur는
호환 중심선을 만들 때 정리하고, 연속 score에는 약한 단절을 남겨 Live-Wire가 건널
가능성을 보존합니다. 방향·coherence·대표 scale은 한 번 계산해
`core/livewire.py`에 전달하며, evidence가 없으면 기준선 경로를 그대로 사용합니다.

SciPy가 있으면 `core/livewire.py`가 최근 기준점 주위의 제한된 창에서 방향·선
거리·우회 비용을 포함한 단일 최단경로 트리를 백그라운드로 만듭니다. 커서 이동은
그 트리의 predecessor를 역추적하므로 매번 전체 A*를 다시 실행하지 않습니다.
SciPy가 없으면 모든 non-SAM edge mode(Ink/LSD/HED/Canny)가 제한 창의 NumPy
nearby-edge snap으로 돌아가며, 기본 ZIP 설치만으로도 Ink와 Canny를 사용할 수
있습니다. `0%` 보조는 엣지와 모델 작업을 건너뛰고 정확한 커서 좌표를 사용하고,
`100%`는 전체 보조 경로를 사용하며 중간값은 두 경로의 실제 좌표를 혼합합니다.

### Smart Recovery (Experimental)

Smart Recovery는 항상 Ink v2 경로를 champion으로 먼저 보여 줍니다. 경로의 하위
분위수 지지도, 가장 긴 저지지도 구간, 방향 일관성, 우회율, 분기 밀도와 끝점 상태가
동결된 policy에서 저신뢰로 판정되고 사용자가 기능을 켠 경우에만
EfficientSAM-Ti challenger를 계산합니다. `Retry with Smart Recovery`는 현재 구간을
명시적으로 다시 평가합니다.

`core/efficientsam_recovery.py`는 content-addressed model bundle과 CPU ONNX Runtime을
offline으로 재검증합니다. bundle hashing과 ONNX session 생성은 취소 가능한
generation-guarded `QgsTask`에서 수행하며 준비 중에도 Ink tracing을 즉시 시작합니다.
Recovery image는 native Byte raster에만 허용하고, 더 넓은 정수형은 Ink v2를 유지한
채 명시적으로 fallback합니다. mask는 최종 line이나 Ink와의 이진 OR가 아니라
`core/smart_recovery.py`가 만드는 soft corridor cost에만 들어갑니다. 시작·끝점,
기존 탐색창·우회 한계, 강한 Ink 보존, 개선량, 평행선 전환과 비정상 분기 검사를 모두
통과한 challenger만 채택합니다. 모델·runtime 미설치, hash 불일치, 잘못된 output,
오류·취소·stale 결과에서는 다른 backend를 조용히 쓰지 않고 같은 Ink champion과
`Ink fallback` 사유를 유지합니다. 모델은 `Install Recovery Model`을 눌렀을 때만
받으며 기본값은 신규·기존 설정과 관계없이 OFF입니다. regular cache object의 hash가
틀리면 같은 버튼이 `Repair Recovery Model`로 바뀌어 명시적 격리·재다운로드를 하고,
실패·취소 시 미완료 object만 되돌립니다. 이미 hash 검증된 replacement는 유지하고
symlink·junction/reparse point를 포함한 unsafe cache object는 자동 변경하지 않습니다.

### Advanced / Legacy methods

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

기존 LSD, HED, MobileSAM, SAM (ViT-B), Legacy Canny의 model index 0–5와 설정 해석은
deprecation 기간 동안 유지됩니다. 이번 변경에서 backend code나 사용자 model 파일을
삭제하지 않고 접힌 UI로만 이동했습니다.

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

QGIS provider read와 객체 변경은 main thread에서 수행합니다. Ink evidence,
Live-Wire tree, Recovery model 검증·session 준비, inference와 DEM 처리는 취소 가능한
`QgsTask`에서 수행합니다.
Ink 결과는 요청 generation과 raster/source·extent·CRS identity가 그대로일 때만
게시합니다. HED/SAM legacy 다운로드·모델 준비와 일부 raster 준비는 아직 UI
thread에서 길어질 수 있어 추가 분리 대상입니다.

## Raster and allocation boundaries

`core/raster_utils.py`는 provider block을 NumPy 배열로 만들기 전에 pixel 수와
자료형별 byte 수를 검사합니다. 현재 단일 읽기는 최대 2,500만 pixel과 64 MiB
payload로 제한합니다. 동결된 v1/display 경로는 기존 8-bit 변환을 그대로 쓰고,
v2는 같은 provider block에서 view-dependent stretch가 없는 native integer DN을
별도로 보존합니다. NoData는 dtype의 고정된 밝은 값으로 채워 인공적인 어두운 선을
만들지 않습니다. float source는 고정된 전역 range 계약이 없으므로 v2에서 fail-closed
하고 v1을 유지합니다. 추적 cache와 SAM/EfficientSAM 입력에도 별도 dimension과
iteration 상한이 있어 손상되거나 과대한 입력을 계산 전에 거부합니다.

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

`core/model_store.py`는 EfficientSAM benchmark와 Smart Recovery artifact에 같은
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

worker request v2는 선택적 `previous_xy`로 이전 확정점에서 시작점으로 들어오는
방향을 표현하며, SAM positive point로 사용하지 않습니다. v1 request는 계속 읽고
동일한 v1 evidence hash를 유지합니다. 독립 method ID는 `ink-livewire-v1`,
`ink-livewire-v2`, `efficientsam-ti-onnx-v1`,
`ink-v2-effsam-recovery-v1`입니다.

포함된 synthetic smoke fixture는 코드 경로와 증거 형식만 검사합니다. 공개 dataset
template은 8개 도엽×6개 무손실 PNG, 도엽 단위 calibration/locked holdout, 8개 난이도
층, 원본·권리 snapshot·crop·주석 검수 hash를 강제합니다. 재배포 가능한 USGS 4개와
권리가 명확한 한국·한반도 4개 도엽 및 독립 검수가 실제로 채워질 때까지
`publication_ranking_eligible`은 false이며 성능 순위를 주장할 수 없습니다.
첫 calibration 도엽 `cal-01`은 USGS HTMC East Denver 1890 원본·권리 snapshot과
6개 PNG·prompt·중심선 초안까지 staged 상태입니다. `benchmarks.public_assets`는
PNG가 선언 좌표의 원본 픽셀과 정확히 같은지 검증하며, 중심선은 별도 검수와
adjudication이 끝날 때까지 명시적으로 draft로 남습니다.

## Source map

| 경로 | 책임 |
| --- | --- |
| `ai_vectorizer/plugin.py` | plugin 등록·dock·toolbar·unload 수명주기 |
| `ai_vectorizer/ui/main_dialog.py` | tracing UI, layer/model 선택, preview와 출력 wiring |
| `ai_vectorizer/ui/dem_dialog.py` | DEM 입력·격자·출력 UI와 task 상태 |
| `ai_vectorizer/tools/smart_trace_tool.py` | map event, trace session, preview, QGIS edit transaction |
| `ai_vectorizer/core/edge_detector.py` | Ink/LSD/HED/Canny edge·centerline과 HED artifact |
| `ai_vectorizer/core/line_evidence.py` | QGIS 독립 연속 Ink score·방향·coherence·scale 계약 |
| `ai_vectorizer/core/livewire.py` | 제한 창의 방향 인식 최단경로 트리와 assist 혼합 |
| `ai_vectorizer/core/smart_recovery.py` | 저신뢰 gate, corridor cost와 champion/challenger arbiter |
| `ai_vectorizer/core/efficientsam_recovery.py` | 검증된 EfficientSAM bundle의 offline CPU 실행 adapter |
| `ai_vectorizer/recovery.py` | UI에서 공유하는 Recovery 상태 계약 |
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
[`RELEASE_READINESS_0.1.6.md`](RELEASE_READINESS_0.1.6.md), 오픈소스 개발 원칙과
작업 gate는 [`OPEN_SOURCE_DEVELOPMENT_PLAN.md`](OPEN_SOURCE_DEVELOPMENT_PLAN.md)에
있습니다.
