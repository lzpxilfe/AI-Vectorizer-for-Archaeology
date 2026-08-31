# ArchaeoTrace: local terrain reconstruction roadmap

## Direction

ArchaeoTrace의 목표는 서버나 유료 서비스에 사용자 자료를 보내지 않고, 각자의 컴퓨터에서 고지도를 등록·벡터화·검수한 뒤 재현 가능한 DEM까지 만드는 오픈소스 QGIS 도구가 되는 것입니다.

핵심 원칙:

- 로컬 실행과 오프라인 재실행을 우선합니다.
- AI 결과를 최종 정답으로 저장하지 않고, 사용자가 검수할 수 있는 비보정 보조 신호로 사용합니다.
- 입력, 모델, 파라미터, CRS, 출력을 기록해 결과를 재현할 수 있게 합니다.
- 자동화 단계마다 고고학적 해석과 표고의 불확실성을 노출합니다.

공식 QGIS 저장소에는 experimental `0.1.5`가 공개되어 있고, 현재 `main`의
후속 기능은 experimental `0.1.6` 후보입니다. 아직 별도 QGIS/GitHub Release나
tag로 게시하지 않았습니다. 추적 추론과 지도
데이터는 로컬에 머뭅니다. 모델 다운로드는
네트워크를 사용하고, SAM 상태 확인은 유효한 로컬 checkpoint가 없을 때만 고정된
source의 availability를 조회합니다. 유효한 로컬 checkpoint는 오프라인에서 크기와
SHA-256을 확인합니다.
등고선/표고점 저장과 실험적 TIN DEM은 구현되어 있고, 위상 QA, 입력·모델 해시
sidecar, 불확실성/NoData 산출은 M3의 미구현 목표입니다.

## Current scope and priority

ArchaeoTrace는 기능 수를 늘리기보다 다음 고고학 작업의 수직 흐름을 완성합니다.

`로컬/오프라인 추적 → 등고선 고도 의미 부여 → 위상 검수 → DEM/hillshade → 불확실성·재현 기록`

제품 우선순위:

1. **배포 신뢰성** — 공식 QGIS 저장소의 stable 릴리스, 결정적 ZIP, 실제 QGIS
   3/4 호환성, 안전한 편집 버퍼·Undo, 모델 무결성을 먼저 보장합니다.
2. **측정 가능한 사용자 도움** — 합법적으로 재배포 가능한 실제 고지도 crop에서
   수동 QGIS와 공개적으로 실행할 수 있는 도구를 같은 조건으로 사용해 시간, 클릭,
   수정, 잘못된 분기, 끊김과 최종 DEM 오류를 함께 측정합니다.
3. **고고학 QA** — 일반 벡터화 서비스가 끝나는 지점에서 등고선 교차, 중복,
   자체 교차, 고도 간격 이상, 결측 구간을 찾아 수정 대상으로 돌려줍니다.
4. **재현 가능한 지형 산출물** — 입력·모델·파라미터·CRS·레이어 해시와 DEM
   검증 결과를 산출물에 결속하고, 낮은 데이터 밀도와 외삽 영역을 불확실성으로
   분리합니다.

범용 polygon·건물·도로 자동완성이나 필수 cloud inference는 위 네 단계가 검증되기
전에는 우선순위에 넣지 않습니다. 기능·구현 경계는
[`docs/FEATURES_AND_ARCHITECTURE.md`](docs/FEATURES_AND_ARCHITECTURE.md), 개발 원칙과
Gate A–E는 [`docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md`](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md),
현재 `0.1.6` 후보의 검증 결과와 남은 release blocker는
[`docs/RELEASE_READINESS_0.1.6.md`](docs/RELEASE_READINESS_0.1.6.md), 공식
`0.1.5`의 역사적 artifact 기록은
[`docs/RELEASE_READINESS_0.1.5.md`](docs/RELEASE_READINESS_0.1.5.md)에 분리합니다.

## Release and community-readiness gate

공식 `0.1.5`는 새 모델을 기본으로 추가하지 않은 experimental release입니다.
현재 `0.1.6` 후보와 다음 release는 데이터 무결성, 모델 공급망, QGIS 편집
수명주기, Python/QGIS 호환성, 재현 패키징을 닫아야 합니다. 진입 조건은
다음과 같습니다.

- 동일 소스가 Linux/Windows에서 같은 ZIP SHA-256을 생성합니다.
- 빌드된 ZIP 자체가 QGIS 3/4에서 import와 편집 회귀를 통과합니다.
- 현재 source는 `experimental=True`를 유지하고, stable 전환 PR에서 `False`로 바꾼
  바로 그 ZIP이 전체 원격 CI를 다시 통과한 뒤 공식 저장소에 올립니다.
- 기본 Ink/Freehand는 추가 pip 없이 동작하고 선택 모델 실패가 기본 흐름을
  망가뜨리지 않습니다.
- 실제 고지도 benchmark의 데이터 라이선스, 기준선 검수 절차와 비교 protocol을
  먼저 고정합니다.

## M0 — Contours to DEM vertical slice

상태: 1차 구현

범위:

- 고도 등고선을 구조선으로, 선택적 표고점을 점 입력으로 사용하는 QGIS 선형 TIN
- GeoTIFF DEM 후속 GDAL hillshade
- 백그라운드 작업, 진행률, 취소, 결과 레이어 자동 추가
- 투영 CRS(m), 고도, 기하, 격자 크기, 편집 상태, 덮어쓰기 검사
- 캔버스 CRS와 출력 레이어 CRS가 다를 때 등고선/표고점 저장 좌표 변환

합격 기준:

- 잘못된 CRS·고도·출력은 처리 시작 전에 이유와 함께 차단됩니다.
- 유효한 소형 입력이 DEM과 hillshade 두 파일을 만들고 QGIS 프로젝트에 로드됩니다.
- UI 스레드가 긴 처리 동안 응답 상태를 유지합니다.

## M0.5 — Human-led Ink Centerline + direction-aware Live-Wire

상태: metadata `0.1.6` experimental 후보 (macOS QGIS 3.44.8 clean-profile 실제
map-tool 저장과 QGIS 3.22.16/3.44.13/4.2.1 원격 package/runtime-safety 계약을
통과한 기능 소스이며, 버전이 반영된 exact ZIP 증거는 0.1.6 readiness 기록에 결속)

완료:

- 검은 획의 국소 black top-hat 반응을 잡음 제거·세선화해 Canny의 이중 경계 대신 단일 중심선 생성
- 기준점을 클릭할 때만 320×320 제한 창의 단일 출발점 최단경로 트리를 백그라운드에서 생성
- 커서 이동 때는 목표별 A* 재계산 없이 predecessor 역추적만 수행
- 영상 구조 텐서의 접선 방향 비용으로 글자 교차와 평행한 인접 등고선 이동을 억제
- 선까지의 거리 비용으로 짧은 인쇄·스캔 단절을 연결하고, 최대 우회율과 제한 창으로 장거리 이탈을 차단
- 진행 방향 쪽으로 탐색 창을 치우쳐 이전 선분을 거슬러 가는 경로를 줄임
- `0%` 정확한 커서, 중간값 좌표 혼합, `100%` 완전 보조 경로인 단일 슬라이더 계약
- 초록색 미리보기와 클릭 결과를 동일하게 유지하고 Ink/Legacy Canny의 OpenCV 의존성 제거
- 제한 창과 predecessor 재사용의 성능 수치는 fixture, hardware, commit, 반복
  분포를 담은 재현 보고서를 만들기 전에는 제품 성능 근거로 사용하지 않음
- Legacy Canny는 비교·호환용 선택지로 보존하고 Ink Centerline을 기본값으로 전환

다음 모델 경계:

- SAM/HED를 기본값으로 교체하지 않습니다. 둘 다 일반 물체·경계 모델이라 숫자, 기호, 도로선도 강하게 반응합니다.
- 실제 모델 교체는 고지도 크롭과 수작업 등고선 중심선으로 학습한 compact U-Net 계열을 우선 검토합니다.
- 학습 증강에는 글자·기호 가림, 짧은 선 단절, 얼룩, 스캔 밝기 변화를 포함하고 clDice 같은 위상 보존 손실을 비교합니다.
- 모델 출력도 최종 선을 독단적으로 만들지 않고 Live-Wire의 비보정 guide cost로만 결합합니다.

## M1 — Local model benchmark before replacement

상태: M1.3 Ink v1/v2/Recovery 독립 worker와 공개 8×6 dataset 계약 구현,
USGS 실도엽 1장·crop 6개 staged, 나머지 7장과 독립 검수 대기

완료:

- SHA-256·출처·라이선스·CPU/fallback/timing을 강제하는 데이터셋 매니페스트
- 모든 방법을 최종 ordered centerline으로 정규화하는 교환 형식
- tolerance F1, exact centerline Dice, 양방향 거리, 단절·연결성 및 과잉/누락 분기를 분리한 topology 지표
- strict JSON, sample CSV, summary CSV, 세대별 hash commit과 합성 smoke fixture
- 동일 CPU/platform/thread 조건, 반복 출력 결정성, 실패 실행의 시간·RAM까지 포함하는 적격성 검사
- EfficientSAM-Ti split ONNX의 고정 commit/hash 및 CPU adapter 계약
- `SmartTraceTool`의 실제 A*·부분 경로·5점 이동평균·Chaikin을 QGIS 독립 공용 커널로 추출
- 제품과 동일한 `EdgeDetector → cost map → trace kernel → ordered centerline` Canny/LSD worker
- sample×method별 fresh process, 1 CPU thread, OpenCL 차단, 반복 해시·시간·RSS 직접 기록
- worker의 입력·설정·소스·반복 출력 해시와 artifact 메타데이터를 대조한 뒤 검증된 manifest를 no-replace 원자 게시하는 `benchmarks generate` 흐름
- 실제 OpenCV 4 통합 회귀와 OpenCV 5 반환 배열 모양의 합성 계약 테스트
  (OpenCV 5 runtime 자체는 `0.1.6` 지원·검증 범위가 아님)
- 고정 commit·크기·SHA-256을 코드 계약으로 둔 EfficientSAM-Ti split ONNX model store
- 명시적 fetch만 네트워크를 사용하고 실행은 content-addressed cache를 매번 재검증하는 offline 경계
- encoder/decoder별 CPUExecutionProvider·단일 thread·sequential/graph-opt 설정과 OpenCV 상태를 실제 readback으로 attest하는 split adapter
- 기존 SAM의 mask close·면적 guard·Canny 보조 cost·skeleton snap·strict A*를 공용 커널로 추출
- EfficientSAM mask를 공용 커널의 최종 ordered centerline으로 바꾸는 fresh-process worker와 1024² 합성 실제 모델 smoke
- semantic prompt·실제 float32 point/label tensor와 반복별 IoU/선택 index/logit·mask SHA-256을 첫 측정 artifact에 결속하는 증거 계약
- 모델 로드·이미지/encoder 준비·warm prompt latency를 분리하고 private model-cache IPC를 게시물에서 제거하는 생성 경계

다음 구현 경계:

- 같은 worker 계약으로 현재 PyTorch MobileSAM/SAM을 동일 prompt에서 실행
- 합법적으로 재배포 가능한 실제 고지도 크롭과 ordered 기준선 구축
- 현재 긴 경로에서 끝점을 이동시키는 historical smoothing과 endpoint-preserving 후보를 분리 평가

범위:

- 지도 종류·인쇄 상태·스캔 품질을 나눈 30–50개 등고선 크롭과 수작업 기준선을 버전 관리합니다.
- Canny/LSD/현재 SAM 모드와 EfficientSAM-Ti ONNX를 같은 입력에서 비교합니다.
- centerline Dice(clDice), 평균/상위 95% 선간 거리, 끊긴 구간, 잘못된 분기, CPU 시간, 최대 RAM을 기록합니다.
- 데이터 출처·라이선스와 모델 라이선스를 항목별로 기록합니다.

합격 기준:

- 재현 가능한 한 명령으로 모든 기준선을 평가하고 JSON/CSV 결과를 만듭니다.
- EfficientSAM을 넣을지 여부를 체감이 아닌 CPU 정확도·속도 지표로 결정합니다.

## M2 — EfficientSAM-guided hybrid line vectorizer

상태: 첫 보수적 `Smart Recovery (Experimental)` 제품 경로 구현 및 실제 QGIS
EfficientSAM-Ti CPU inference·모델 없음/정상/손상 lifecycle smoke 통과, 기본 승격 전
locked holdout 대기

범위:

- Apache-2.0 EfficientSAM-Ti encoder/decoder ONNX를 기본 CPU 런타임 후보로 사용합니다.
- 클릭/시드 → SAM의 비보정 관심 영역 score → 색·엣지·방향 guide cost → 세선화/그래프 → A* 또는 최단 경로 순서로 결합합니다.
- SAM 마스크는 최종 폴리곤/라인으로 바로 변환하지 않고 현재 A* 추적의 soft prior로만 사용합니다.
- 최초 한 번만 모델을 다운로드하고, 고정 URL·SHA-256 검증·임시 파일 후 원자적 이동을 사용합니다. 설치 후에는 오프라인으로 작동해야 합니다.

합격 기준:

- 기준 CPU 환경에서 정해진 상한 시간/RAM 안에 한 크롭을 추적합니다.
- M1 기준선 대비 정확도 향상이 사전에 정한 최소치를 넘고, 실패 시 기존 엣지/A* 모드로 안전하게 돌아갑니다.
- 모델 출처·라이선스·해시·런타임 버전이 상태 리포트에 남습니다.

## M3 — Topology and DEM quality control

범위:

- 짧은 간격 연결, 가짜 분기 억제, 중복 선, 자체 교차, 서로 다른 고도의 등고선 교차를 검출합니다.
- 표고점을 한 개씩 제외한 교차 검증으로 DEM 오차를 평가하고, RMSE·MAE·최대 오차를 보고합니다.
- 결측·TIN 범위 밖·낮은 데이터 밀도를 NoData/불확실성 레이어로 별도 표시합니다.
- CRS, 격자 크기, 입력 레이어 해시, 보간 설정, 검증 지표를 머신 판독 JSON 매니페스트로 저장합니다.

합격 기준:

- 치명적 위상 오류는 DEM 실행 전에 차단되고, 교정할 피처로 선택할 수 있습니다.
- 각 DEM에 입력·설정·오차·불확실성을 추적할 수 있는 보고서가 함께 생성됩니다.

## Deliberate non-goals

- 초기 단계에서 cloud upload나 원격 inference를 필수화하지 않습니다.
- SAM 마스크를 고고학적 정답이나 최종 벡터로 취급하지 않습니다.
- 검증 없이 지도 바깥으로 지형을 임의 외삽하지 않습니다.

## Technical references

- [QGIS TIN interpolation](https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/interpolation.html)
- [QGIS GDAL hillshade](https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/gdal/rasteranalysis.html)
- [QGIS background tasks](https://docs.qgis.org/3.44/en/docs/pyqgis_developer_cookbook/tasks.html)
- [EfficientSAM official repository](https://github.com/yformer/EfficientSAM)
- [EfficientSAM CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.html)
- [Intelligent Scissors / Live Wire](https://pubmed.ncbi.nlm.nih.gov/10782619/)
- [clDice topology-preserving loss](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
