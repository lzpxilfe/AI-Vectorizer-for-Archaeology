# ArchaeoTrace open-source development plan

기준일: 2026-08-23

현재 코드의 기능과 설계는 [`FEATURES_AND_ARCHITECTURE.md`](FEATURES_AND_ARCHITECTURE.md),
`0.1.5` 후보의 실행 증거와 공개 차단 조건은
[`RELEASE_READINESS_0.1.5.md`](RELEASE_READINESS_0.1.5.md)에 기록합니다.

## Why this project exists

ArchaeoTrace는 고지도에서 등고선을 복원하는 사람이 계정, 원격 추론 서비스나
특정 기관의 인프라 없이도 QGIS 안에서 작업하고 그 과정을 검토·재현할 수 있게 하는
오픈소스 도구입니다.

```text
로컬 추적 → 사용자의 경로 검수 → 고도·표고점 저장
         → 위상·표고 QA → DEM/hillshade → 불확실성·재현 기록
```

현재 `0.1.5` 후보에는 로컬 추적, QGIS 편집 버퍼, 고도·표고점 저장과 실험적
TIN DEM/hillshade까지 구현되어 있습니다. 위상 QA, uncertainty/NoData,
DEM provenance sidecar는 아직 구현되지 않았으므로 현재 기능처럼 소개하지 않습니다.

## Principles

1. **Human-reviewed by construction** — 자동 결과는 사용자가 채택·수정하는 보조
   신호이며 고고학적 사실로 자동 확정하지 않습니다.
2. **Local-first and inspectable** — 지도와 추론은 로컬에서 처리합니다. 네트워크가
   필요한 모델 관리 작업은 사용자가 명시적으로 시작하고, 출처와 해시를 확인합니다.
3. **Safe QGIS editing** — provider에 우회 저장하지 않고 edit buffer, constraint와
   Undo 계약을 존중합니다.
4. **Reproducible evidence** — 기능 주장은 코드, 입력, 환경과 결과를 다시 확인할 수
   있는 테스트나 benchmark 증거와 연결합니다.
5. **Graceful baseline** — 선택 모델이나 선택 패키지가 없어도 Freehand와 기본 Ink
   경로를 유지합니다.
6. **Community benefit before feature count** — 새 모델 수보다 설치 성공, 이해 가능한
   실패, 데이터 보존, 문서와 재현 가능한 예제를 먼저 개선합니다.

## North-star outcome

목표는 자동 vertex 수가 아니라 **검수가 끝난 고도 등고선과 검토 가능한 지형
산출물까지 안전하게 완료하는 것**입니다. 실제 데이터 평가에서는 다음을 함께
기록합니다.

- 작업 완료 시간, 모델 대기와 fallback
- 클릭, 수동 vertex, Undo와 잘못된 분기 수정 횟수
- centerline F1/Dice, 평균·95% 선간 거리, 끊김과 과잉·누락 분기
- 자체 교차, 다른 고도 등고선 교차, 중복과 비정상 고도 간격
- 입력 고도 오류, 보류 피처와 비보정 보조 신호
- spot-height 검증 DEM MAE/RMSE, NoData와 외삽 면적
- CPU, 최대 RAM, network 사용 여부와 실패율

합성 fixture 결과는 이 목록의 제품 품질 증거로 사용하지 않습니다.

## Development gates

### Gate A — Trustworthy `0.1.5` distribution

완료 조건:

- metadata, citation, 문서, release tree와 ZIP의 버전이 하나로 일치합니다.
- 같은 소스가 Linux와 Windows에서 byte-identical ZIP을 만들고, QGIS matrix가
  그 ZIP 자체를 import해 edit·Undo·unload 회귀를 실행합니다.
- Python 3.8은 source compile과 추가 의존성 없는 계약을, Python 3.10/3.12는
  전체 의존성 테스트와 현재 resolve된 requirements 감사를 통과합니다.
- 실제 QGIS에서 새 피처, 기존 피처 연장, 한번의 Undo, DEM/hillshade와 안전한
  overwrite 거부를 확인합니다.
- HED/SAM asset은 출처, 정확한 byte 수와 SHA-256, staged write, 원자 게시와
  rollback 계약을 통과합니다.
- `experimental=True`를 유지하고, stable 전환은 같은 후보 commit의 전체 원격
  CI와 clean-profile GUI 확인 뒤 별도 결정합니다.
- ZIP SHA-256, commit, CI run, 검증 범위와 남은 위험을 한 릴리스 기록에 묶습니다.

공식 QGIS plugin 저장소의 `0.1.4` 다음 배포판을 `0.1.5`로 정리합니다. Git
history의 `0.1.5–0.1.7`과 미공개 worktree의 `0.1.8`은 QGIS 저장소에 게시되지 않은
개발 metadata였으며 이력은 rewrite하지 않습니다. 그 개발판을 ZIP이나 source로
직접 설치한 사용자는 QGIS가 낮은 번호를 자동 update로 보지 않을 수 있으므로 기존
plugin을 제거하고 검증된 `0.1.5` ZIP을 다시 설치해야 합니다.

### Gate B — Real historical-map benchmark

필요 산출물:

- 재배포 권리를 확인한 실제 역사 지도 crop 30–50개
- 지도 종류, 해상도, 선 굵기, 변색, 글자·기호 교차와 단절을 포함한 strata
- 두 명의 검수자가 합의한 ordered contour centerline과 고도 기준선
- Manual QGIS와 공개적으로 실행 가능한 추적 도구에 동일한 작업 protocol
- 원시 event log, 환경 manifest, JSON/CSV 결과와 실패 사례 gallery

합격 기준은 실행 전에 고정합니다. 기본 Ink가 수작업을 줄이면서 centerline·위상
오류 상한을 넘지 않아야 합니다. 어느 방법이든 유리한 사례와 실패 사례를 함께
공개하고, 재현되지 않은 성능 표현은 README나 metadata에 넣지 않습니다.

### Gate C — Archaeology QA and safer storage

1. 서로 다른 고도 contour의 교차, 자체 교차, 중복, dangling endpoint와 급격한
   고도 간격 변화를 검출하고 수정할 피처를 선택합니다.
2. 숫자 라벨 후보와 인접 contour 문맥으로 고도를 제안하되 자동 확정하지 않고
   비보정 score와 근거를 표시합니다.
3. 새 출력 기본값을 Shapefile에서 GeoPackage로 옮기고 기존 vector layer 직접
   편집은 유지합니다.
4. QA 결과와 수정 전·후 통계를 프로젝트에 저장합니다.

### Gate D — Reproducible terrain hypothesis

- 입력 raster/vector, 모델 artifact, 파라미터, CRS, QGIS/GDAL 버전을 DEM sidecar에
  기록합니다.
- spot height leave-one-out와 contour holdout 검증을 분리합니다.
- 낮은 데이터 밀도, convex hull 밖 영역과 큰 contour 간격을 uncertainty/NoData로
  출력합니다.
- 치명적 QA 오류가 있으면 DEM을 차단하고 수정할 피처로 이동합니다.

### Gate E — Community usability

- 공식 stable QGIS 릴리스와 사용자 중심 changelog
- 작은 공개 sample project, 화면이 포함된 한국어·영어 quick start
- 운영체제별 clean-profile 설치와 첫 contour 완료 절차
- 재현 정보를 담되 민감한 local path와 환경값을 지우는 bug-report 안내
- 실제 사용자의 시작 성공률, 첫 contour 완료 단계와 설치·모델 실패 지점 관찰

기본 telemetry는 추가하지 않습니다. 진단 자료도 사용자가 내용을 검토하고
명시적으로 공유하는 흐름으로 유지합니다.

## Architecture work packages

현재 동작을 테스트로 고정한 뒤 다음 순서로 나눕니다.

1. `smart_trace_tool.py`의 session 상태, QGIS event, 경로 계산과 편집 transaction을
   `TraceSession`, `TraceController`, pure kernel, `LayerEditService`로 분리합니다.
2. `main_dialog.py`는 widget wiring을 남기고 model 설치·검증, preview raster와
   output 생성을 service로 옮깁니다.
3. model download, raster 읽기, HED/SAM 준비와 prompt inference를 취소 가능한
   background task로 옮기고 generation token으로 최신 결과만 UI에 게시합니다.
4. HED, SAM, EfficientSAM의 artifact 코드를 하나의 `ModelArtifactStore` 계약으로
   수렴시킵니다.
5. DEM과 vector output에 공통 `OutputGuard`를 적용해 loaded target, sidecar,
   path alias와 rollback을 한곳에서 처리합니다.
6. 제품 경로에서 쓰이지 않는 `core/path_finder.py`와 `core/vectorizer.py`는 외부
   import를 확인한 뒤 제거하거나 명시적 adapter로 바꿉니다.

완료 기준은 파일 길이가 아니라 상태 전이입니다. 취소, model switch, layer removal,
edit failure와 unload 뒤 늦은 task callback이 UI나 edit buffer를 다시 변경하지 않아야
합니다.

## Explicit non-goals before Gate C

- 계정이나 필수 cloud inference 인프라
- 범용 건물·도로·필지 polygon 자동완성
- 더 큰 foundation model의 기본 설치
- 실제 데이터 없이 합성 fixture 점수를 제품 정확도라고 표현하는 것
- 보조 경로나 DEM을 고고학적 사실로 자동 확정하는 것

새 기능은 외부 도구에 있다는 이유만으로 시작하지 않습니다. 사용자 작업의 완결성,
검수 오류, 데이터 안전이나 재현성을 실제로 개선하고 실패 시 안전한 fallback을
제시할 수 있을 때 기본 흐름에 넣습니다.
