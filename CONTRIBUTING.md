# Contributing to ArchaeoTrace

bug report, code, 문서, 번역과 재배포 가능한 sample map 기여를 환영합니다.
ArchaeoTrace는 연구·현장 자료를 안전하게 다루는 QGIS plugin이므로 기능 수보다
재현성, 실패 시 data 보존과 사람이 검수할 수 있는 흐름을 우선합니다.

## Start with the project boundaries

- 실제 기능과 module map: [`docs/FEATURES_AND_ARCHITECTURE.md`](docs/FEATURES_AND_ARCHITECTURE.md)
- 다음 단계와 비목표: [`ROADMAP.md`](ROADMAP.md)
- 안전 경계와 공개 gate: [`docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md`](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md)
- 현재 검증 증거: [`docs/RELEASE_READINESS_0.1.5.md`](docs/RELEASE_READINESS_0.1.5.md)

계획 기능을 이미 구현된 것처럼 문서화하지 마세요. 합성 fixture 결과를 실제 역사
지도 정확도나 사용자 작업 속도 근거로 사용하지 않습니다.

## Development setup

Python 3.10+의 격리 환경을 권장합니다. 실제 plugin은 QGIS Python에서 실행되므로
system Python에 설치한 package가 QGIS에 보이지 않을 수 있습니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Windows에서는 `.venv\Scripts\python.exe`를 사용하세요. 선택 SAM backend를 직접
실행할 때만 `requirements-sam-mobile.txt` 또는 `requirements-sam-full.txt`를
추가합니다. QGIS가 공유하는 Python 환경을 변경하기 전에 profile과 환경을
백업하고 pip의 변경 계획을 확인하세요.

## Test tiers

변경 위험에 맞는 가장 작은 test부터 실행한 뒤 범위를 넓힙니다.

1. pure kernel·문서·packaging 변경

   ```bash
   python -m pytest -q
   python scripts/package_release.py
   python scripts/package_release.py --check
   ```

2. Python 3.8 source 계약

   Python 3.8은 EOL이라 선택 dependency stack을 새로 설치하는 대상이 아닙니다.
   모든 배포·test module의 compile과 CI에 정의된 no-dependency suite만 확인합니다.

   ```bash
   python3.8 -m compileall -q ai_vectorizer benchmarks scripts tests \
     package_plugin.py litmus_sam_status.py
   ```

3. QGIS 편집, UI lifecycle 또는 DEM 변경

   결정적 ZIP을 새 임시 경로에 풀고 실제 QGIS Python으로
   `scripts/qgis_import_smoke.py`와 `tests/test_qgis_runtime_safety.py`를 실행합니다.
   새 feature, 기존 contour 연장, 한 번의 Undo, layer removal, unload와 실제
   TIN→GeoTIFF→hillshade를 변경 범위에 맞게 확인합니다.

4. model 또는 dependency 변경

   source URL, license, immutable commit, 정확한 size·SHA-256, offline 재검증,
   symlink 거부, atomic publication과 rollback test를 함께 갱신합니다. versioned
   requirement graph는 CI의 `pip-audit --strict` 범위를 확인합니다.

## Code ownership by area

- tracing math: `ai_vectorizer/core/trace_kernel.py`, `livewire.py`,
  `sam_trace_kernel.py`
- detector and model path: `edge_detector.py`, `sam_engine.py`, `model_store.py`
- QGIS interaction and edits: `ai_vectorizer/tools/smart_trace_tool.py`
- UI wiring: `ai_vectorizer/ui/main_dialog.py`, `dem_dialog.py`
- terrain: `ai_vectorizer/core/dem_spec.py`, `dem_pipeline.py`
- benchmark evidence: `benchmarks/`
- release: `scripts/package_release.py`, `tests/test_release_tooling.py`

QGIS object는 main thread에서만 변경하세요. background result에는 generation/session
identity를 결속하고 취소, model switch, layer removal과 unload 뒤 늦은 callback이
상태를 바꾸지 못하게 test를 추가하세요.

## Version discipline

`ai_vectorizer/metadata.txt`가 배포 버전의 source of truth입니다.

- 일반 branch와 pull request에서는 metadata 숫자를 올리지 않고
  [`CHANGELOG.md`](CHANGELOG.md)의 `Unreleased`에 사용자 변경을 기록합니다.
- release 준비를 시작할 때 공식 QGIS plugin 저장소의 최신 공개판을 기준으로 다음
  버전을 한 번 정합니다.
- 그 한 변경에서 metadata, `CITATION.cff`, changelog heading, 문서와 artifact 이름을
  동기화합니다. CI의 consistency test를 임의로 우회하지 않습니다.
- 공개일은 실제로 같은 artifact를 게시할 때만 `CITATION.cff`와 release record에
  추가합니다.
- 과거 commit과 tag를 rewrite하거나 개발판 흔적을 공개판처럼 설명하지 않습니다.
- package unit test에는 제품 버전 대신 `9.8.7` 같은 독립 fixture를 사용합니다.

## Documentation and translations

- root `README.md`는 project overview, `README.en.md`는 English overview,
  `ai_vectorizer/README.md`는 ZIP에 포함되는 offline user guide입니다.
- 설치된 ZIP은 root docs를 포함하지 않습니다. package README에서는 repository의
  파일을 가리킬 때 GitHub absolute link를 사용하세요.
- UI label, 한국어·English 설명과 shortcut이 달라지면 두 언어를 함께 갱신합니다.
- `faster`, `accurate`, `safe` 같은 표현에는 비교 조건, test와 남은 한계를 붙입니다.

## Map and benchmark data

실제 map sample에는 출처, 저작권·재배포 license, crop 생성 절차, georeferencing과
개인정보 여부를 기록해야 합니다. 권리가 불명확한 raster를 저장소에 추가하지
마세요. 기준선은 검수자, 합의 절차와 coordinate semantics를 기록하고 raw evidence와
실패 사례를 함께 보존합니다.

## Bug reports and diagnostics

가능하면 OS, QGIS/Python/plugin version, 재현 단계, expected/actual behavior와 data
손상 여부를 적어 주세요. 최소 재현 project에는 민감하거나 재배포할 수 없는 원본을
넣지 마세요.

`SAM Status Report`는 working directory, `QGIS_PREFIX_PATH`, `PYTHONPATH`, model
path와 system 정보를 포함할 수 있고 clipboard에도 복사됩니다. 첨부 전에 반드시
읽고 local username, project path, network mount와 token이 든 환경값을 지우세요.
취약점이나 data-loss 경로는 public issue 대신 [`SECURITY.md`](SECURITY.md)의 비공개
방법을 사용하세요.

## Pull request checklist

- 변경한 동작과 의도하지 않은 동작을 설명했습니다.
- 관련 pure/QGIS test와 실패 회귀를 추가했습니다.
- 사용자 문서와 `CHANGELOG.md`를 갱신했습니다.
- 실제 데이터·model·dependency의 출처와 license를 확인했습니다.
- 생성 ZIP과 source가 일치하고 package check가 통과합니다.
- 성능이나 정확도 표현에 재현 가능한 evidence와 limitation이 있습니다.
