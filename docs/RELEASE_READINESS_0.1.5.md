# ArchaeoTrace 0.1.5 release-readiness review

기준일: 2026-08-23
검토 기준 commit: `9cf2bd7ba119f03cb4771c6fcc5bd796d8800e8a` 위의 `0.1.5` 후보 worktree
최종 로컬 산출물: `dist/ai_vectorizer-0.1.5.zip`

## Decision

판정은 두 단계로 나눈다.

- **Experimental 0.1.5 후보: 조건부 GO.** 현재 로컬 산출물은 Python
  3.8/3.10/3.12, macOS QGIS 3.44.8 실제 API/runtime, QGIS upstream upload validator
  script와 결정적 packaging 검증을 통과했다. 다음 단계는 이 worktree를 하나의
  commit으로 고정해 원격 CI를 실행하는 것이다. validator 로컬 통과는 QGIS 저장소
  업로드나 공식 승인을 뜻하지 않는다.
- **Stable 공개 전환: NO-GO.** 원격 Linux/Windows/QGIS 3.22/3.44/4.2 행렬, clean
  profile Windows/macOS GUI, `0.1.4 → 0.1.5` upgrade가 아직 실행되지 않았다.
  `experimental=True`를 유지하며, 실제 역사 지도 benchmark 전에는 제품 정확도나
  속도 우위를 주장하지 않는다.

커밋, push, tag, GitHub Release, QGIS 저장소 upload는 이 검토에서 수행하지 않았다.
공식 QGIS plugin 저장소에는 검토일 현재 experimental `0.1.4`가 남아 있었다. 작업
시작 당시 원격 `main` metadata는 `0.1.7`이었고, 이 worktree에는 미공개 `0.1.8`
정리가 있었다. Git history의 `0.1.5–0.1.7`과 이 `0.1.8`은 QGIS 저장소에 게시되지
않은 개발 metadata였으므로 공식 `0.1.4` 다음 후보를 `0.1.5`로 정규화했다. 이력은
rewrite하지 않는다.

## Final artifact identity

| 항목 | 값 |
| --- | --- |
| 파일 | `dist/ai_vectorizer-0.1.5.zip` |
| SHA-256 | `d2925198dc2192bbb7eebe579bb48207c860179a94d4216df77e746d0451789a` |
| 크기 | 1,464,897 bytes |
| ZIP entries | 29 |
| 재빌드 | 같은 source에서 2회 byte-identical |
| metadata | `0.1.5`, QGIS `3.22–4.99`, `experimental=True` |
| 모델 weight/native binary | 포함하지 않음 |

ZIP은 1980-01-01 timestamp, 정렬된 entry, 고정 파일 mode와 저장 압축을 사용한다.
저장 압축은 zlib 구현 차이 없이 Linux/Windows 바이트 동일성을 만들기 위한 선택이다.
업로드 상한은 QGIS 공개 지침의 decimal 20 MB를 적용한다.

## Confirmed findings fixed

### P0 — data loss

- 로드된 Shapefile과 같은 경로를 새 출력으로 선택하면 QGIS writer가 `NoError`를
  반환하면서 디스크 파일을 0 feature로 교체했고, 이후 process exit 139까지
  재현됐다. provider URI, percent/file URL, canonical path, case와 symlink alias를
  비교해 로드된 대상에는 writer를 호출하지 않도록 막았다.

### P1 — edit, model and terrain integrity

- 새 feature/기존 contour 연장, elevation field·값 변경을 하나의 QGIS edit command로
  묶었다. hard field constraint나 add/update 실패 시 schema와 geometry를 함께
  되돌리고, 저장은 QGIS edit buffer에 남겨 한 번의 Undo가 동작한다.
- SAM 다운로드와 legacy migration은 기존 checkpoint 백업, staged SHA-256/크기
  확인, 원자 교체, 게시 경로 재검증, 실패 시 원본 복원을 거친다. model directory와
  최종 파일 symlink도 거부한다.
- HED의 잘못된 package prototxt를 고정 upstream artifact로 교체했다. prototxt와
  caffemodel을 한 쌍으로 검증·forward-load한 뒤 게시하며, 부분 실패 시 두 파일을
  함께 복원한다. legacy migration도 게시 후 다시 검증한다.
- DEM/hillshade 게시 전 source, destination, sidecar 집합과 inode/size/mtime을
  snapshot한다. 처리 중 경로가 생기거나 바뀌면 게시를 거부하고 기존 결과를
  복원한다. 비동기 처리 도중 target이 QGIS에 로드되는 race도 최종 게시 직전에
  다시 차단한다.
- raster provider는 알려진 dtype의 byte budget을 호출 전에 확인한다. payload는
  64 MiB, 25M pixels 이하로 제한하고 Int64/UInt64를 잘못 Float64로 해석하지 않는다.
- trace 누적 cost overflow, Live-Wire NaN 전파, NaN/비정수/과대 설정, SAM mask와
  EfficientSAM 입력의 allocation 전 dimension 우회를 fail-closed 처리했다.

### P2 — lifecycle, benchmark and supply chain

- map tool, 여섯 rubber band, timers, preview temporary directory, owned spot-height
  layer와 toolbar를 layer removal/deactivate/dialog close/plugin unload에 맞춰 해제한다.
  stale CRS와 교체된 output source도 활성 trace를 중단시킨다.
- benchmark worker request는 1 MiB+1만 읽고, 검증 뒤 artifact가 나타난 경우 atomic
  no-replace publication으로 기존 파일을 덮지 않는다. generator/worker warm-up 상한도
  일치시켰다.
- metadata의 bare `%`가 QGIS `ConfigParser`를 깨뜨리던 문제를 `%%`로 고쳤고, 실제
  parse 결과는 다시 `0-100%`가 된다. icon은 픽셀 동일한 lossless 최적화로 1 MB
  resource 제한 아래로 내렸다.
- requirements를 base/OpenCV/SAM-common/commit-pinned backend/dev로 분리했다.
  Python 3.8은 no-pip source 계약만 유지하고, 보안 유지 가능한 선택 stack은 Python
  3.10+로 명시했다.
- release builder는 symlink/junction/reparse point, hidden residue, model weight,
  native binary와 path escape를 거부하며 임시 ZIP을 fsync한 뒤 원자 게시한다.
- GitHub Actions와 QGIS image는 digest로 고정했다. QGIS 행렬이 재빌드한 ZIP은 Linux
  release job의 SHA-256과 같아야만 테스트를 진행한다.

## Verification evidence

| 검증 | 결과 |
| --- | --- |
| Python 3.10.20 fresh env, full pytest | `323 passed, 22 skipped, 86 subtests` |
| Python 3.12.13 fresh env, full pytest | `323 passed, 22 skipped, 86 subtests` |
| Python 3.8.20 no-dependency CI contract | `210 passed, 2 skipped` (총 212 tests); 전체 compile 통과 |
| Extracted final ZIP, QGIS 3.44.8-Solothurn | package 경로 import 확인, runtime `20/20` 통과 |
| 실제 terrain path | QGIS TIN → GeoTIFF → GDAL hillshade 생성 및 resource 해제 통과 |
| Runtime benchmark smoke | Canny/LSD generate → validate → evaluate 통과; 두 방법 local eligible, publication ineligible |
| Runtime benchmark manifest SHA-256 | `f1d281a00c28a89a0c77b639958a5ca7bd04457838e7f1c54b1ac3cd806ee282` |
| QGIS upstream upload validator script | commit `4f9b451658c3599c721a6da3fb20d33706580e81`에서 package/version/URL 로컬 검증 통과; 업로드·공식 승인 아님 |
| `pip-audit==2.10.1` | Python 3.12/macOS ARM64에서 uv-compiled base/OpenCV/dev/SAM-common graph를 `--strict --disable-pip --no-deps`로 감사: 9/10/15/25 packages, known vulnerability 0 |
| Bandit 1.9.4 | 배포·benchmark 38 files/19,531 LOC, medium/high finding 0; low 24는 별도 검토 대상 |
| Ruff 0.16.4 fatal rules | Python 64 files, `E9,F63,F7,F82` 통과 |
| Zizmor 1.29.0 | offline/pedantic/strict collection finding 0 |
| Workflow/YAML/diff | YAML parse, `git diff --check` 통과 |

이번 로컬 환경의 직접 `pip-audit --strict -r` 수집기는 uv-managed Python의 내부
`ensurepip` SIGABRT로 끝나 취약점 판정 증거로 사용하지 않았다. 대신 각 requirements를
같은 Python/플랫폼에서 완전 resolve한 뒤 위 표의 strict no-deps 감사를 실행했다.
이는 현재 macOS ARM64/Python 3.12 resolve와 PyPI advisory DB 범위이며 타 플랫폼,
package hash·license, Git backend, native library나 공급망 전체의 안전을 보증하지
않는다. 원격 CI는 별도 환경에서 직접 strict 수집기를 다시 실행하도록 구성되어 있다.

실제 asset 경로도 별도로 확인했다.

- MobileSAM `40,728,226` bytes,
  SHA-256 `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`:
  다운로드·게시 후 네트워크 차단 상태의 `up_to_date` 확인.
- HED caffemodel `58,876,104` bytes,
  SHA-256 `4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7`:
  다운로드·게시 및 OpenCV 4.11 Caffe forward 확인.
- EfficientSAM split encoder/decoder: content-addressed download·SHA 검증 확인. 실제
  ONNX inference smoke와 PyTorch MobileSAM/SAM inference는 최종 행렬에 포함되지 않았다.

합성 runtime fixture의 F1=1은 wiring 계약만 증명한다. 역사 지도 품질 근거가 아니며
report의 `publication_eligible`도 의도적으로 `false`다.

## Architecture review

방향은 맞지만 분리가 절반만 끝났다.

- pure kernel인 `trace_kernel.py`, `livewire.py`, `sam_trace_kernel.py`, `dem_spec.py`는
  QGIS 밖에서 경계·결정성·실패를 검증할 수 있다. 이번 결함 대부분을 여기서
  재현하고 회귀로 고정할 수 있었다.
- 반면 `smart_trace_tool.py` 약 3.1K lines, `main_dialog.py` 약 2.0K lines,
  `edge_detector.py` 약 1.5K lines가 QGIS lifecycle, UI, IO, 모델과 계산을 함께
  가진다. 변경 한 번의 회귀 범위가 너무 크다.
- HED, SAM, EfficientSAM에는 비슷하지만 서로 다른 artifact publication 코드가
  세 벌 있다. 하나의 `ModelArtifactStore`와 공통 `OutputGuard`로 수렴해야 한다.
- `core/path_finder.py`, `core/vectorizer.py`는 제품 경로에서 사용되지 않는다.
  외부 import 호환성을 확인한 뒤 삭제하거나 명시적 adapter로 바꿔야 한다.
- ad-hoc Pyright는 dependency-aware pure/benchmark 범위에서도 44 diagnostics를
  남긴다. 대부분 runtime validator 뒤의 narrowing과 dynamic OpenCV/NumPy API지만,
  typed CI로 올리기 전에 정리해야 한다. 비-QGIS source coverage는 QGIS 모듈을 0으로
  포함해 73%였으며, GUI/state transition은 실제 QGIS 회귀가 보완한다.

구체적인 분리 순서와 완료 조건은
[`OPEN_SOURCE_DEVELOPMENT_PLAN.md`](OPEN_SOURCE_DEVELOPMENT_PLAN.md)의 architecture work
packages에 둔다.

## Residual risk register

| 우선도 | 남은 위험 | release 처리 |
| --- | --- | --- |
| P1 | 이 worktree의 원격 CI가 아직 실행되지 않음 | experimental/stable 모두 upload 전에 필수 |
| P1 | QGIS 3.22/3.44 Linux, QGIS 4.2, Windows package/runtime를 로컬에서 재현하지 못함 | digest-pinned remote matrix 필수; Windows/macOS clean-profile 수동 smoke 추가 |
| P1 | 실제 MobileSAM/SAM PyTorch 및 EfficientSAM ONNX inference가 최종 환경에서 미실행 | 모델을 기본값으로 승격하거나 성능 주장 전에 pinned smoke 필수 |
| P1 | 실제 역사 지도와 독립 기준선 benchmark가 없음 | 정확도·속도 주장 금지; Gate B 선행 |
| P1 | sparse/wrong contour TIN이 그럴듯한 오답을 만들 수 있고 uncertainty/provenance가 아직 없음 | `experimental terrain hypothesis` 유지; Gate C/D 전 연구 결론에 자동 사용 금지 |
| P2 | 모델 download/HED/SAM inference와 일부 raster 준비가 UI thread에서 길게 실행될 수 있음 | cancellable background task와 generation token으로 이동 |
| P2 | DEM 입력 layer/provider 자체는 전체 task 동안 immutable snapshot이 아님 | 입력 export/hash manifest 후 processing 실행 |
| P2 | 새 vector 기본 출력이 Shapefile이며 생성 자체는 transactional dataset publish가 아님 | GeoPackage 기본값과 공통 OutputGuard 구현 |
| P2 | model/output parent ancestor와 benchmark input/output parent의 hostile swap까지 dirfd chain으로 고정하지 않음 | local same-user attacker를 threat model에 포함하면 openat 계층 적용 |
| P2 | benchmark custom worker stdout/stderr capture가 무제한이고 수동 import는 self-attested | bounded log와 provenance tier/signature 추가 |
| P2 | historical smoothing v1이 endpoint를 옮김 | manifest v1 유지, endpoint-preserving-v2 별도 비교 |
| P2 | open dependency range, Git backend, QGIS/native library는 현재 `pip-audit` 범위 밖 | lock/SBOM, commit source/license/dependency review, native scan |
| P2 | 진단 JSON을 공유하면 local path/system 정보가 노출될 수 있음 | explicit redaction/data-flow 문서화 |
| P2 | `GPLv2` 표기와 일부 source header의 `version 2 or later`, 저자 표기가 완전히 정렬되지 않음 | stable 전에 권리자 확인 후 SPDX/headers 통일 |
| P3 | 세 대형 monolith와 두 unused module, 44 type diagnostics | release 이후 state/service 분리와 typed CI 도입 |

## Stable release checklist

### Artifact identity

- [x] source → release tree → ZIP 동기화
- [x] 두 번의 로컬 build가 byte-identical
- [x] 최종 ZIP을 별도 경로에서 실제 QGIS import/runtime 검증
- [x] QGIS upstream upload validator script 로컬 통과
- [ ] 하나의 release commit/tag와 원격 CI run URL에 SHA-256 결속
- [ ] Linux = Windows = QGIS matrix SHA-256 확인
- [ ] SBOM과 서명/attestation 적용 또는 미적용 사유 기록

### Runtime and upgrade

- [x] Python 3.8 source/no-dependency, Python 3.10/3.12 full suite
- [x] macOS QGIS 3.44.8 offscreen API/runtime 및 실제 DEM pipeline
- [ ] digest-pinned QGIS 3.22/3.44/4.2 remote jobs green
- [ ] Windows/macOS clean profile GUI와 HiDPI/한국어·영어 smoke
- [ ] 공식 `0.1.4`에서 `0.1.5`로 upgrade/model migration/rollback smoke
- [ ] 개발판 `0.1.6–0.1.8` 제거 후 `0.1.5` 수동 재설치·model 보존 smoke

### Product and scientific evidence

- [x] RC와 terrain을 experimental로 표시
- [x] 합성 fixture를 제품 accuracy evidence로 사용하지 않음
- [ ] 재배포 권리를 확인한 30–50개 실제 역사 지도 crop
- [ ] 사전 고정 protocol, 두 검수자 기준선, raw event/hardware/repetition 기록
- [ ] topology QA, uncertainty/NoData, DEM provenance sidecar

따라서 다음 안전한 동작은 **commit을 고정하고 원격 CI를 실행하는 것**이다. 그
결과 없이 `experimental=False`, tag, upload 또는 성능 우위 문구를 진행하지 않는다.
