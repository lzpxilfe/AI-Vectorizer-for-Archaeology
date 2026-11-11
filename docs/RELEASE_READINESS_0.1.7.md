# ArchaeoTrace 0.1.7 experimental release candidate

후보 준비·로컬 검사: 2026-11-11 · 결과 확인: 2026-11-12

현재 QGIS 제출본과 승인 상태는 [최종 제출 기록](QGIS_SUBMISSION_0.1.7.md)을 따른다.
아래는 첫 동결 후보의 역사 기록이며 최종 업로드 ZIP의 identity가 아니다.

> 이 문서의 ZIP·통과 기록은 **첫 동결 후보**에 해당한다. 이후 사용자 흐름 검토에서
> 수동 클릭·기존 고도·DEM 필드·작은 화면 문제가 추가로 확인되었다. 현재 소스의
> 수정과 후속 설치 후보는 [2026-11-12 재검토](RELEASE_REVIEW_2026-11-12.md)를 따른다.
> 아래 첫 ZIP은 보존용이며 새로 권장하는 설치 파일과 혼동하지 않는다.

공식 QGIS 최신 공개판 `0.1.6`에서 패치 버전을 올린 **미공개 experimental 후보**다.
이 문서는 최종 소스·ZIP·검증을 결속하는 기록이며 QGIS 게시 승인을 뜻하지 않는다.
기존 공개 기록은 [0.1.6](RELEASE_READINESS_0.1.6.md)에 보존한다.

## Decision

- 버전: plugin metadata와 citation을 `0.1.7`로 맞추고 `experimental=True`를 유지한다.
- 공개 상태: 이 후보의 QGIS 업로드·GitHub tag·GitHub Release는 아직 없다.
  `CITATION.cff`에는 공개일을 넣지 않는다.
- 로컬 설치·검토 후보: 통과. 원격 CI 12개 작업도 모두 통과했으며 아래 exact CI에 결속한다.
- 정확도·속도: 사람 검수 Holdout 및 사용자 작업 시간 개선을 주장하지 않는다.
  학습 보조는 기본 OFF이며 기존 Korea 모델을 유지한다.

## Candidate scope

- 학습 등고선 보조·주석·문자 공백 처리와 선택적 경로 진단의 현재 소스를 포함한다.
  Colab 세 seed·ONNX·QGIS 비교와 모델 유지 판단은
  [학습 실행 기록](../learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md)을 따른다.
- 추적 중 Undo: QGIS의 전역 `QAction`이 map-tool 키 처리보다 먼저 실행되어 연장
  대상 선을 삭제할 수 있는 경우를 막는다. 지도 캔버스의 활성 추적에서는 체크포인트를
  되돌리고 첫 앵커에서 멈추며, idle에서는 일반 QGIS 편집 Undo를 사용한다.
- DEM: 창을 열 때 이미 선택된 표고점 레이어의 고도 필드 목록을 동기화한다.
  등고선·선택적 표고점의 숫자형 고도 필드와 저장·좌표계 조건은 계속 검사한다.
- 래스터 읽기: NoData 및 alpha 픽셀을 배경으로 처리한다. 고정 원본 크기가 없는
  WMS/WMTS 계열 provider의 화면 크기 읽기는 bounded Ink fallback으로 다루고,
  native-grid 학습 보조·Recovery의 자격으로 사용하지 않는다.
- O 단어 그룹을 추가 검출하는 별도 실험은 경로 개선이 확인되지 않아 기본 제품에
  채택하지 않는다. 실험 증거와 기존 끝점·옆 선 보호 조건은 보존한다.

## Frozen repository-local candidate

| 항목 | 최종 후보 기록 |
| --- | --- |
| 파일명 | `ai_vectorizer-0.1.7.zip` |
| Source commit | `31518442891d41ae153039d53dabba26c289d0be` |
| ZIP SHA-256 | a3f4eebf07b51febf3a6f9eeddc2213764c9fda407617611f1811bfbae72ba28 |
| 크기 / entries | 2,098,014 bytes / 52 |
| metadata | `0.1.7`, QGIS `3.22–4.99`, `experimental=True` |
| model weights / 원본 지도 | ZIP에 포함하지 않음 |
| exact CI | `552973e36f7847c843f00f33a550ee0c850ec3c5` · [run 43050237514](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43050237514) |

이 ZIP은 동결 후보이며 결정적 builder가 같은 plugin tree에서 같은 bytes를 만든다.
Git에는 source·builder·해시를 보존하고 ZIP은 별도 전달한다. `0.1.5`·`0.1.6`의 과거 동결 후보는 그대로 보존하며 덮어쓰지 않는다.

## Verification

아래 로컬 검사는 위 exact ZIP을 대상으로 완료했다. 원격 CI도 같은 ZIP SHA를 재생성했다. [검증 요약·근거 해시](evidence/release-0.1.7-validation.json)에 결속했다. 원격 CI는 별도다.

| 검사 | 결과·근거 |
| --- | --- |
| 일반 Python suite | 1,246 passed, 180 skipped, 171 subtests passed |
| Python 3.8 source / 무의존성 계약 | 원격 CI compile·무의존성 suite 통과 |
| 실제 QGIS import / runtime safety | QGIS 3.44.8, 13개 모듈 198 tests, 실패·오류·skip 0 |
| 전역 Undo 충돌 회귀 / idle 복귀 | QAction 충돌·연장 시작점·폼 Undo·해제/폐기 후 복귀 통과 |
| DEM 표고점 필드 초기화 | 9 tests; Point/MultiPoint TIN→GeoTIFF→hillshade 포함 |
| raster NoData / ARGB / 원본 크기 없는 provider | 실제 QGIS 9 tests; WMS형 provider cache·Live-Wire·미리보기 포함 |
| 기존 추적·저장·Undo 회귀 | 기존 17+개발 6건, 세 방법 69쌍 저장 좌표 동일; 기존 입력·Undo 보존 |
| 결정적 ZIP / source 일치 / metadata | 생성·check 통과, 실제 plugin-loader 0.1.7 로드·격리 profile 제거 |
| 원격 CI | 12/12 통과: Python 3.11/3.12, CPU 학습 계약, 보안·의존성, Linux/Windows 동일 ZIP, QGIS 3.22.16/3.44.13/4.2.1 |

## CI compatibility follow-up

첫 원격 검사에서 QGIS 3.22의 선택적 SciPy 부재를 테스트가 잘못 실패로 처리했다.
11-12에 테스트를 수정해 SciPy가 있으면 Live-Wire를, 없으면 기본 NumPy 경로를
검사한다. 기본 경로는 모든 호스트에서 강제로 한 번 더 검사하며 테스트 전체를
skip하지 않는다. 수정한 실제 QGIS 9개 검사는 통과했다. 제품 ZIP 바이트는 동일하다. 수정 후 QGIS 3.22를 포함한 원격 CI 전체가 통과했다.

## NoData read cost

실제 QgsRasterBlock의 1000×1000 RGB 입력에서 sentinel NoData 읽기는 약 4.8ms,
bitmap-only 읽기는 약 616ms였다. 네트워크·화면·추적 시간을 제외한 로컬 측정이며
bitmap은 픽셀별 API 검사를 유지한다. 전체 플러그인의 속도 개선 수치로 사용하지 않는다.

## Remaining boundaries

- 재현용 Qt·provider·DEM 테스트의 통과와 사용자가 겪은 실제 프로젝트/WMS 서버의
  문제 해결은 구분한다. 해당 프로젝트·서버의 인증, 응답, 레이어 구성은 아직 직접
  재현하지 않았다.
- 원본 크기가 없는 서비스는 해상도가 고정된 학습용 raster가 아니다. 지원되지 않는
  모델 기능은 기본 Ink/Freehand로 돌아가며 화면·서비스 해상도에 따른 한계가 남는다.
- DEM은 저장된 입력에 의존한다. 희소하거나 잘못된 고도·좌표는 실제 지형과 다른
  결과를 만들 수 있으며, 이 후보가 지형의 정확성을 입증하지 않는다.
- Windows GUI·HiDPI·사용자 프로젝트의 업그레이드 범위는 실제 수행한 검사만 기록한다.
- 공식 공개는 최종 산출물을 사용자가 게시하는 별도 절차다. 공개일·tag·Release를
  검증 완료만으로 추가하지 않는다.
