# ArchaeoTrace

[English](README.en.md)

스캔한 옛 지형도에서 등고선을 따고, 입력한 고도로 DEM과 음영기복도를 만드는
QGIS 플러그인입니다. 초록색 미리보기를 확인하며 구간을 선택하고, 필요한 곳은
수동으로 그릴 수 있습니다. 지도와 추적 결과는 내 컴퓨터에서 처리합니다.

QGIS 3.22–4.99 · 실험판 · GPL v2

## 설치와 언어

QGIS의 `플러그인 > 플러그인 관리 및 설치`에서 실험적 플러그인 표시를 켜고
ArchaeoTrace를 검색합니다. 후보 ZIP은 `ZIP에서 설치`로 설치할 수 있습니다.
현재 승인된 최신 버전은 `0.1.6`입니다. `0.1.7`은 Qt6 검사를 마친 업로드용
ZIP을 준비한 상태이며, 앞서 제출한 버전은 삭제했습니다.
[업로드용 ZIP 검증](docs/QT6_UPLOAD_0.1.7.md) ·
[보안 알림과 예외 목록 검토](docs/SECURITY_SCAN_0.1.7.md)

패널 상단의 `언어 / Language`에서 `한국어` 또는 `English`를 선택합니다.
처음에는 QGIS 언어를 따르며, 한국어 외의 언어에서는 영어를 사용합니다.
직접 선택한 언어는 다음 실행에도 유지됩니다.

기본 Freehand와 Ink Centerline은 별도 패키지 설치 없이 사용할 수 있습니다.
선택 모델의 설치와 사용법은 [한국어 사용 설명서](ai_vectorizer/README.md)와
[English user guide](ai_vectorizer/README.en.md)에 있습니다. 두 문서는 ZIP에도 포함됩니다.

## 등고선 그리기

1. 지리 좌표가 설정된 지도 래스터와 출력 선 레이어를 선택합니다.
2. Ink Centerline과 보조 강도를 정하고 추적을 시작합니다.
3. 초록색 경로를 확인한 뒤 클릭해 구간을 채택합니다.
4. `Enter` 또는 우클릭으로 선을 마치고 고도를 입력합니다.
5. QGIS의 `레이어 편집 저장 / Save Layer Edits`로 파일에 저장합니다.

기존 선의 끝점에서 이어 그릴 수 있습니다. 추적 중 `Ctrl+Z`는 마지막
체크포인트를 되돌리며, 선을 마친 뒤에는 QGIS 편집 작업을 되돌립니다.
`Shift` 또는 `Ctrl`을 누른 클릭은 커서까지 직선을 그립니다.

## 기능

| 기능 | 용도 |
| --- | --- |
| Ink Centerline | 인쇄된 선을 따라 경로 제안 |
| Freehand와 보조 강도 | 수동 입력 또는 커서와 제안 경로의 비율 조절 |
| 공백 연결 `G` | 끊긴 등고선 사이의 곡선을 확인한 뒤 채택 |
| 선 색상 `C` · 회피 영역 `Alt`+클릭 | 원하는 잉크색 지정, 글자·얼룩 회피 |
| Smart Recovery · 학습 등고선 보조 | 로컬 모델을 사용하는 선택 기능; 기본 꺼짐 |
| DEM과 hillshade | 등고선 및 표고점으로 선형 TIN 지형 생성 |

DEM 창에서는 실제 고도 필드를 확인하세요. `elevation` 외의 이름은 직접
선택해야 합니다. 입력 편집을 저장하고 미터 단위 투영 좌표계를 사용해야 합니다.

## 사용 사례

[Geoawesome의 사용 후기](https://geoawesome.com/digitising-historical-maps-testing-ai-assisted-contour-tracing-in-qgis/)에서는
폴란드의 모래 채굴지를 대상으로 1929–39년 WIG 지도와 1960–80년대 지형도를
디지타이징하고, 복원 지형을 현재 LiDAR DEM과 비교했습니다(2026년 10월).
외부 사용자의 작업 사례이며 플러그인의 일반적인 정확도 평가와는 구분합니다.

## 검증과 남은 과제

0.1.7 검토에서는 이어 그리기 Undo, 수동 미리보기·저장 일치, 기존 고도 보존,
DEM 입력과 작은 화면, QGIS 4의 레이어 선택 상태를 수정했습니다.
[검토 기록](docs/RELEASE_REVIEW_2026-11-12.md)에 설치 후보와 실행 검사 범위를 기록합니다.

Colab에서는 세 seed 학습과 ONNX 검증, 같은 조건의 QGIS 비교를 마쳤습니다.
일반 글자용 새 모델은 17개 진단 중 양성 13건의 저장 경로를 개선하지 못해
기존 Korea 모델을 유지합니다. 후속 단어 그룹 검출 보완도 경로 개선이 없어
채택하지 않았습니다.

- [학습 및 모델 비교](learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md)
- [일반 글자 실패 진단](learning/GENERAL_TEXT_FAILURE_DIAGNOSIS_2026-10-19.md) · [후속 검출 실험](learning/WORD_GROUP_EXPERIMENT_2026-11-11.md)
- [실제 지도 시범](docs/REAL_MAP_PILOT_2026-10-19.md): 고도·지리 좌표를 부여하기 전의 열린 선 세 구간

사람이 검수한 정답 기준의 정확도와 작업 시간 단축은 아직 검증하지 못했습니다.
생성한 DEM은 입력 등고선에 민감하므로 원본 지도와 함께 검토해야 합니다.
위상 검사와 불확실성 레이어는 아직 제공하지 않습니다.

## 데이터와 개발

지도와 추적 결과를 원격 추론 서버로 보내지 않으며 사용 기록을 수집하지 않습니다.
네트워크는 선택한 WMS/WMTS 지도를 읽거나 요청한 모델을 내려받을 때 사용합니다.
모델은 정해진 크기와 SHA-256으로 검증합니다. [보안 안내](SECURITY.md)

[구조](docs/FEATURES_AND_ARCHITECTURE.md) · [로드맵](ROADMAP.md) ·
[벤치마크](benchmarks/README.md) · [학습 기록](learning/README.md) ·
[기여 안내](CONTRIBUTING.md) · [인용 정보](CITATION.cff) · [라이선스](LICENSE)
