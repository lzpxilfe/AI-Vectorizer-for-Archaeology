# 🏛️ ArchaeoTrace

[English](README.en.md)

옛 지형도의 등고선을 사람이 확인하며 벡터로 따고, 그 등고선으로 DEM과
음영기복도(hillshade)까지 만드는 QGIS 플러그인입니다. 모든 처리는 내 컴퓨터에서
이루어지며 지도를 외부 서버로 보내지 않습니다.

![QGIS 3.22-4.99](https://img.shields.io/badge/QGIS-3.22--4.99-3c8c3c.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)
![Experimental](https://img.shields.io/badge/status-experimental-f28c28.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 💡 왜 만들었나

스캔한 고지도에서 등고선을 손으로 따는 일은 느리고 반복적이며 실수하기 쉽습니다.
그렇다고 완전 자동 추출에 맡기면 고도 숫자, 격자, 흐려진 선에서 자주 틀리고, 어디가
틀렸는지 찾는 일이 또 하나의 작업이 됩니다.

ArchaeoTrace는 그 중간을 택합니다. 프로그램이 선을 따라갈 경로를 초록색 미리보기로
제안하면, 사람이 확인하고 클릭해서 채택합니다. **채택한 구간만 결과가 됩니다.**
검출 결과나 AI 마스크가 그대로 저장되는 일은 없습니다.

## ✨ 주요 기능

- **Ink Centerline (기본)** 인쇄된 선의 중심을 여러 굵기에서 찾아 따라가는 보조 추적입니다.
  추가 설치 없이 동작합니다.
- **보조 강도 조절** `0%`(커서 그대로)부터 `100%`(제안 경로 그대로)까지 얼마나 맡길지
  정할 수 있습니다.
- **고지도 특유의 방해 요소 처리** 고도 숫자로 끊긴 등고선 잇기(`G`), 격자선 따라가기
  억제, 글자·얼룩 회피 영역(`Alt`+클릭), 원하는 선 색 지정(`C`)을 지원합니다.
- **선택형 AI 보조 (기본 OFF)** Ink가 불확실한 구간에만 EfficientSAM-Ti를 쓰는
  Smart Recovery와, 로컬 ONNX 모델을 쓰는 학습 등고선 보조가 있습니다. 실패하면 항상
  Ink 경로로 돌아갑니다.
- **지형 복원** 등고선 고도와 표고점을 입력하면 선형 TIN DEM과 hillshade를 만듭니다.
- **QGIS 편집 흐름 그대로** edit buffer, 한 번의 Undo, `Save Layer Edits`를 따릅니다.
- **한국어·English UI**

## ⚙️ 작동 방식

```text
스캔 지도 → 선 증거 계산 → 경로 제안(초록 미리보기) → 사람이 클릭해 채택
          → QGIS 레이어 저장 → 고도 입력 → DEM · hillshade
```

선택형 AI 보조는 "제안 경로"를 더 낫게 만드는 데만 쓰이고, 안전 검사를 통과하지 못하면
버려집니다. 구조와 모듈별 책임은 [기능과 아키텍처](docs/FEATURES_AND_ARCHITECTURE.md)에
정리했습니다.

## 📦 설치

1. QGIS에서 `플러그인 > 플러그인 관리 및 설치`를 열고, 설정에서 **실험적 플러그인
   표시**를 켭니다.
2. `ArchaeoTrace`를 검색해 설치합니다. 배포 ZIP이 있다면 `ZIP에서 설치`를 써도 됩니다.
3. 기본 기능(Freehand, Ink Centerline)은 이것으로 끝입니다. AI 보조나 OpenCV 기반
   방식이 필요할 때만 [사용자 가이드](ai_vectorizer/README.md)의 선택 설치를 따르세요.

## 🚀 빠르게 시작하기

1. 좌표가 잡힌(georeferenced) 지도 래스터와 저장할 선 레이어를 고릅니다.
2. `Ink Centerline`과 보조 강도를 고릅니다.
3. 등고선 위를 클릭해 나가며 초록 미리보기를 확인하고 채택합니다. `Enter`나 우클릭으로
   선을 마칩니다.
4. 고도를 입력하고 `Save Layer Edits`로 저장합니다.
5. `Step 4 > DEM 생성…`에서 DEM과 hillshade를 만듭니다.

단축키와 문제 해결은 [사용자 가이드](ai_vectorizer/README.md)에 있습니다.

## 🌍 사용 사례

- [Geoawesome, "Digitising Historical Maps: Testing AI-Assisted Contour Tracing in QGIS"](https://geoawesome.com/digitising-historical-maps-testing-ai-assisted-contour-tracing-in-qgis/)
  (2026-10) 폴란드의 소규모 모래 채굴지를 대상으로 1929-39년 WIG 1:25,000 지도와
  1960-80년대 지형도를 디지타이징해 채굴 전·중·후의 지형을 복원하고 현재 LiDAR DEM과
  비교한 외부 사용 사례입니다.

연구나 실무에서 사용하셨다면 이슈로 알려 주세요. 이 목록에 추가하겠습니다.

## 🧪 Colab 실험에서 확인한 것

세 seed 학습·ONNX 검증·실제 QGIS 비교까지 완료했습니다. 일반 글자용 새 모델은
17개 개발 진단 중 양성 13건의 저장 경로를 개선하지 못해 기존 Korea 모델을 유지합니다.
사람 승인 주석은 0개이며 공식 Calibration/Holdout 평가는 남아 있습니다.

- [학습·QGIS 비교 기록](learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md): 조건·해시·17개 사례와 모델 유지 판단.
- [실패 진단](learning/GENERAL_TEXT_FAILURE_DIAGNOSIS_2026-10-19.md): 개발용 6쌍과 선택적 로그로 O의 단어 그룹 검출 누락을 확인. 검출 보완과 회귀 검증이 다음 작업입니다.
- [실제 지도 시범](docs/REAL_MAP_PILOT_2026-10-19.md): 학습 모델을 사용하지 않은 Ink 커널로 열린 선 3구간을 저장했습니다. 고도·지리 좌표가 없는 작은 초안입니다.

## 🚧 현재 상태와 한계

- 실험 단계 플러그인입니다. 공식 QGIS 저장소의 최신판은 `0.1.6`(2026-09-09 공개)이고,
  현재 소스는 **미공개 `0.1.7` experimental 후보**입니다. 추적 중 Undo, 수동 클릭과
  미리보기 일치, 연장 시 기존 고도 유지, DEM 입력 검증과 작은 화면 UI를 보완합니다.
  학습 보조는 기본 OFF입니다. [사용 흐름 재검토](docs/RELEASE_REVIEW_2026-11-12.md)도 기록합니다.
  [변경 내역](CHANGELOG.md)과 [후보 검증 기록](docs/RELEASE_READINESS_0.1.7.md)을 확인하세요.
- 사람이 검수한 정답 기준의 정확도 평가와 작업 시간 단축 효과는 **아직 검증 중**입니다.
  진행 상황은 [benchmarks](benchmarks/README.md)와 [learning](learning/README.md)에
  기록합니다.
- 만들어진 DEM은 지형의 정답이 아니라 **검토할 가설**입니다. 희소하거나 잘못된 등고선에
  민감하며, 위상 검사와 불확실성 레이어는 아직 없습니다.
- 자동으로 제안된 선은 반드시 눈으로 확인하세요.

## 🔒 데이터와 개인정보

지도, 추적한 선, AI 추론 결과는 외부로 전송되지 않으며 사용 기록 수집도 없습니다.
네트워크는 사용자가 선택한 WMS/WMTS 지도를 읽거나 AI 모델을 내려받을 때 사용합니다.
받은 모델은 고정된 크기와 SHA-256으로 검증합니다. 자세한 내용은 [SECURITY](SECURITY.md)에 있습니다.

## 📚 더 읽을거리

| 문서 | 내용 |
| --- | --- |
| [사용자 가이드](ai_vectorizer/README.md) | 설치 옵션, 모드별 사용법, 단축키, 문제 해결 |
| [기능과 아키텍처](docs/FEATURES_AND_ARCHITECTURE.md) | 구현 흐름, 안전 경계, 모듈 구성 |
| [로드맵](ROADMAP.md) | 다음 단계와 의도적으로 하지 않는 것 |
| [학습 파일럿](learning/README.md) | 학습 등고선 보조의 데이터·학습·평가와 실험 기록 |
| [벤치마크](benchmarks/README.md) | 재현 가능한 평가 도구와 증거 형식 |
| [기여 안내](CONTRIBUTING.md) | 개발 환경, 테스트, 릴리스 절차 |

## 🤝 인용, 기여, 라이선스

연구·교육·실무에 사용하셨다면 [CITATION.cff](CITATION.cff)로 인용해 주세요.
버그 제보, 번역, 문서, 재배포 가능한 샘플 지도 기여를 환영합니다. 시작은
[CONTRIBUTING](CONTRIBUTING.md)에서 하시면 됩니다.

GNU General Public License v2.0 · [LICENSE](LICENSE)
