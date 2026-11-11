# 0.1.7 사용자 흐름·코드 재검토

2026-11-12 · 미공개 experimental 후보

지도 읽기 → 추적 → 이어 그리기 → 고도 입력 → DEM → 저장 흐름을 검토하고
재현한 문제를 수정했다. 로컬 검증·지도 회귀·원격 CI 12개 작업을 모두 통과해
**0.1.7 experimental 설치·업그레이드 후보로 사용 가능**하다.
공식 QGIS 업로드·tag·GitHub Release는 만들지 않았다.

이번 결과는 편집 의도와 입력값 보존, 화면 접근성에 관한 것이다. 기존 Korea 모델과
학습 보조 기본 OFF는 유지하며, 일반 글자 학습 정확도나 작업 시간 개선을 주장하지 않는다.

## 재현한 문제와 수정

| 작업 | 확인한 문제 | 수정 |
| --- | --- | --- |
| Shift/Ctrl 수동 클릭 | 깊은 U자에서 직선 미리보기 후 259개 점의 곡선이 채택됨 | 클릭·닫기 모두 수동 경로를 채택하고 늦은 Recovery/SAM 결과를 무효화 |
| 기존 등고선 연장 | 125m 선을 연장하고 기본 고도창을 수락하면 0m로 바뀜 | 기존 유한 고도를 초기값으로 사용; 소수 자릿수·범위와 명시적 수정 보존 |
| DEM 사용자 필드 | `id,height_m`에서 ID 1·2를 실제 100·120m 대신 자동 선택 | `elevation` 외에는 직접 선택; 같은 레이어의 명시 선택 보존 |
| DEM 불량 좌표 | NaN XY가 있는 선의 정점 일부가 TIN에 들어가 지형 변경 | 비유한 XY가 있는 등고선·표고점을 생성 전에 거부 |
| 작은 화면·큰 글꼴 | 700px 창에 최소 높이 약 1179px 강제; Linux 영어 문구의 가로 넘침 | 세로 스크롤·줄바꿈·버튼 배치 보완; 기본/Fusion 18px 및 모델 설치 화면 확인 |
| 준비 상태·안내 | 입력 없이 시작 가능, Enter를 파일 저장으로 오해할 문구 | 지도·출력 선 레이어가 있어야 시작; 선 확정과 레이어 편집 저장을 한·영으로 구분 |
| QGIS 4 첫 레이어 선택 | 선택은 정상인데 내부 출력 레이어와 시작 버튼 상태가 갱신되지 않음 | 선택·모델 변경 후 상태 정합화; 같은 추적 유지, 변경·삭제 시 종료, cleanup 뒤 지연 처리 차단 |

앞서 수정한 활성 추적 Ctrl+Z, DEM 표고점 필드 초기화, 투명·NoData 래스터 읽기도
이번 전체 검사에 포함한다. [첫 후보 기록](RELEASE_READINESS_0.1.7.md)은 별도로 보존한다.

## 최종 산출물

| 항목 | 기록 |
| --- | --- |
| 설치 후보 | `ai_vectorizer-0.1.7-review-v3-2026-11-12.zip` |
| 제품 구현 commit | `048605891e661d3c6b01a8876d17341f5cca62ec` |
| 테스트를 포함한 검증 commit | `9c85d8a98e3ce09ff3d9c24db41e1f5c917b0350` |
| SHA-256 | `7829bc9cba216b5105f4a504cae0e61cdfc1bcd630725e20b25cb3969d74b075` |
| 크기 / 파일 수 | 2,107,839 bytes / 52 entries, Python 46 files |
| metadata | `0.1.7`, `experimental=True`, QGIS `3.22–4.99` |
| 공개 상태 | 미공개; model weights·원본 지도를 ZIP에 포함하지 않음 |

원래 동결 ZIP `a3f4eebf07b51febf3a6f9eeddc2213764c9fda407617611f1811bfbae72ba28`과
중간 재검토 파일은 덮어쓰지 않았다. 새 설치에는 위 v3 파일을 사용한다.

## 검증

| 검사 | 결과 |
| --- | --- |
| 일반 Python | 1,246 passed, 202 skipped, 171 subtests passed |
| 실제 QGIS 3.44.8 | 15개 모듈 220 tests, 실패·오류·skip 0 |
| 격리 설치 | 새 ZIP import·실제 plugin loader 7개 시나리오 통과; 임시 profile 제거 |
| 기존 지도 회귀 | 23개 사례 × 세 방법 = 69쌍의 저장 좌표·입력 클릭 동일; 138/138 Undo 통과 |
| UI 크기 | KO/EN × Advanced 접힘/펼침, 1024×700 창·360px dock, 가로 스크롤 0, Start/DEM/Help 접근 가능 |
| 글꼴·추가 설치 화면 | Fusion 18px, HED 모델/런타임 없음, MobileSAM/SAM 설치 상태의 버튼·명령 접근 확인 |
| 결정적 패키지 | build/check와 ZIP 내 모든 파일의 현재 소스 일치 확인 |
| 원격 CI | 12/12 통과: Python 3.11/3.12, Python 3.8 소스 계약, CPU 학습 계약, 보안·의존성, Linux/Windows 동일 ZIP, QGIS 3.22.16/3.44.13/4.2.1 · [run 43058158450](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43058158450) |

[기계 판독 검증 요약](evidence/release-review-2026-11-12-validation.json)에 ZIP identity,
테스트 소스·원시 로그 해시, 사례별 비교 해시, UI 측정과 exact CI를 함께 기록했다.

지도 비교는 첫 동결 후보와 최종 제품을 같은 입력·클릭·CPU 조건으로 비교했다.
`qgis_grid_comparison.py`의 `usgs` 키는 **기존 Korea 모델**, `korea`는 새 일반 글자
모델이다. 각 방법 안에서 제품 변경 전후의 경로 보존을 확인한 결과이며 모델 성능
향상을 뜻하지 않는다. 가려진 곡선과 공식 Calibration/Holdout은 채점하지 않았다.

초기 CI가 잡은 Linux 폭 문제와 QGIS 4 선택 상태 문제는 제품에서 수정했다.
마지막 Qt6 실패는 새 테스트의 정수형 모의 마우스 버튼을 실제 Qt의 `NoButton`
자료형으로 고쳤으며 수정 후 QGIS 4를 포함한 전체 CI가 통과했다.
이 테스트 수정은 ZIP 바이트를 바꾸지 않는다.

## 검토 범위와 한계

- 추적: 미리보기·클릭 일치, 수동/비동기 경로, 연장·고도·Undo, 도구 종료와 레이어 제거.
- 데이터: NoData/alpha·WMS형 provider 경계, DEM 필드·XY·CRS·편집 저장·출력 교체,
  모델 해시·오프라인 검증·캐시. 주석은 사용자 검수와 저장 경계를 유지한다.
- UI: 한국어·English, 빈 프로젝트, 작은 창·확대 글꼴·고급 모델 설치 상태,
  시작·DEM·도움말과 종료·재열기. macOS QGIS 재시작과 스크롤 패널 로딩을 확인했고,
  좁은 창과 하단 접근의 정량 검증은 실제 QGIS Qt 위젯 테스트로 수행했다.
- 실제 제보자의 WMS 서버와 DEM 프로젝트 전체를 재현한 것은 아니다. Windows
  desktop·다양한 HiDPI 화면의 수동 QA와 사람 검수 Holdout 정확도는 미검증 범위다.
- 검토와 회귀 통과는 모든 결함의 부재나 실제 지형의 정확성을 증명하지 않는다.
