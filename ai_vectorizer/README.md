# ArchaeoTrace 사용 설명서

[English](README.en.md)

ArchaeoTrace는 스캔 지도에서 등고선을 따라 그리는 QGIS 플러그인입니다.
초록색 경로를 확인하고 클릭한 구간을 선 레이어에 추가합니다.
고도를 입력한 등고선과 표고점으로 DEM과 음영기복도를 만들 수 있습니다.

현재 소스는 미공개 `0.1.7` 실험판 후보입니다. 공식 공개판은 `0.1.6`입니다.
[검토 기록](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/docs/LANGUAGE_REVIEW_2026-11-12.md)과
[변경 내역](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CHANGELOG.md)에서
검증한 파일과 범위를 확인할 수 있습니다.

## 설치와 언어

1. QGIS의 `플러그인 > 플러그인 관리 및 설치 > ZIP에서 설치`를 엽니다.
2. 제공된 ZIP을 선택하고 ArchaeoTrace를 활성화합니다.
3. 패널 상단의 `언어 / Language`에서 한국어 또는 English를 선택합니다.

처음 실행하면 QGIS 언어를 따릅니다. 한국어 이외의 언어에는 영어를 사용하며,
직접 선택한 언어는 다음 실행에도 유지됩니다. 영어 사용 설명서도 ZIP에 포함됩니다.

Freehand와 기본 Ink Centerline에는 추가 패키지나 모델이 필요하지 않습니다.
QGIS의 NumPy를 사용하며, SciPy가 있으면 방향을 고려한 Live-Wire 경로 탐색을
사용합니다. 없으면 가까운 선을 찾는 NumPy 방식으로 동작합니다.

## 지도와 출력 레이어

지리 좌표가 설정된 래스터 지도를 QGIS에 불러오고 입력 지도로 선택합니다.
출력에는 기존 Shapefile·GeoPackage 선 레이어를 선택하거나 새 Shapefile을 만듭니다.
입력 지도와 출력 선 레이어가 모두 준비되어야 추적을 시작할 수 있습니다.
패널이 화면보다 길면 세로로 스크롤합니다.

투명·NoData 픽셀은 배경으로 읽습니다. 원본 픽셀 크기를 제공하지 않는 WMS/WMTS는
현재 화면 해상도를 기준으로 제한된 크기의 Ink 입력을 사용합니다. 이 경우 원본
픽셀에 의존하는 학습 보조·Smart Recovery를 사용할 수 없습니다.
서버 응답이나 배경 처리가 불명확하면 배경이 있는 지리 참조 래스터로 확인하세요.

## 선 그리기와 저장

1. Ink Centerline과 보조 강도를 선택한 뒤 추적을 시작합니다.
2. 등고선 위에 첫 기준점을 놓습니다.
3. 초록색 미리보기를 확인하고 클릭해 다음 구간을 채택합니다.
4. `Enter` 또는 우클릭으로 선을 마치고 고도를 입력합니다.
5. QGIS의 `레이어 편집 저장 / Save Layer Edits`를 눌러 파일에 저장합니다.

Enter와 우클릭은 편집 버퍼에 선을 확정합니다. 파일 저장은 별도입니다.
기존 선의 끝점을 클릭해 이어 그릴 수 있으며, 고도 입력창에는 기존 값이 표시됩니다.
연장을 마친 뒤 한 번의 QGIS Undo로 원래 선과 고도를 복원할 수 있습니다.

보조 강도 `0%`는 커서 좌표를 그대로 사용하고 모델을 실행하지 않습니다.
`100%`는 제안 경로를 사용하며, 중간값은 커서와 경로 좌표를 혼합합니다.
옆 등고선이나 문자로 잘못 이어질 수 있으므로 미리보기를 확인하세요.

| 조작 | 동작 |
| --- | --- |
| 좌클릭 | 기준점 또는 미리보기 구간 채택 |
| Shift/Ctrl을 누른 클릭 | 커서까지 수동 직선 |
| Enter / 우클릭 | 선 확정 및 고도 입력 |
| 추적 중 Ctrl+Z / Cmd+Z / Backspace | 마지막 체크포인트 되돌리기; 첫 점에서 멈춤 |
| 선을 마친 뒤 Ctrl+Z / Cmd+Z | QGIS 편집 작업 되돌리기 |
| Esc / Delete | 현재 추적 취소 |
| 시작점 근처 클릭 | 닫힌 선 만들기 |

## 글자 공백과 선 색상

`G`는 공백을 가로지르는 곡선을 미리 봅니다. 공백 직전의 보이는 선에 기준점을 놓고,
반대편 첫 선 끝에 커서를 옮겨 누르세요. 곡선을 확인한 뒤 같은 끝점을 클릭해야
채택됩니다. `Esc`나 다른 위치 클릭으로 취소할 수 있습니다.
일부 숫자·글자 공백은 자동으로 제안되지만 모든 가림을 복원하지는 못합니다.

Ink 추적 중 `Alt`를 누른 채 사각형의 두 모서리를 클릭하면 문자나 얼룩을 피하도록
힌트를 줄 수 있습니다. `Alt+Backspace`는 마지막 영역을,
`Alt+Shift+Backspace`는 모든 영역을 지웁니다. 통과를 완전히 막는 경계는 아닙니다.

유색 등고선 위에서 `C`를 누르면 그 색상을 기준으로 추적을 보조합니다.
`Shift+C`로 해제합니다. 원본 Byte RGB 지도, 준비된 Ink 증거와 SciPy가 필요합니다.
회색 선·배경·모호한 색상은 거부될 수 있습니다. 같은 색 글자를 구분해 주지는 않습니다.
색상과 회피 영역은 현재 추적에만 적용되며 완료·취소하면 지워집니다.

## DEM과 음영기복도

1. 등고선과 표고점의 편집 내용을 저장합니다.
2. `DEM 생성 / Build DEM`을 엽니다.
3. 등고선 레이어와 실제 고도 필드를 선택합니다.
4. 표고점을 사용할 경우 점 레이어와 해당 고도 필드를 선택합니다. 필요 없으면 None을 고릅니다.
5. 셀 크기와 출력 경로를 확인하고 실행합니다.

`elevation`만 자동 선택합니다. `height_m` 등은 직접 선택하고 ID를 고도로 쓰지
않도록 확인하세요. 등고선·표고점의 고도 단위는 같아야 하며 좌표계도 같아야 합니다.
미터 단위 투영 CRS, 유한한 XY와 숫자 고도, 서로 다른 고도 두 개 이상,
한 직선 위에 있지 않은 정점 세 개 이상이 필요합니다.

선형 TIN 보간으로 DEM을 만든 뒤 GDAL로 음영기복도를 생성합니다.
QGIS Processing의 `qgis:tininterpolation`, `gdal:translate`,
`native:rasterlayerstatistics`, `gdal:hillshade`가 필요합니다.
기본 제한은 2,500만 셀이며 기존 출력 파일을 덮어쓰려면 확인이 필요합니다.
결과는 입력 선의 오류와 밀도에 영향을 받습니다. 원본 지도와 함께 검토하세요.

## 선택 모델

기본 기능으로 충분하면 이 절의 설치는 필요하지 않습니다.

| 기능 | 필요한 구성 |
| --- | --- |
| Smart Recovery | ONNX Runtime과 별도 EfficientSAM-Ti 모델; 기본 꺼짐 |
| 학습 등고선 보조 | SciPy, ONNX, ONNX Runtime과 로컬 모델 폴더; 기본 꺼짐 |
| LSD | OpenCV |
| HED | OpenCV 4.8–4.11과 HED 모델 |
| MobileSAM / SAM ViT-B | OpenCV, PyTorch, 해당 backend와 모델 |
| Legacy Canny | NumPy; OpenCV와 SciPy는 선택 사항 |

Smart Recovery는 약한 Ink 구간을 로컬 EfficientSAM 모델로 검토합니다.
설치·오류·취소 상태에서는 Ink를 유지합니다. `Install Recovery Model` 버튼을
눌렀을 때만 약 39.4 MiB의 모델을 내려받아 크기와 SHA-256을 확인합니다.
손상된 파일의 복구도 사용자가 요청해야 시작합니다.

학습 보조는 `Advanced > Import local model`에서 `model.onnx`와
`manifest.json`이 있는 폴더를 가져온 뒤 켭니다. 원본 픽셀 격자와 북쪽 방향이
유지된 세 밴드 Byte RGB GeoTIFF·PNG·JPEG를 사용합니다.
준비가 늦게 끝나도 이미 표시한 수동 경로나 채택 대기 구간을 바꾸지 않습니다.

HED·MobileSAM·SAM은 Advanced에서 선택하고 모델 상태를 확인한 뒤 필요한 파일을
다운로드합니다. 모델은 플러그인 폴더 밖의 QGIS profile에 저장됩니다.
지도를 외부 추론 서버로 보내지 않습니다.

### 선택 패키지 설치

아래 명령의 `<QGIS_PYTHON>`은 QGIS가 사용하는 Python 실행 파일로 바꾸세요.
새 선택 패키지는 Python 3.11 이상 환경에서 설치합니다. QGIS 3.22/Python 3.8은
기본 ZIP의 소스·무의존성 계약 대상이며 최신 pip 의존성을 새로 설치하는 대상이 아닙니다.
공유 QGIS 환경을 변경하기 전에 pip가 바꿀 패키지를 확인하세요.

```bash
<QGIS_PYTHON> -m pip install scipy scikit-image
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"
<QGIS_PYTHON> -m pip install "onnxruntime>=1.17,<2"
<QGIS_PYTHON> -m pip install "scipy>=1.10" "onnx==1.23.2" "onnxruntime==1.31.0"
```

각 줄은 각각 선택 기능을 위한 명령이며 모두 설치할 필요는 없습니다.
MobileSAM과 SAM의 고정 backend 버전은 다음 파일에 있습니다.

- [MobileSAM requirements](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/requirements-sam-mobile.txt)
- [SAM requirements](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/requirements-sam-full.txt)
- [개발·검증 환경](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CONTRIBUTING.md)

모델 학습은 QGIS와 분리된 Python·Colab 환경에서 수행합니다.

## 학습 주석

Advanced의 `Annotate training crop`으로 512×512 RGB PNG를 엽니다.
등고선, 그 외 잉크·영역, 검수한 영역과 판단할 수 없는 영역을 편집할 수 있습니다.
초기 선은 검수 전 초안입니다.

각 레이어의 편집을 저장한 뒤 `Review / export labels`를 사용합니다.
실제 검수자와 보이는 등고선 전체의 확인이 있어야 단일 검수자 파일럿 승인이 됩니다.
평가용 선은 열린 선 하나를 선택하고 난도를 지정합니다. 내보낸 JSON을 학습에 쓰려면
별도 [학습 도구](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/learning/README.md)에서
새 데이터 매니페스트를 만드세요. 검출 결과가 자동 정답이 되지는 않습니다.

## 문제 해결과 데이터

- 시작 버튼이 꺼져 있으면 입력 지도와 출력 선 레이어를 확인하세요.
- 경로가 옆 선을 따르면 기준점을 더 가깝게 놓거나 보조 강도를 낮추고 수동 구간을 사용하세요.
- 선택 모델을 불러오지 못하면 QGIS Python의 패키지와 모델 파일을 확인하세요.
- DEM 생성이 거부되면 고도 필드, 저장하지 않은 편집, XY 좌표와 CRS를 확인하세요.
- `SAM Status Report`에는 로컬 경로·환경 정보가 들어가며 클립보드에도 복사됩니다.
  공유 전에 내용을 확인하고 민감한 경로·환경값을 지우세요.

사용 기록은 수집하지 않습니다. 네트워크는 선택한 WMS/WMTS 읽기, 요청한 모델
다운로드와 모델 파일이 없을 때의 고정 출처 확인에 사용합니다.
사람이 검수한 정답 기준의 정확도와 작업 시간 단축은 아직 검증하지 못했습니다.
일반 글자 재학습에서 경로 개선이 없어 기존 Korea 모델과 기본 꺼짐 설정을 유지합니다.

[프로젝트](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology) ·
[보안 안내](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/SECURITY.md) ·
[인용 정보](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CITATION.cff)

GNU General Public License v2.0 · [LICENSE](LICENSE)
