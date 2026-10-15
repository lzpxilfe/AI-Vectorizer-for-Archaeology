# 🏛️ ArchaeoTrace

로컬 컴퓨터에서 고지도 등고선을 사람이 검수하며 벡터화하고, 고도선으로 DEM과
hillshade까지 만드는 QGIS 플러그인입니다. 지도와 추적 결과를 원격 추론 서버로
보내지 않습니다.

![QGIS release metadata 3.22+](https://img.shields.io/badge/QGIS_release_metadata-3.22%2B-3c8c3c.svg)
![Source Python 3.8+](https://img.shields.io/badge/source_Python-3.8%2B-3776ab.svg)
![Experimental](https://img.shields.io/badge/status-experimental-f28c28.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 🚧 Current Source Status

- 공식 QGIS 저장소에 experimental `0.1.5`가 2026-08-26 공개됐습니다.
  현재 checkout과 설치 후보는 metadata `0.1.6`, `experimental=True`이며 아래
  Ink v2·Smart Recovery와 Unreleased 학습 등고선 보조·주석 도구를 포함합니다.
  `0.1.6`은 아직 QGIS 저장소나 GitHub
  Release에 게시되지 않은 검증 후보입니다.
- 공식 `0.1.5` artifact와 Ink v1 fallback은 보존합니다. 현재 `0.1.6` UI와 사용법은
  이 문서가 source of truth이며, 정확한 후보 ZIP·commit·CI 범위는 repository의
  `docs/RELEASE_READINESS_0.1.6.md`에 기록합니다.
- QGIS `3.22–4.99`와 Python `3.8+` source 호환성을 대상으로 합니다. 로컬 검증은
  Python 3.8/3.10/3.12와 macOS QGIS 3.44.8에서 수행했습니다. 원격 CI의 QGIS
  3.22.16/3.44.13/4.2.1 package import·runtime safety와 Linux/Windows 결정적 ZIP도
  통과했습니다.
- 기본 UI는 `Freehand`, 다중 스케일 `Ink Centerline`, 기본 OFF의
  `Smart Recovery (Experimental)`입니다. LSD, HED, MobileSAM, SAM (ViT-B),
  Legacy Canny와 기존 model index 0–5는 접힌 Advanced 영역에 보존됩니다.
- Smart Recovery의 EfficientSAM-Ti ONNX는 Ink가 약한 구간의 soft corridor로만
  사용하며 model 미설치·오류·취소 시 Ink 경로를 그대로 유지합니다.
- `학습 등고선 보조 / Learned contour guide`는 Advanced의 기본 OFF 실험 기능입니다.
  로컬 ONNX 모델을 사용하며, 준비된 USGS 16개 도엽·144개 크롭은 아직 사람이 검수할
  초안입니다. 실제 지도 학습·Holdout 평가는 완료되지 않았고, 합성 자료 검증용
  모델은 등고선 학습 모델이나 실제 지도 정확도 근거가 아닙니다.
  별도 기계 라벨 실험 모델은 `human_review_required=true`로 기록되며 사람 검수
  Calibration/Holdout 평가와 합격 모델 승격에 사용할 수 없습니다.
- 지도와 추적 처리는 로컬에서 수행됩니다. model 다운로드는 network를 사용합니다.
  SAM Check/Status는 유효한 local checkpoint가 있으면 size와 SHA-256을 offline
  확인하고, 파일이 없을 때만 고정 source의 availability를 조회합니다.
- 과거 미공개 개발판 `0.1.7–0.1.8`을 설치했다면 QGIS가 `0.1.6`을 자동 update로
  보지 않을 수 있습니다. 기존 plugin을 제거한 뒤 `0.1.6` ZIP을 설치하세요.
  profile에 저장된 검증 model은 plugin 폴더 밖에 유지됩니다.

## 🎯 What You Can Do

- ✏️ `Freehand` 모드로 순수 수동 디지타이징
- 🖋️ 9·15·31 source-pixel과 RGB/명도에서 만든 연속 선 증거를 기준점마다 한 번
  계산하는 방향 인식 Live-Wire
- 🛟 검증된 local EfficientSAM으로 저신뢰 Ink 구간만 보완하고 안전하게 나아진
  challenger만 채택하는 Smart Recovery
- 🧠 로컬 ONNX의 등고선 지지·문자/도로 회피 점수로 Ink 경로를 보조하는 실험 기능
- 📝 512픽셀 학습 크롭의 등고선·비등고선·검수·ignore 주석을 QGIS에서 편집하고 내보내기
- 🎚️ `0%` 정확한 커서부터 `100%` 완전 보조 경로까지 실제 좌표 비율로 혼합
- 👁️ 클릭 한 번으로 채택될 정확한 경로를 같은 초록색 선으로 실시간 표시
- 🎨 `C`로 지정하고 `Shift+C`로 해제하는 현재 추적 전용 선 색상 보조
- 🪜 실제 숫자 공백 양쪽을 사용자가 지정해 검수하는 `⤴ 라벨 공백 연결` preview
- 👁️ `Ink Centerline`/`LSD`/`HED`/`Legacy Canny` 검출 미리보기와 SAM 계열의 인터랙티브 초록색 경로 미리보기
- ⛰️ 등고선 고도값 입력 및 `Spot Heights` 포인트 저장
- 🏔️ 고도 등고선을 선형 TIN `DEM`/`hillshade` GeoTIFF로 변환
- 📄 `Verify Selected SAM Model` / `SAM Status Report`로 모델 상태 점검
- 🌏 한국어 / English UI 지원

## 🧠 Tracing Modes

| UI option | 역할 | 필요한 런타임 | 상태 |
| --- | --- | --- | --- |
| `✏️ Freehand` | 사용자가 직접 선을 입력 | QGIS `NumPy`; 추가 pip·model 없음 | baseline |
| `🖋 Ink Centerline` | 다중 스케일·색상 연속 증거와 방향 인식 Live-Wire | QGIS `NumPy`; `SciPy`/`scikit-image`는 선택 최적화 | 기본 방식 |
| `🛟 Smart Recovery` | 저신뢰 Ink 구간의 EfficientSAM corridor challenger | `onnxruntime>=1.17,<2` + 약 39.4 MiB split ONNX | 실험적·기본 OFF |
| `Learned contour guide` | 로컬 모델의 등고선 지지·비등고선 회피 점수로 Ink 보조 | `SciPy` + `onnx` + `onnxruntime` + 로컬 검증 bundle | Advanced·실험적·기본 OFF |
| `📐 LSD` | 선분 검출 결과를 Live-Wire에 결합 | `OpenCV`; `SciPy`는 Live-Wire 선택 최적화 | Advanced / Legacy |
| `🧠 HED` | 학습된 edge map으로 추적 보조 | `OpenCV 4.8–4.11` + 약 `56.1 MiB` model; `SciPy` 선택 | Advanced / Legacy |
| `🎯 MobileSAM` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `MobileSAM` + 약 `38.8 MiB` weights | Advanced / Legacy |
| `🧩 SAM (ViT-B)` | point-prompt mask와 edge/A*를 결합 | `OpenCV` + `PyTorch` + `segment_anything` + 약 `357.7 MiB` checkpoint | Advanced / Legacy |
| `🔧 Legacy Canny` | 기존 그래디언트 경계 검출과 Live-Wire | `NumPy`; `OpenCV`/`SciPy` 선택 | Advanced / Legacy |

> `0%`는 edge 감지와 model 실행을 건너뛰고 cursor를 그대로 사용합니다. `1~99%`는
> 같은 완전 보조 경로와 cursor 경로를 좌표 비율로 혼합하고, `100%`는 전체 보조
> 경로입니다. 탐색 창을 최근 anchor 주변으로 제한해도 인접 contour, 글자나 기호로
> 오추적할 수 있습니다. 초록색 경로를 확인하고 anchor나 Freehand로 교정하세요.
> 실제 역사 지도 기준 dataset 전에는 model 간 정확도 순위를 주장하지 않습니다.

> Smart Recovery는 SAM mask를 최종 선으로 쓰거나 Ink와 이진 OR하지 않습니다. 시작·끝점,
> 기존 탐색창·우회 한계, 강한 Ink 보존, 평행선·분기 전환과 개선량을 검사합니다.
> 실제 재배포 가능 고지도 holdout 결과 전에는 기본 승격이나 정확도 우위를 주장하지
> 않습니다. 모델 파일은 플러그인 ZIP에 포함되지 않습니다.

원본 Byte Ink에서는 긴 가로·세로 검정/저채도 선이 각각 2개 이상 보일 때 격자
후보를 자동 검출합니다. 격자를 따라가는 비용은 높이고, 등고선의 횡단은 허용합니다.
검출은 증거 준비와 함께 백그라운드에서 실행되며 실패하면 기존 Ink를 유지합니다.
짧거나 끊긴 격자·큰 회전·채도 높은 파란 격자는 놓칠 수 있고, 두꺼운 격자의
가장자리를 따라갈 수도 있습니다. 화면의 캐시 문맥에 따라 판정이 달라질 수 있습니다.

기존 Ink를 우선 보존하는 학습 경로 보호와 관측 잉크 주변으로 학습 지지를 제한하는
후보는 실제 사례의 퇴행 때문에 기본 적용하지 않습니다. 학습 보조는 계속 기본 OFF이며,
옆 등고선으로 넘어가는 문제는 해결되지 않았습니다. 초록색 선을 검수하세요.
숫자 공백은 `Label-gap bridge`에서 지정한 두 끝점을 곡선 미리보기로 연결합니다.

경로 비용 선택 기록이 없는 모델은 `등고선 지지만 사용 · 문자 회피 설정 없음`으로
표시하며, 학습된 문자·도로 회피 비용을 적용하지 않습니다. 실제 등고선을 문자로
오판한 비용을 줄이기 위한 변경입니다. 평가 해시에 결속된 선택 비용이 있는 모델은
그 설정을 사용합니다. 수동 회피와 선 색상 지정은 유지됩니다. 유용한 학습 회피도
빠질 수 있으며, 모든 모델의 정확도가 좋아진다는 뜻은 아닙니다.

### 🧠 Experimental: 학습 등고선 보조

1. `Advanced: 기존 추적 모델 / Advanced: legacy tracing models`를 펼치고
   `로컬 모델 가져오기 / Import local model`을 누릅니다.
2. `model.onnx`와 `manifest.json`이 함께 있는 로컬 폴더를 선택합니다. 파일 해시,
   모델 계약과 출력 형식을 확인한 뒤 QGIS profile의 별도 모델 cache에 복사합니다.
   모델을 자동으로 내려받지 않습니다.
3. `학습 등고선 보조 (실험적, 기본 꺼짐) / Learned contour guide (Experimental,
   default OFF)`을 켜고 `0%`보다 큰 보조 강도로 Ink 추적을 시작합니다.
4. `지도 준비 대기`, `로컬 추론 중`, `다음 앵커에서 사용`, `적용 중` 상태를 확인하고,
   초록색 미리보기를 검수한 다음 클릭해 채택합니다. 모델이 없거나 읽기·런타임·출력
   검증에 실패하면 `Ink 사용 / using Ink` 상태로 기존 추적을 계속합니다.

이 보조에는 SciPy와 ONNX 런타임, 북쪽이 위인 로컬 **3밴드 Byte RGB**
GeoTIFF·PNG·JPEG가 필요합니다. 원본 좌표에 고정한 512픽셀 타일의 중앙 384픽셀을
사용해 화면 이동에도 같은 원본 픽셀의 문맥을 유지합니다. 지원하지 않는 래스터는
Ink를 사용합니다. `0%`에서는 학습 추론도 실행하지 않습니다.

학습 회피 점수는 수동 회피 영역·선 색상 힌트와 픽셀별 최댓값으로 합쳐 Ink와
Smart Recovery에 전달합니다. 늦게 도착한 결과는 이미 보이는 미리보기, 대기 중인
클릭이나 수동 제안을 바꾸지 않고 다음 앵커에서 사용합니다. 기존과 같이 저장은
QGIS 편집 버퍼로 들어가며, 한 번의 Undo로 되돌릴 수 있습니다.

### 📝 학습 크롭 주석 편집

1. Advanced에서 `학습 크롭 주석 편집 / Annotate training crop`을 누르고 준비된
   **512×512 Byte RGB PNG 크롭**과 주석 작업 폴더를 선택합니다.
2. `contours`의 Ink 초안을 수정해 검수할 영역의 **모든 보이는 등고선 중심선**을
   그립니다. 문자·도로 등 비등고선은 `other_ink` 선 또는 `other_ink_regions` 영역에
   표시합니다. 검수한 영역은 `reviewed`, 판단 불가능하거나 가려진 부분은 `ignore`
   폴리곤에 그립니다. 실제로 지워진 선 공백을 임의로 이어 정답으로 만들지 마세요.
   `reviewed` 밖과 `ignore`는 학습 손실에서 제외됩니다.
3. 각 주석 레이어에서 `Save Layer Edits`로 변경을 저장합니다. Calibration/Holdout용
   크롭은 평가할 **열린 등고선 하나**를 명시적으로 선택하고, 그 양 끝점이 320픽셀
   Live-Wire 창에 들어가도록 구간을 정합니다.
4. `주석 검수·내보내기 / Review / export labels`에서 새 JSON 경로와 실제 검수자
   식별자를 입력합니다. 모든 보이는 등고선을 확인했을 때만 `Yes`를 선택합니다.
   `No`는 미승인 초안으로 내보냅니다. 선택한 평가선에는 일반/고난도 여부도 지정합니다.
   한 사람의 검수는 `single_reviewer_pilot`이며 독립 검수·공식 성능 검증을 뜻하지 않습니다.
5. 저장소의 별도 `learning` CLI에서 내보낸 주석을 새 버전의 데이터 매니페스트에
   연결합니다. ZIP에는 학습 코드·Colab 노트북이 포함되지 않습니다.
   [데이터 연결·Colab 학습·평가 안내](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/learning/README.md)를
   따라 144개 크롭의 주석·분할을 동결한 뒤 학습하세요.

### 🎨 Experimental: 선 색상 지정

Ink 추적을 시작하고 원하는 유색 등고선 위에 커서를 둔 뒤 `C`를 누르세요.
`선 색상 지정` 버튼은 마지막 지도 커서 위치를 사용하며 선택한 RGB를 상태에
표시합니다. 회색·배경이나 교차점 등 여러 색이 가까워 모호한 표본은 거부하므로
뚜렷한 유색 선에서 다시 시도하세요. 준비된 native Byte RGB Ink v2 지도, SciPy와
`0%`보다 큰 assist가 필요합니다. `Shift+C` 또는 `색상 해제`로 제거합니다.

다른 색의 Ink에 유한한 soft cost만 더하고, 회피 영역과 픽셀별 최댓값으로 합쳐
Ink와 Recovery에 같은 guidance를 전달합니다. 같은 색의 글자·기호는 구분하지
못하므로 초록 preview를 계속 검수하세요. pan 뒤 새 cache에도 선택한 색을 적용하며
추적 종료·취소·reset에서 지웁니다. 저장·업로드하지 않으며 실제 지도 정확도나
작업시간 개선이 검증된 기능은 아닙니다.

### 🪜 Experimental: label-gap bridge

검은·갈색 등고선이 숫자로 **실제로 끊긴** 경우에만, 공백 전 anchor를 확정하고 반대편
선에 커서를 둔 뒤 `G`를 누르세요. 초록 preview를 검토한 다음 그 끝점을 클릭해야
채택됩니다. `Esc`나 다른 위치 클릭은 proposal만 취소하고 hover는 유지합니다.

3px의 local 방향이 숫자 획에 가려지면 두 명시적 endpoint에서 공백 **바깥쪽** 12px
중심선만 다시 검사합니다. 양쪽의 가까운 지지와 방향 정렬이 모두 있어야 제한된
Hermite preview를 만듭니다. 대상선이 없거나 문자·원격 평행선만 보이고, 방향이 퍼지거나
3–128px 범위를 벗어나면 bridge를 만들지 않고 기존 Ink를 유지합니다. OCR, model,
자동 endpoint 선택은 사용하지 않습니다. 합성 예시 검증 단계이므로 지도에서 직접
검수하세요.

## 📦 Installation

### 1. Install the plugin

1. QGIS에서 `Plugins > Manage and Install Plugins > Install from ZIP`을 엽니다.
2. 배포 ZIP을 선택해 설치합니다.
3. QGIS를 재시작한 뒤 `ArchaeoTrace`를 활성화합니다.

### 2. Install only the dependencies you need

`Freehand`와 기본 `Ink Centerline`은 QGIS ZIP 설치 직후 추가 pip 없이 동작합니다.
`SciPy`가 없으면 모든 non-SAM edge mode(Ink/LSD/HED/Canny)는 방향 인식 Live-Wire
대신 제한 창의 NumPy nearby-edge snap을 사용합니다. 아래 선택 패키지는
시스템 Python이 아니라 QGIS가 사용하는 Python에 설치해야 합니다.
OpenCV는 검증한 4.8–4.11 범위로 제한하지만 이 제한만으로 QGIS의 공유
NumPy ABI가 고정되지는 않습니다. 설치 전에 pip의 변경 계획을 확인하세요.

QGIS 3.22/Python 3.8은 기본 ZIP 경로의 source·무의존성 계약 대상입니다. `0.1.6`
후보는 QGIS 3.22.16/3.44.13/4.2.1에서 exact package import·runtime-safety CI를
통과해야 하며 결과는 repository의 release-readiness 기록에 결속합니다. 이 검증은
이미 게시된 공식 0.1.5 ZIP을 소급해 증명하지 않습니다. Python 3.8은 EOL이며 최신 보안
수정이 적용된 Pillow/pytest
의존성 계열을 설치할 수 없습니다. 이 릴리스의 선택적 SciPy/scikit-image,
OpenCV, SAM pip 스택과 `requirements-dev.txt`는 보안 유지 대상 Python
3.10+ 환경에서 사용하세요. 가능하면 최신 지원 QGIS/Python을 사용하고,
공유 QGIS 환경을 변경하기 전에 프로필과 환경을 백업하세요.

```bash
# LSD / HED / SAM용 OpenCV (Ink Centerline과 Legacy Canny에는 불필요)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"

# Optional: non-SAM edge mode의 방향 인식 Live-Wire와 더 빠른 morphology/thinning
<QGIS_PYTHON> -m pip install scipy scikit-image

# Optional: Smart Recovery local CPU runtime (plugin이 자동 실행하지 않음)
<QGIS_PYTHON> -m pip install "onnxruntime>=1.17,<2"

# Optional: learned contour guide graph validation + single-thread CPU inference
# Python 3.12에서 함께 검증한 버전; QGIS Python 3.10+ 선택 환경에 설치
<QGIS_PYTHON> -m pip install "scipy>=1.10" "onnx==1.23.2" "onnxruntime==1.31.0"

# Optional: MobileSAM/SAM download and update checks
<QGIS_PYTHON> -m pip install requests

# MobileSAM (immutable upstream commit)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"

# SAM (ViT-B, immutable upstream commit)
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
```

저장소 checkout의 격리된 Python 3.10+ 개발 환경에서는 기본
`requirements.txt`, Smart Recovery용 `requirements-smart-recovery.txt`, OpenCV용
`requirements-opencv.txt`, 학습 보조용 `requirements-learned-guide.txt`, 백엔드별
`requirements-sam-mobile.txt` 또는 `requirements-sam-full.txt`를 사용할 수
있습니다. 학습은 QGIS와 분리한 Python 3.12+·Colab 환경의
`requirements-learning.txt`를 사용합니다. `requirements-dev.txt`는 Python 3.10+
테스트 전용입니다. QGIS에 PyTorch·SMP를 설치할 필요는 없습니다.

macOS QGIS.app 예시:

```bash
"/Applications/QGIS.app/Contents/MacOS/python3.12" -m pip install "opencv-python-headless>=4.8,<4.12"
```

### 3. Prepare optional model weights

- `Learned contour guide`: Advanced의 `Import local model`로 검증 매니페스트와
  단일 FP32 ONNX가 들어 있는 폴더를 가져옵니다. SHA-256과 모델 계약을 확인하고
  CPU ONNX Runtime을 한 스레드로 준비합니다. 자동 다운로드나 원격 추론은 없습니다.
- `Smart Recovery`: 먼저 위 ONNX Runtime을 QGIS Python에 설치합니다. QGIS를 다시
  시작한 뒤 `Smart Recovery (Experimental)`을 켜고 `Install Recovery Model`을
  명시적으로 누릅니다. encoder 24,799,761 bytes와 decoder 16,565,728 bytes를 고정
  URL에서 받아 SHA-256을 검증한 후에만 활성화됩니다. 자동 다운로드나 지도 업로드는
  없습니다. 파일 검증과 ONNX session 준비는 background task에서 진행되어 그동안도
  Ink tracing을 사용할 수 있습니다. Recovery는 native Byte raster에서만 실행하고
  그 밖의 정수형에서는 Ink v2를 유지합니다. 상태는 `Ink`, `Recovering`, `Enhanced`,
  `Ink fallback`으로 표시됩니다. 검증된 regular model 파일이 손상됐으면 버튼이
  `Repair Recovery Model`로 바뀌며, 명시적으로 눌렀을 때만 기존 파일을 격리하고
  다시 받습니다. 실패·취소 시 미완료 파일은 격리본을 복원하고, 이미 hash 검증된
  replacement는 유지하며 symlink·directory·Windows junction/reparse point는 변경하지
  않습니다.
- `HED`: `Step 3`에서 HED를 선택한 뒤 `Download HED`를 사용합니다.
- `MobileSAM` / `SAM`: model을 선택하고 `Verify Selected SAM Model`로 local 상태를
  확인한 뒤 필요한 경우 해당 `Download` 버튼을 사용합니다. local checkpoint가
  없을 때만 고정 source availability 조회가 일어납니다.
- MobileSAM/SAM 진단이 필요하면 `SAM Status Report`를 생성합니다. report에는 현재
  작업 경로, QGIS/Python 환경값과 model 경로가 포함될 수 있고 clipboard에도
  복사됩니다. 공유 전에 내용을 검토하고 민감한 local path·환경값을 지우세요.

<details>
<summary>📥 Manual model paths</summary>

브라우저로 직접 받아야 하는 경우 아래 파일과 경로를 사용하면 됩니다.

- `HED` network definition: `https://raw.githubusercontent.com/s9xie/hed/912632b986acc6dd6cc33b95603b2f279d7bd9f2/examples/hed/deploy.prototxt`
- `HED` weights: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- `MobileSAM` weights: `https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt`
- `SAM (ViT-B)` weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

HED 수동 다운로드도 정확히 검증해야 합니다: definition은 8,186 bytes / SHA-256
`378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a`, weights는
58,876,104 bytes / SHA-256 `4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7`입니다.

복사 경로:

- `HED` definition: `<QGIS_PROFILE>/ai_vectorizer/models/hed_deploy.prototxt`
- `HED` weights: `<QGIS_PROFILE>/ai_vectorizer/models/hed_pretrained_bsds.caffemodel`
- `MobileSAM`: `<QGIS_PROFILE>/ai_vectorizer/models/mobile_sam.pt`
- `SAM (ViT-B)`: `<QGIS_PROFILE>/ai_vectorizer/models/sam_vit_b_01ec64.pth`

HED and SAM assets are stored outside the plugin directory so reinstalling or
upgrading the ZIP does not require downloading them again. Exact verified
plugin-local HED assets from an older install are migrated automatically on
first use.

</details>

## 🗺️ Quick Workflow

1. 래스터 지도를 선택합니다.
2. 출력 라인 레이어를 새로 만들거나 기존 SHP를 고릅니다.
3. 원하는 모델을 선택합니다.
4. 기본 Ink의 초록색 경로와 Recovery 상태를 확인합니다. 기존 LSD/HED/SAM/Canny가
   필요하면 `Advanced: legacy tracing models`를 펼칩니다.
5. 클릭/드래그로 등고선을 추적합니다.
6. `Enter` 또는 우클릭으로 QGIS 편집 버퍼에 추가합니다. 한 번의 Undo로 추가/연장을 되돌릴 수 있습니다.
7. 시작점 근처를 다시 클릭하면 폐합 후 고도값을 입력할 수 있습니다.
8. QGIS의 `Save Layer Edits`로 변경을 확정한 뒤 `Step 4 > DEM 생성…`에서 격자 크기와 출력 경로를 확인합니다.

## 🏔️ Terrain Reconstruction

`elevation` 등고선 + 선택적 표고점 → QGIS 선형 TIN 보간 → GeoTIFF DEM → GDAL hillshade

- 비어 있지 않은 geometry, finite 숫자형 고도, 서로 다른 두 개 이상의 고도값, 비공선 vertex 3개 이상, 미터 단위 투영 CRS가 필요합니다.
- 편집을 저장한 후 실행하세요. 임시 `Spot Heights`도 사용할 수 있지만 point geometry와 등고선과 동일한 CRS가 필요하며, 재현을 위해 파일 레이어로 저장하는 것을 권장합니다.
- QGIS Processing에서 `qgis:tininterpolation`, `gdal:translate`, `native:rasterlayerstatistics`, `gdal:hillshade` 알고리즘이 사용 가능해야 합니다.
- 출력은 기본 2,500만 셀까지 허용하고, 기존 파일은 확인 후에만 덮어씁니다.
- TIN의 범위 밖·입력 누락 구간은 추정 한계로 남습니다. 결과는 자동 확정값이 아니라 검토할 수 있는 지형 가설입니다.

## ⌨️ Shortcuts

- 트레이싱 중 `Ctrl+Z` / `Backspace`: 마지막 체크포인트로 되돌리기
- Ink 추적 중 `Alt`+두 번 클릭: 문자·기호·얼룩 회피 영역. `Alt+Backspace`: 마지막
  영역 제거, `Alt+Shift+Backspace`: 모든 영역 제거
- Ink 추적 시작 후 유색 선 위에서 `C`: 선 색상 지정. `Shift+C`: 색상 해제
- 저장 후 `Ctrl+Z`: QGIS 편집 스택에서 방금 추가·연장한 작업 되돌리기
- `Esc` / `Delete`: 현재 트레이싱 취소
- `Enter` / 우클릭: 현재 선 저장

## 🧯 Troubleshooting

- `ModuleNotFoundError: No module named 'cv2'`
  QGIS Python에 `opencv-python-headless`가 설치되지 않은 상태입니다. 시스템 Python이 아니라 QGIS Python에 설치해야 합니다.
- `HED model is invalid or failed to load`
  손상된 모델일 수 있습니다. 플러그인 UI에서 다시 다운로드하세요.
- MobileSAM/SAM 최신 확인 또는 다운로드가 실패함
  네트워크 또는 `requests` 미설치 문제일 수 있습니다. 필요 시 수동 다운로드 경로를 사용하세요. HED 다운로드는 `requests`와 무관합니다.
- 엣지 미리보기에 아무것도 보이지 않음
  래스터 범위 안으로 확대하고, 다른 모델로도 비교해보세요.
- 선택 model 기능이 안 되는 환경임
  유효한 raster와 2D line output이 있으면 추가 pip·model 없이 `Freehand`를 사용할
  수 있습니다.
- `학습 보조: Ink 사용 / Learned guide: using Ink`
  표시된 상세 상태를 확인하세요. 로컬 `model.onnx`·`manifest.json`, QGIS Python의
  ONNX·ONNX Runtime·SciPy, north-up 3밴드 Byte RGB 지도가 필요합니다. 파일 손상,
  모델 형식 오류나 지원하지 않는 래스터에서도 기존 Ink로 추적할 수 있습니다.
- DEM 실행 전 차단됨
  편집·고도·비공선 vertex·투영 CRS·Processing provider를 확인하세요. 덮어쓸 출력이 QGIS에 로드되어 있다면 먼저 제거해야 합니다.

## 🧩 Repository Note

- 저장소의 [GitHub README](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology)는
  구현 구조, 개발 계획과 release 검증 문서로 연결합니다.
- 설치된 플러그인 폴더에서는 일반 사용자 기준으로 별도 패키징 스크립트가 필요하지 않습니다.

## 🇬🇧 English Summary

- ArchaeoTrace is a local-first QGIS plugin for tracing elevation contours on historical maps and building reviewable terrain hypotheses.
- The QGIS repository published experimental `0.1.5` on 2026-08-26. This guide describes the unpublished experimental `0.1.6` candidate and its Unreleased learned-contour and annotation additions.
- Cursor movement performs only a predecessor lookup after one asynchronous tree build per accepted anchor; the green line is the exact one-click result.
- Assist is literal from 0% (exact cursor, no model work), through coordinate blending, to 100% (the full Live-Wire route).
- `Freehand` needs no additional pip package, external model, or OpenCV, but uses NumPy from the QGIS Python environment.
- Default `Ink Centerline` works after ZIP installation using QGIS' NumPy. SciPy enables the bounded direction-aware Live-Wire for every non-SAM edge mode; without it those modes use the local NumPy nearby-edge snap.
- Native Byte Ink detects dark grid candidates when at least two long horizontal
  and two vertical lines are visible. Following the grid costs more; transverse
  contour crossings remain possible. Detection is asynchronous and falls back
  to Ink on failure. Saturated blue, broken or highly rotated grids can be missed,
  thick grid shoulders can still attract the route, and classification depends
  on the cache context after panning.
- Trial restrictions on learned support and a guard preferring Ink regressed
  real validation cases, so both stay disabled by default. Learned guidance
  remains default OFF. Neighboring-contour switching is unresolved; inspect
  the green preview.
- Models without report-bound selected path costs use contour support only;
  ready/active status states that text avoidance has no selected setting.
  Configured bundles retain their costs. Manual avoidance and ink-color guidance
  still apply, but useful learned text/road exclusion can also be lost.
- After starting an Ink trace, hover over a clear colored contour and press `C`;
  the sample button uses that last map position. `Shift+C` clears it. Ready
  native Byte RGB Ink v2 evidence, SciPy, and assist above 0% are required;
  gray ink, paper, and ambiguous colors are refused. The finite soft mismatch
  cost is max-composed with avoid guidance and shared with Recovery. Panning
  reapplies it, while finishing/cancelling/reset clears it. Samples stay local
  and session-only; same-color text remains ambiguous and performance is unvalidated.
- Trace additions and extensions stay in QGIS' edit buffer; one Undo reverts each operation, and `Save Layer Edits` commits it.
- `MobileSAM` and `SAM` also require `PyTorch` plus their backend packages and model weights.
- Smart Recovery is an experimental, default-OFF EfficientSAM-Ti corridor prior.
  It never replaces a failed run with another backend and always preserves the Ink champion.
- Under Advanced, **Import local model** selects a folder containing `model.onnx`
  and `manifest.json`; explicitly enable **Learned contour guide (Experimental,
  default OFF)** to guide Ink. It requires SciPy, ONNX graph validation and
  single-thread CPU ONNX Runtime on a local north-up three-band Byte RGB
  GeoTIFF/PNG/JPEG. There are no model downloads. Missing or invalid models,
  runtimes, reads or outputs retain Ink, and 0% runs no inference. A late prior
  joins the next anchor without changing a displayed preview, pending click or
  manual proposal.
- **Annotate training crop** opens a native 512×512 RGB PNG with editable
  `contours`, `other_ink`, `other_ink_regions`, `reviewed` and `ignore` layers.
  Label every visible contour in reviewed areas, exclude uncertain pixels, and
  save layer edits before **Review / export labels**. A real human ID and explicit
  all-visible confirmation permit single-reviewer pilot approval. Select one
  open evaluation contour and declare standard/hard difficulty before training.
  Attach the exported JSON to a new versioned manifest with the separate
  [learning CLI](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/learning/README.md).
- The 144 USGS pilot crops are unreviewed drafts. Actual contour training and
  holdout evaluation are pending; synthetic diagnostic weights establish no
  real-map accuracy or Korean-map generalization.
- The DEM workflow requires saved elevation contours in a projected metre CRS; its output is a reviewable hypothesis, not an archaeological ground truth.

## 📚 Citation

Use the repository's
[CITATION.cff](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CITATION.cff)
as the citation source of truth.

## 📄 License

GNU General Public License v2.0
