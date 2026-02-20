# ArchaeoTrace v0.1.2

AI-assisted contour digitizing plugin for QGIS, focused on historical maps.

![QGIS 3.22+](https://img.shields.io/badge/QGIS-3.22+-green.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-GPLv2-red.svg)

## Korean (한국어)

### 소개
ArchaeoTrace는 고지도에서 등고선을 반자동으로 추적하여 라인 벡터로 저장하는 QGIS 플러그인입니다.

### v0.1.2 주요 기능
- AI 자동 경로 미리보기 + 클릭 확정
- Canny/LSD/HED/MobileSAM 4가지 모델
- 한국어/영어 UI 선택
- MobileSAM 최신 확인 및 업데이트 버튼
- SAM 상태 리포트(JSON) 생성 기능

### 설치
1. 저장소 클론 또는 ZIP 다운로드
   - `git clone https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology.git`
2. QGIS 플러그인 폴더에 `ai_vectorizer` 배치
   - 기본 경로: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer`
3. (선택) 의존성 설치
   - `python -m pip install opencv-python-headless scikit-image`
4. QGIS 재시작 후 플러그인 활성화

### 사용 순서
1. 래스터 지도 선택
2. 출력 SHP 생성/선택
3. 모델 선택 후 트레이싱 시작
4. 클릭/드래그로 추적
5. 우클릭 또는 Enter로 저장

### 단축키
- `Ctrl+Z`/`Backspace`: 체크포인트 되돌리기
- `Esc`/`Delete`: 현재 작업 취소
- `Enter`/우클릭: 현재 선 저장

### SAM 최신 확인/업데이트
1. 모델을 MobileSAM으로 선택
2. `MobileSAM 최신 확인` 클릭
3. 상태에 따라 `다운로드` 또는 `업데이트` 클릭
4. 필요 시 `SAM 상태 리포트`로 진단 JSON 생성

## English

### Overview
ArchaeoTrace is a QGIS plugin for semi-automatic contour tracing on historical maps, saving results as vector lines.

### Highlights in v0.1.2
- AI-assisted hover preview + click-to-confirm tracing
- Four models: Canny, LSD, HED, MobileSAM
- UI language switch: Korean / English
- MobileSAM latest-check and update flow
- Built-in SAM status report export (JSON)

### Installation
1. Clone repo or download ZIP
   - `git clone https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology.git`
2. Place `ai_vectorizer` under QGIS plugin directory
   - Default: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer`
3. (Optional) install dependencies
   - `python -m pip install opencv-python-headless scikit-image`
4. Restart QGIS and enable the plugin

### Quick Start
1. Select raster map
2. Create/select output SHP
3. Choose model and start tracing
4. Click/drag along contours
5. Save with right-click or Enter

### Shortcuts
- `Ctrl+Z`/`Backspace`: undo to checkpoint
- `Esc`/`Delete`: cancel current trace
- `Enter`/right-click: save current line

### MobileSAM Version Check/Update
1. Select MobileSAM model
2. Click `Check MobileSAM Latest`
3. Click `Download` or `Update` based on status
4. Use `SAM Status Report` for support diagnostics

## Citation

```bibtex
@software{ArchaeoTrace2026,
  author = {lzpxilfe},
  title = {ArchaeoTrace: AI-assisted contour digitizing QGIS plugin for historical maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology},
  version = {0.1.2}
}
```

## License
GNU General Public License v2.0
