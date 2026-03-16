# ArchaeoTrace v0.1.3

AI-assisted contour digitizing plugin for QGIS, focused on historical maps.

![QGIS 3.22+](https://img.shields.io/badge/QGIS-3.22+-green.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-GPLv2-red.svg)

## Korean (한국어)

### 중요 안내 (QGIS 업로드 25MB 제한 대응)
- QGIS 플러그인 업로드 제한(25MB)에 맞추기 위해, 대용량 AI 가중치 파일은 ZIP에 포함하지 않습니다.
- 기본 모델 `Canny`, `LSD`는 바로 사용 가능합니다.
- `HED`(약 56MB), `MobileSAM`(약 39MB + PyTorch), `SAM`(약 358MB + PyTorch)은 필요할 때 다운로드하여 사용합니다.

### 설치
1. ZIP 설치 또는 소스 설치
- ZIP 설치: QGIS `플러그인 > 플러그인 설치 및 관리 > ZIP에서 설치`
- 소스 설치: `ai_vectorizer` 폴더를 아래 위치에 복사

2. 플러그인 폴더 경로
- Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer`
- macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/ai_vectorizer`
- Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ai_vectorizer`

3. (선택) 의존성 설치
- 기본 트레이싱: `python -m pip install opencv-python-headless scikit-image`
- 모델 다운로드/업데이트 확인: `python -m pip install requests`
- MobileSAM 사용 시: `python -m pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git`
- SAM 사용 시: `python -m pip install torch torchvision git+https://github.com/facebookresearch/segment-anything.git`

4. QGIS 재시작 후 플러그인 활성화

### AI 모델 다운로드 (쉬운 방법)
1. 플러그인 `Step 3`에서 모델을 `HED`, `MobileSAM`, 또는 `SAM`으로 선택
2. `선택 SAM 모델 최신 확인` 버튼으로 원격 최신 상태 확인
3. `다운로드` 또는 `업데이트` 버튼 실행
4. 실패 시 `SAM 상태 리포트`를 생성해 진단 정보(JSON) 확인

### AI 모델 다운로드가 어려운 경우 (수동 설치)
회사망/프록시/방화벽 환경에서는 브라우저 수동 다운로드가 더 잘 되는 경우가 있습니다.

1. 아래 파일을 브라우저로 다운로드
- HED 가중치: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- MobileSAM 가중치: `https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt`
- SAM (ViT-B) 가중치: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

2. 아래 경로에 직접 복사
- HED: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/core/models/hed_pretrained_bsds.caffemodel`
- MobileSAM: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/mobile_sam.pt`
- SAM (ViT-B): `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/sam_vit_b_01ec64.pth`

3. QGIS 재시작 후 플러그인에서 상태 확인
- HED: `✅ HED 모델 로드됨`
- MobileSAM: `✅ MobileSAM loaded`
- SAM: `✅ SAM (VIT_B) loaded`

### 사용 순서
1. 래스터 지도 선택
2. 출력 SHP 생성/선택
3. 모델 선택 후 트레이싱 시작
4. 클릭/드래그로 추적
5. 우클릭 또는 `Enter`로 저장

### 단축키
- `Ctrl+Z` / `Backspace`: 체크포인트 되돌리기
- `Esc` / `Delete`: 현재 작업 취소
- `Enter` / 우클릭: 현재 선 저장

### 문제 해결
- 모델 다운로드 실패: 네트워크/프록시 확인 후 재시도, 필요 시 수동 설치
- HED가 Canny로 동작: HED 가중치 파일 경로 및 파일명 확인
- MobileSAM 인식 실패: `torch`, `mobile_sam` 설치 여부 확인
- SAM 인식 실패: `torch`, `segment_anything` 설치 여부 확인

---

## English

### Important Note (QGIS 25MB Plugin Upload Limit)
- To stay under the QGIS plugin upload limit (25MB), large AI weight files are not bundled in the plugin ZIP.
- `Canny` and `LSD` work out of the box.
- `HED` (~56MB), `MobileSAM` (~39MB + PyTorch), and `SAM` (~358MB + PyTorch) are downloaded on demand.

### Installation
1. Install via ZIP or source
- ZIP: QGIS `Plugins > Manage and Install Plugins > Install from ZIP`
- Source: copy `ai_vectorizer` into the plugin directory

2. Plugin directory paths
- Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer`
- macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/ai_vectorizer`
- Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ai_vectorizer`

3. (Optional) install dependencies
- Core tracing: `python -m pip install opencv-python-headless scikit-image`
- For model downloads/update checks: `python -m pip install requests`
- For MobileSAM: `python -m pip install torch torchvision git+https://github.com/ChaoningZhang/MobileSAM.git`
- For SAM: `python -m pip install torch torchvision git+https://github.com/facebookresearch/segment-anything.git`

4. Restart QGIS and enable the plugin

### AI Model Download (Easy In-App Flow)
1. In `Step 3`, select `HED`, `MobileSAM`, or `SAM`
2. Click `Check Selected SAM Model`
3. Click `Download` or `Update`
4. If needed, export `SAM Status Report` (JSON) for diagnostics

### If Download Is Difficult (Manual Fallback)
In corporate networks (proxy/firewall), browser download + manual copy is often more reliable.

1. Download files manually
- HED weights: `https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel`
- MobileSAM weights: `https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt`
- SAM (ViT-B) weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

2. Copy to exact locations
- HED: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/core/models/hed_pretrained_bsds.caffemodel`
- MobileSAM: `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/mobile_sam.pt`
- SAM (ViT-B): `<QGIS_PROFILE>/python/plugins/ai_vectorizer/models/sam_vit_b_01ec64.pth`

3. Restart QGIS and verify status in plugin UI
- HED: `✅ HED model loaded`
- MobileSAM: `✅ MobileSAM loaded`
- SAM: `✅ SAM (VIT_B) loaded`

### Quick Start
1. Select raster map
2. Create/select output SHP
3. Choose model and start tracing
4. Click/drag along contours
5. Save with right-click or `Enter`

### Shortcuts
- `Ctrl+Z` / `Backspace`: undo to checkpoint
- `Esc` / `Delete`: cancel current trace
- `Enter` / right-click: save current line

### Troubleshooting
- Download failed: verify network/proxy and retry, or use manual fallback
- HED keeps falling back to Canny: check HED file path and filename
- MobileSAM not detected: verify `torch` and `mobile_sam` installation
- SAM not detected: verify `torch` and `segment_anything` installation

### Release Packaging
- Do not edit the versioned release folder by hand. Rebuild it from the root source tree instead.
- Build release artifacts with `python3 scripts/package_release.py`
- Verify that the versioned release folder and `dist/ai_vectorizer-<version>.zip` are in sync with `python3 scripts/package_release.py --check`
- The generated ZIP uses the installable plugin folder name `ai_vectorizer/` inside the archive for QGIS `Install from ZIP`

## Citation

```bibtex
@software{ArchaeoTrace2026,
  author = {lzpxilfe},
  title = {ArchaeoTrace: AI-assisted contour digitizing QGIS plugin for historical maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology},
  version = {0.1.3}
}
```

## License
GNU General Public License v2.0
