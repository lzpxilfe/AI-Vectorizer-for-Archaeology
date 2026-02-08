# 🏛️ ArchaeoTrace (v0.1.0 "Pallet Town") - AI 등고선 벡터화 QGIS 플러그인

고지도에서 등고선을 AI로 추적하여 벡터 데이터로 변환하는 QGIS 플러그인입니다.

![QGIS 3.34+](https://img.shields.io/badge/QGIS-3.34+-green.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-GPLv2-red.svg)

## 🌟 Citation & Star

이 플러그인이 유용했다면 **GitHub Star ⭐**를 눌러주세요! 개발자에게 큰 힘이 됩니다.  
If you find this repository useful, please consider giving it a star ⭐ and citing it in your work:

```bibtex
@software{ArchDistribution2026,
  author = {lzpxilfe},
  title = {ArchDistribution: Automated QGIS plugin for archaeological distribution maps},
  year = {2026},
  url = {https://github.com/lzpxilfe/ArchDistribution},
  version = {0.1.0}
}
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **🚀 AI 오토 패스** | 마우스만 움직여도(Hover) **최적 경로 미리보기** (초록색 점선) 및 클릭 확정 |
| **👀 시인성 강화** | 더 두껍고 명확한 경로 미리보기 제공 |
| **🦴 엣지 골격화** | 두꺼운 등고선을 **1px 중심선**으로 변환하여 정밀도 향상 |
| **💾 체크포인트** | 클릭으로 중간 저장, **폴리곤(Straight Lines) 스타일** 스무딩 적용 |
| **🛡️ 안전한 우클릭** | 실수 방지를 위한 **컨텍스트 메뉴** (Undo/Finish/Cancel) |
| **� 다국어 지원** | QGIS 언어 설정을 따르는 **국제화(i18n)** 지원 (한글/영어) |
| **⌨️ 단축키 지원** | **Ctrl+Z**, **Backspace**로 손쉬운 되돌리기 |

### 🤖 AI 모델 (4가지)

| 모델 | 속도 | 특징 |
|------|------|------|
| 🔧 **OpenCV Canny** | ⚡최고 | 기본, 추가 설치 불필요 |
| 📐 **LSD** | ⚡빠름 | 선분 기반 검출 |
| 🧠 **HED** | 보통 | 딥러닝 엣지 (56MB) |
| 🎯 **MobileSAM** | 느림 | 최고 품질 (PyTorch 필요) |

---

## 📦 설치 방법

### Step 1: 플러그인 다운로드

```bash
git clone https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology.git
```

또는 [ZIP 다운로드](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/archive/refs/heads/main.zip)

### Step 2: QGIS에 연결 (심볼릭 링크)

**Windows PowerShell (관리자 권한):**
```powershell
$src = "C:\다운로드경로\AI-Vectorizer-for-Archaeology\ai_vectorizer"
$dest = "$env:APPDATA\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer"
New-Item -ItemType Junction -Path $dest -Target $src
```

### Step 3: 기본 의존성 설치

**OSGeo4W Shell** (시작메뉴에서 검색):
```bash
python -m pip install opencv-python-headless scikit-image
```

### Step 4: QGIS 재시작
플러그인 → ArchaeoTrace 활성화

---

## 🖱️ 사용법

### 🖱️ 사용법 (하이브리드 모드)
1. **마우스 이동 (버튼 뗌)**: 현재 위치에서 마우스까지의 **AI 추천 경로**를 미리봅니다 (초록색 점선).
2. **클릭**: 추천 경로를 확정하고 해당 지점부터 다시 시작합니다. (Point-and-Click)
3. **드래그**: 기존처럼 직접 그릴 수도 있습니다 (수동 보정 시 유용).

### 기본 흐름
1. QGIS에 고지도 래스터 로드
2. 플러그인 열기 (좌측 도킹)
3. **래스터 선택** → **AI 모델 선택 (Default: Canny)**
4. **트레이싱 시작** 클릭
5. 등고선을 따라 마우스를 움직이며 클릭, 클릭!

### ⌨️ 조작법

| 동작 | 기능 |
|------|------|
| **마우스 이동** | AI 경로 미리보기 (초록색 점선) |
| **Shift/Ctrl + 이동** | **수동 직선 모드** (AI 끄고 직선 그리기) |
| **클릭** | 경로 확정 (빨간색) 및 체크포인트 |
| **더블 클릭 (시작점)** | **점(Spot Height) 생성** (독립된 높이 점) |
| **기존 선 끝 조준** | **이어 그리기 (Snap)** (기존 선에서 계속 그리기) |
| **드래그** | 수동 그리기 (Mouse Following) |
| **Ctrl+Z / Backspace** | 실행 취소 (이전 체크포인트로) |
| **우클릭** | **현재 선 저장** (열린 선으로 즉시 저장) |
| **Enter** | **현재 선 저장** (열린 선으로 즉시 저장) |
| **Esc** | **그리기 취소** (현재 선 초기화) |
| **Delete** | 전체 취소 및 초기화 |
| **시작점 클릭** | 폴리곤 닫기 (폐곡선 - 해발값 입력) |
| **이어그리기** | 기존 선 끝점에 마우스 올리기(분홍 점) → **클릭** |

### 💡 팁
*   **원 그리기**: 2점보다는 **3점**을 찍어서 닫는 것을 추천합니다. (더 예쁜 원이 됩니다)
*   **저장된 선 지우기**: `Ctrl+Z`는 그리기 중인 선만 취소합니다. 이미 저장된 선은 QGIS의 **객체 지우기** 도구를 사용하세요.

### 💡 추천 워크플로우

```
드래그 시작 → 클릭(체크포인트) → 드래그 → 클릭(체크포인트) → ...
                ↓                                        
           실수하면 Ctrl+Z → 체크포인트로 돌아감 → 다시 드래그
```

---

## 🧠 고급 AI 모델 설치 가이드 (선택사항)
> ⚠️ **OpenCV, LSD 모드는 설치 없이 바로 사용 가능합니다!**

### 1. HED 모델 (딥러닝 엣지)
*   **특징**: 딥러닝 기반으로 매우 부드러운 엣지를 검출합니다.
*   **설치법**:
    1. 플러그인에서 **🧠 HED** 선택
    2. **⬇️ 다운로드** 버튼 클릭 (자동 설치)

### 2. MobileSAM (최고 품질)
*   **특징**: 페이스북의 SAM 모델을 경량화하여 최고의 정확도를 제공합니다. (PyTorch 필요)
*   **설치법**:

#### Step 1: PyTorch 설치
**OSGeo4W Shell**에서 실행:
```bash
python -m pip install torch torchvision timm
```

#### Step 2: MobileSAM 라이브러리 설치
1. [MobileSAM 라이브러리 ZIP 다운로드](https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip)
2. 압축 해제 후 `mobile_sam` 폴더를 다음 경로에 복사:
   `C:\Users\[사용자명]\AppData\Roaming\Python\Python312\site-packages\`

#### Step 3: 모델 가중치 (Weights)
1. 플러그인에서 **🎯 MobileSAM** 선택
2. **⬇️ 다운로드** 버튼 클릭 (자동 설치)

---

## 🛠️ 문제 해결

### "PyTorch/MobileSAM 미설치" 오류
→ 위의 MobileSAM 설치 단계를 따라주세요.
→ **OpenCV 모드**는 추가 설치 없이 바로 사용 가능합니다!

### "Pathfinding timeout" 경고
→ 경로 탐색 시간이 너무 오래 걸릴 때 나타납니다.
→ 줌을 확대(Zoom In)하거나 더 짧은 구간을 선택하세요.

### 플러그인이 보이지 않음
→ QGIS 재시작 후 플러그인 관리자에서 활성화

### 선이 자글자글함
→ scikit-image가 설치되었는지 확인하세요.
→ 천천히 부드럽게 드래그하세요.

### 체크포인트 활용
→ 긴 등고선은 중간중간 클릭해서 체크포인트 저장
→ 실수하면 Ctrl+Z로 되돌린 후 다시 그리기

---

## 📄 라이선스

GNU General Public License v2.0

## 👤 개발자

**lzpxilfe (balguljang2)**  
고고학을 위한 GIS 도구 개발

---

*이 플러그인은 일제강점기 지형도 등 역사지도의 디지털화를 돕기 위해 개발되었습니다.*
