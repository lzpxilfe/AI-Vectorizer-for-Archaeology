# 🏛️ ArchaeoTrace - AI 등고선 벡터화 QGIS 플러그인

고지도에서 등고선을 AI로 추적하여 벡터 데이터로 변환하는 QGIS 플러그인입니다.

![QGIS 3.34+](https://img.shields.io/badge/QGIS-3.34+-green.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-GPLv2-red.svg)

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
| **클릭** | 경로 확정 (빨간색) 및 체크포인트 |
| **드래그** | 수동 그리기 (Mouse Following) |
| **Ctrl+Z / Backspace** | 실행 취소 (이전 체크포인트로) |
| **우클릭** | **메뉴 호출** (Finish, Undo, Cancel) |
| **Enter** | 현재 라인 저장 및 종료 |
| **Esc** | 최근 10개 점 취소 |
| **Delete** | 전체 취소 및 초기화 |
| **시작점 클릭** | 폴리곤 닫기 (폐곡선) |

### 💡 추천 워크플로우

```
드래그 시작 → 클릭(체크포인트) → 드래그 → 클릭(체크포인트) → ...
                ↓                                        
           실수하면 Ctrl+Z → 체크포인트로 돌아감 → 다시 드래그
```

---

## 🧠 MobileSAM 설치 (선택사항)

> ⚠️ **OpenCV 모드만 사용할 경우 이 단계는 건너뛰세요!**

MobileSAM은 딥러닝 기반으로 더 정확하지만, 추가 설치가 필요합니다.

### Step 1: PyTorch 설치

**OSGeo4W Shell**에서:
```bash
python -m pip install torch torchvision timm
```

### Step 2: MobileSAM 다운로드

1. [MobileSAM ZIP 다운로드](https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip)
2. 압축 풀기

### Step 3: mobile_sam 폴더 복사

압축 푼 폴더에서 `mobile_sam` 폴더를:
```
MobileSAM-master\mobile_sam
```

아래 경로에 복사:
```
C:\Users\[사용자명]\AppData\Roaming\Python\Python312\site-packages\
```

### Step 4: 설치 확인

**OSGeo4W Shell**:
```bash
python -c "from mobile_sam import sam_model_registry; print('OK')"
```

`OK`가 출력되면 성공! 🎉

### Step 5: 모델 가중치 다운로드

플러그인에서 **🎯 MobileSAM** 선택 → **⬇️ 다운로드** 버튼 클릭

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
