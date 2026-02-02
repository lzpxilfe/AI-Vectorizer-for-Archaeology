# 🏛️ ArchaeoTrace - AI 등고선 벡터화 QGIS 플러그인

고지도에서 등고선을 AI로 추적하여 벡터 데이터로 변환하는 QGIS 플러그인입니다.

![QGIS 3.34+](https://img.shields.io/badge/QGIS-3.34+-green.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-GPLv2-red.svg)

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **🖊️ 스마트 트레이싱** | 클릭 → 자동 등고선 추적 |
| **✏️ 프리핸드 모드** | AI 없이 자유롭게 그리기 |
| **🔵 폴리곤 닫기** | 시작점 근처에서 자동 닫힘 |
| **🔧 OpenCV 모드** | 빠름, 추가 설치 불필요 |
| **🧠 MobileSAM 모드** | 딥러닝 기반 고정밀 추적 (선택) |

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

플러그인에서 **🧠 MobileSAM** 선택 → **⬇️ 다운로드** 버튼 클릭

---

## 🖱️ 사용법

1. QGIS에 고지도 래스터 로드
2. 플러그인 열기 (좌측 도킹)
3. **래스터 선택** → **SHP 생성**
4. **트레이싱 시작** 클릭
5. 등고선 위를 클릭하며 추적

### ⌨️ 단축키

| 키 | 기능 |
|----|------|
| `좌클릭` | 점 추가 |
| `우클릭` | 저장 후 완료 |
| `Esc` | 마지막 점 취소 |
| `Delete` | 전체 취소 |
| `Enter` | 저장 |

---

## 🛠️ 문제 해결

### "PyTorch/MobileSAM 미설치" 오류
→ 위의 MobileSAM 설치 단계를 따라주세요.
→ **OpenCV 모드**는 추가 설치 없이 바로 사용 가능합니다!

### 플러그인이 보이지 않음
→ QGIS 재시작 후 플러그인 관리자에서 활성화

### 트레이싱이 막힘
→ **프리핸드 모드** 체크 또는 **AI 강도** 슬라이더 낮추기

---

## 📄 라이선스

GNU General Public License v2.0

## 👤 개발자

**lzpxilfe (balguljang2)**  
고고학을 위한 GIS 도구 개발

---

*이 플러그인은 일제강점기 지형도 등 역사지도의 디지털화를 돕기 위해 개발되었습니다.*
