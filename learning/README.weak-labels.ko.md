# AI 약한 정답으로 시작하는 Colab 실험

이 경로는 사람이 검수한 파일럿과 분리된 **자동 주석 학습 실험**입니다. AI가 만든
주석은 `machine_labeled`, 검수 수준은 `automated_weak_labels`이며 사람의 승인을
대신하지 않습니다. Validation 손실은 자동 정답과의 일치 정도입니다. 실제 등고선
정확도나 공식 Holdout 성능을 의미하지 않습니다.

원본 파일럿을 보존한 채 후보와 별도 데이터 ZIP을 재현하려면 Python 3.12 환경에서
`requirements-weak-labels.txt`를 설치하고 다음 명령을 실행합니다. 이미 존재하는
실험 결과 폴더나 ZIP은 덮어쓰지 않습니다. Tesseract가 있으면 OCR 상자를 판단 보류
영역으로만 사용하며 실행 환경과 생성 코드 해시를 데이터에 기록합니다.

```bash
python scripts/prepare_ai_drafts.py --dataset /absolute/pilot-data --output /absolute/ai-drafts
python scripts/build_weak_experiment.py --dataset /absolute/pilot-data \
  --drafts /absolute/ai-drafts --output /absolute/weak-data --archive /absolute/weak-dataset.zip
```

색상 분리가 불가능한 흑백 잉크는 판단 보류로 남깁니다. 직선·OCR 마스크는 실제
등고선까지 제외할 수 있어 오버레이와 크롭별 감독 픽셀 통계를 함께 확인해야 합니다.
`reviewed` 폴리곤은 이 실험에서 기계가 처리한 범위이며 사람 검수를 뜻하지 않습니다.

1. `weak-dataset.zip`을 Drive의 `ArchaeoTrace/ai-weak-pilot` 폴더에 올립니다.
   원래의 사람 검수용 `approved-dataset.zip`과 실험 결과 폴더는 유지합니다.
   업로드 크기 제한으로 ZIP을 나눴다면 `weak-dataset.zip.part-01`,
   `weak-dataset.zip.part-02`, `weak-dataset.parts.json`을 같은 폴더에 올립니다.
   전체 ZIP이 없으면 노트북이 각 조각과 완성 ZIP의 크기·SHA-256을 검증하여
   Colab 로컬 디스크에서 원래 ZIP을 복원합니다. 학습 데이터와 해시는 같습니다.
2. `ArchaeoTrace_Colab_weak_labels.ipynb`를 Colab에서 열고 기존 계정의 GPU
   런타임을 연결합니다. 추가 구독이나 계산 단위 구매는 필요할 때 별도로 결정합니다.
3. 설정 셀의 `CODE_COMMIT`에 약한 정답 지원 구현의 검증된 전체 40자리 커밋을
   넣습니다. 배포본에 고정값이 있으면 유지합니다. `ALLOW_WEAK_LABELS=True`,
   `RESUME_EXISTING=True`, `OOM_RETRY=False`로 시작합니다.
4. 환경 확인, Drive 연결, 코드 설치, 데이터 검증 셀을 차례로 실행합니다.
   패키지 설치 셀에서 재시작 안내로 중단되면 **런타임 → 세션 다시 시작** 후
   같은 설정으로 **모두 실행**합니다. 설치 전 불러온 Pillow·NumPy 등을 현재
   커널에서 계속 사용하지 않도록 막는 절차이며, 모델 학습은 아직 시작하지 않습니다.
   전체 144개와 `train=72`, `validation=24`, `calibration=24`, `holdout=24`를
   확인합니다. 출력의 `Weak-label train/validation: 72 24`와 실제 CUDA GPU 이름을
   확인한 다음 학습 셀을 실행합니다.
5. 세 seed `3407`, `3408`, `3409`가 순서대로 학습됩니다. 각 epoch의 학습·Validation
   손실과 실제 optimizer 진행이 출력됩니다. Drive의
   `ArchaeoTrace/ai-weak-pilot/runs/seed-<seed>`에 `latest.pt`, `best.pt`가 저장됩니다.
6. 연결이 끊기면 같은 데이터·코드·런타임 버전·결과 폴더로 노트북을 다시
   실행합니다. `latest.pt`가 있으면 optimizer·RNG·샘플 순서까지 재개합니다.
   GPU 메모리 부족이면 `OOM_RETRY=True`로 바꾸고 설정 셀과 학습 셀을 다시
   실행합니다. 배치 1·누적 16으로 명시적으로 재개합니다.
7. 학습이 끝나면 마지막 셀을 실행합니다. 자동 정답 Validation 손실로 후보 하나를
   선택하여 `selected-model.json`을 동결하고 해당 seed의 `bundle-<hash>`에
   `model.onnx`, `manifest.json`을 함께 내보냅니다. CPU PyTorch/ONNX 출력 오차도
   검사합니다. 두 파일을 함께 보관합니다.

Drive 경로의 기본값은 다음과 같습니다.

```python
DATASET_ZIP = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-pilot/weak-dataset.zip"
DATASET_PARTS_MANIFEST = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-pilot/weak-dataset.parts.json"
OUTPUT_DIRECTORY = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-pilot/runs"
```

분할 업로드를 사용하면 데이터 셀의 `Verified split upload` 출력에서 조각 수,
완성 ZIP의 바이트 수와 SHA-256을 확인합니다. 조각이 누락·변경되거나 경로·해시가
맞지 않으면 학습 전에 중단합니다. 전체 ZIP이 있으면 기존 방식으로 읽습니다.

학습은 Colab 로컬 디스크의 데이터만 읽으며 체크포인트는 매 epoch 또는 5분 경과 후
다음 optimizer 저장 지점에 Drive로 저장합니다. 최대 60 epoch, patience 8, 초기
배치 2·누적 8입니다. 데이터나 주석을 수정했다면 새 실험 결과 폴더를 사용합니다.
버전이 다른 런타임은 정확한 재개 조건에 맞지 않아 거부될 수 있습니다.

Calibration/Holdout 48개는 학습 입력으로 열지 않습니다. 자동 주석에는 공식 평가
대상 선을 넣지 않으며, 이 실험의 결과로 기존 Holdout 합격 조건을 통과했다고
주장할 수 없습니다. 모델 매니페스트에도 `human_review_required=true`,
`review_level=automated_weak_labels`, `research_ready=false`가 남습니다.

QGIS에서 실험 모델을 시각적으로 살펴볼 수 있지만 기본 기능으로 승격하려면
사람이 검수한 독립 정답을 확보하고 원래 Calibration/Holdout 평가를 수행해야 합니다.
