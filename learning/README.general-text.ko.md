# 일반 글자 공백용 Colab 실험

[`colab_train_korea_text_weak.ipynb`](colab_train_korea_text_weak.ipynb)는 기존 혼합 데이터의
Train 162개·Validation 48개를 재사용하는 별도 **약한 정답 실험**입니다. 원래 파일럿
144개는 사람 검수를 완료한 데이터가 아닙니다. 새 지도 진단 크롭과 이번 `a`·`面`
비교 사례를 학습 데이터에 추가하지 않습니다. Calibration/Holdout도 학습하지 않습니다.

새 `word_gaps` 증강은 학습 정답에 이미 연속해서 표시된 선 위에만 합성 글자를 씁니다.
저장소에 직접 작성한 Latin·한자·한글 획 atlas를 사용하므로 호스트 폰트나 외부 폰트
다운로드가 필요하지 않습니다. 실제 역사 지도 서체나 원문을 재현했다고 주장하지 않습니다.
흰 여백을 만든 인쇄와 검정·파랑·갈색 덧인쇄를 섞습니다. 원래 보이지 않던 역사 지도
공백을 임의 정답으로 연결하지 않으며, `ignore` 픽셀은 합성 글자로 덮지 않습니다.

가까운 등고선이 있는 예제를 함께 뽑지만 **옆 등고선도 계속 양성 등고선 정답**입니다.
새 글자 획 중 모든 등고선 정답 통로 밖의 픽셀만 회피 대상으로 추가합니다. 이 두 출력
모델은 등고선 인스턴스나 고도 동일성을 직접 학습하지 않으므로, QGIS의 끝점·접선·
순서와 옆 선 교차 검사는 계속 필요합니다. 픽셀 손실 개선만으로 선 갈아타기 해결을
선언하지 않습니다.

노트북은 다음 경로를 사용합니다.

```python
# 기존 파일은 읽기 전용으로 재사용합니다.
DATASET_ZIP = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-korea/weak-dataset.zip"
DATASET_PARTS_MANIFEST = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-korea/weak-dataset.parts.json"
# 이전 모델·체크포인트는 유지합니다.
OUTPUT_DIRECTORY = "/content/drive/MyDrive/ArchaeoTrace/ai-weak-korea-general-text-v1/runs"
TRAIN_OPTIONS = {"ink_recolor": True, "label_gaps": True, "word_gaps": True}
```

설정 셀의 `CODE_COMMIT`에 검증하고 `main`에 게시한 학습 코드의 40자리 커밋을 넣습니다.
데이터 매니페스트 SHA-256은 기존 `b2fd2a087f75de4e5e1d31b44f13dfdb1a8af3963e18e68f81f9cb4a91e32249`
로 고정합니다. 설정·코드·atlas·데이터를 `text-experiment.json`에 결속하고 Train에만
적용한 원본/합성 미리보기를 먼저 보여 줍니다. 예제가 하나도 검증되지 않으면 학습하지
않습니다. 이전 실험 소스가 커널에 남아 있으면 세션을 다시 시작해야 합니다.

세 seed `3407`, `3408`, `3409`는 ImageNet ResNet18에서 새로 시작합니다. 기존 숫자 공백
모델 체크포인트를 새 증강에 그대로 `resume`하지 않습니다. 같은 새 실험의 연결이 끊긴
경우에만 `latest.pt`의 optimizer·RNG·진행 위치를 재개합니다. 배치 2·누적 8, AdamW
`1e-4`, 최대 60 epoch, patience 8, BCE·Dice·가중치 `0.2`의 soft-clDice는 유지합니다.
체크포인트는 매 epoch 또는 5분 경과 후 다음 optimizer 경계에서 Drive에 저장합니다.

학습이 완료되면 기존 Validation만으로 seed를 선택하여
`ai-weak-contour-guide-korea-text-v1`의 FP32 ONNX 번들을 새 디렉터리에 내보냅니다.
약한 정답·사람 검수 필요 상태는 결과에 남습니다. 세 seed가 실제로 끝나고 ONNX가
검증되기 전에는 새 모델이 완성됐다고 보지 않습니다. 기존 Ink·이전 두 모델과 실제
QGIS의 최종 저장 경로·옆 선 접촉·취소·한 번의 Undo를 비교한 뒤 실험용으로 판단합니다.

같은 실험을 명령행에서 실행하는 예시는 다음과 같습니다.

```bash
python -m learning.train /absolute/frozen-data/manifest.json /absolute/new-text-runs/seed-3407 \
  --device cuda --seed 3407 --allow-weak-labels --initialize-pretrained \
  --ink-recolor --label-gaps --word-gaps
```
