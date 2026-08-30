# Ink Centerline v2와 Smart Recovery

이 문서는 `Unreleased` 개발 소스에 추가된 차세대 추적 경로를 설명합니다. 공개된
`0.1.5` artifact와 metadata는 그대로 유지됩니다. 새 정확도 수치나 다른 서비스보다
우수하다는 주장은 공개 실지도 holdout이 완성되고 사전 정의된 gate를 통과하기 전에는
하지 않습니다.

## 설계 원칙

Ink Centerline은 항상 첫 번째 경로인 **champion**입니다. EfficientSAM-Ti는 Ink가
약하다고 판단된 구간에서만 의미 영역을 제안하는 **challenger provider**입니다.
SAM의 mask를 선으로 바로 저장하거나 Ink와 이진 OR하지 않습니다.

```text
Ink v2 evidence → bounded Live-Wire champion
                         │
                         ├─ confident → Ink 유지
                         │
                         └─ low confidence 또는 명시적 Retry
                              → EfficientSAM corridor
                              → continuous Ink+corridor cost
                              → challenger
                              → 안전 arbiter
                                   ├─ accept → Enhanced
                                   └─ reject/error/cancel → Ink fallback
```

Smart Recovery는 기본적으로 꺼져 있습니다. 모델 설치와 기능 활성화는 서로 다른
명시적 사용자 동작이며, 추적 중에는 네트워크를 사용하지 않습니다.

## Ink v2 evidence

`core/line_evidence.py`의 `LineEvidence`는 QGIS와 분리된 불변 NumPy 계약입니다.

- `center_score`: `[0, 1]` 범위의 연속 guide score. 보정된 확률이 아닙니다.
- `centerline`: 기존 snap과 미리보기에 사용할 한 픽셀 이진 중심선
- `tangent_x`, `tangent_y`, `coherence`: Live-Wire 방향 비용
- `scale_px`: 각 위치에서 가장 강한 black top-hat scale

`EdgeDetector.detect_ink_evidence()`는 9·15·31px black top-hat, 명도와 RGB 채널
증거, source-grid 기준 고정 tile과 halo 정규화를 결합합니다. 기존
`detect_edges(method="ink")`는 `0.1.5` 회귀 비교를 위해 변경하지 않았습니다.

필터와 tile 단위는 cache pixel이 아니라 source pixel입니다. UI는 보이는 범위가
걸친 128px tile core를 먼저 완성한 뒤, 16px response halo와 31px 필터의 15px
반경을 합친 31px source context를 읽습니다. 각 tile은 이 raw-response halo만으로
threshold·호환 중심선·방향장을 독립 계산하므로, 완전한 core는 겹치는 cache
read에서도 같은 배열을 냅니다.

현재 이 계약을 제품에서 활성화하는 입력은 고정 dtype 범위를 가진 native integer
raster입니다. block별 min/max stretch가 필요한 float raster와, halo 포함 범위가
1000×1000px을 넘어 native-resolution cache를 만들 수 없는 축척에서는 두 좌표계를
섞어 근사하지 않습니다. UI가 구체적인 `Ink fallback` 사유를 표시하고 기존 Ink v1을
유지합니다. 지원되는 integer raster에서는 충분히 확대하면 v2가 source-grid에서
다시 활성화됩니다. Recovery 입력은 추가로 native `Byte` raster만 허용합니다.
`UInt16` 등 더 넓은 정수형에서는 범위 전체를 8-bit로 축소해 실제 0–255 잉크를
검게 뭉개는 대신 Ink v2 champion만 유지하고 구체적인 fallback 사유를 표시합니다.

v1 fallback은 이 확장 block을 재사용하지 않습니다. UI가 0.1.5와 같은 visible
extent·resampling·8-bit 변환을 별도로 읽어 task에 보관하고, v2가 성공했을 때만
확장 cache를 게시합니다. v2 비활성·오류·취소에서는 visible v1 image, mask,
transform을 그대로 게시하므로 넓어진 문맥이 기준선 결과를 바꾸지 않습니다.

Live-Wire의 `evidence=None`은 기존 계산을 그대로 사용합니다. evidence가 있으면
연속 중심 score와 미리 계산된 축 방향을 직접 사용합니다. 이 경로에서는 거리 변환,
명암 정규화, Gaussian/Sobel과 구조 텐서를 다시 계산하지 않습니다. 기존 320px 제한
창, 진행 방향 bias, endpoint snap, 최대 우회율과 0–100% geometry blend는 유지됩니다.

## Recovery gate와 안전 arbiter

`core/smart_recovery.py`는 모델이나 QGIS를 import하지 않는 정책 계층입니다.

- gate 입력: 경로의 하위 10% Ink 지지도, 평균 지지도, 가장 긴 저지지도 run,
  coherence, detour ratio, branch density와 endpoint error
- corridor 결합: 강한 Ink는 mask 밖에서도 낮은 비용을 유지하고, 약한 구간에서만
  mask 바깥 비용을 높임
- arbiter: endpoint, 탐색 범위, detour, 강한 Ink 보존율, 경로 간 p95 거리와 실제
  약한 구간 개선을 모두 검사

현재 정책 ID는 `smart-recovery-gate-v1-provisional`입니다. 모든 임계값은 canonical
JSON SHA-256과 함께 benchmark evidence에 기록됩니다. 24개 calibration crop에서
임계값을 고정하고 24개 locked holdout을 실행하기 전까지는 provisional 표시를
제거하지 않습니다.

EfficientSAM 제품 adapter는 content-addressed encoder/decoder를 매번 크기와 SHA-256으로
검증한 뒤 CPU ONNX Runtime으로만 엽니다. 41MB bundle hashing과 ONNX session 생성도
취소·generation 검사를 갖춘 별도 `QgsTask`에서 수행하므로 checkbox나 trace 시작이
UI thread를 막지 않습니다. 준비 중에는 Ink만 즉시 시작하며, 현재 task에 검증된
engine이 도착한 뒤에만 Recovery를 활성화합니다. 모델 없음, 손상, dependency 부재,
NaN·shape 오류, task 취소 또는 stale cache에서는 challenger를 채택하지 않습니다.

## Benchmark 계약

비교 ID는 다음처럼 분리합니다.

- `ink-livewire-v1`: 동결된 `0.1.5` Ink/Live-Wire 기준선
- `ink-livewire-v2`: 같은 prompt에서 `LineEvidence`를 사용하는 경로
- `efficientsam-ti-onnx-v1`: 기존 독립 EfficientSAM 기록 보존
- `ink-v2-effsam-recovery-v1`: gate, corridor와 arbiter까지 포함한 제품 경로

prompt schema v2의 선택 필드 `previous_xy`는 이전 확정점에서 `start_xy`로 들어오는
방향만 나타냅니다. SAM positive prompt에는 포함하지 않습니다. v1 요청과 기존 prompt
hash는 계속 읽고 검증할 수 있습니다. `previous_xy`를 생략한 새 worker request /2도
prompt evidence /2로 hash하며, request /1만 기존 prompt evidence /1 hash를 유지합니다.
공개 materialized crop은 prompt의 `schema_version`을 명시적으로 /2로 기록합니다.

실제 crop은 `source_tile_origin_xy`를 명시합니다. Ink v2와 Recovery worker는 이를
`detect_ink_evidence(..., tile_origin=...)`에 전달하고 crop image SHA-256과 함께 별도의
`source_grid_input_sha256`으로 묶습니다. `generated://` synthetic sample만 `[0,0]`을
기본값으로 쓸 수 있으며, Ink v1의 기존 input·prompt·configuration hash에는 이 필드를
추가하지 않습니다.

실지도 자료 자체는 권리를 확인하지 않은 채 저장소에 넣지 않습니다. 공개 dataset
validator는 8개 독립 도엽, 도엽당 6개 crop, 4/4 calibration/holdout 도엽 분리,
8개 난이도 층, 원본·crop·권리 snapshot hash와 이중 주석 검수를 요구합니다. USGS
퍼블릭 도메인 4개와 항목별 재배포 권리가 명확한 한국·한반도 지도 4개가 모두
준비되기 전에는 `publication_ranking_eligible`을 `false`로 유지합니다.
권리 범위가 제한된 Library of Congress L851 자료와 권리 근거가 불명확한 자료는
source slot에 넣지 않습니다.

## 사전 고정 승격 기준

아래 값은 실제 결과를 본 뒤 바꾸는 목표치가 아니라 locked holdout 전에 적용할
승격 gate입니다.

- Ink v2: holdout 전부 완료, fallback·비결정 결과 0건, failure-adjusted macro
  F1@3px가 v1보다 `+0.02` 이상, 각 난이도 층 저하 `0.01` 이하, 잘못된 평행선 선택
  증가 0건, p95 거리와 break 수 악화 각각 5% 이하, warm cursor traceback p95
  `16ms` 이하, background 계산 중 QGIS input stall 0건
- Smart Recovery: 전체 macro F1@3px가 Ink v2 이상, 사전에 판정한 저신뢰 subset은
  `+0.03` 이상, 새 catastrophic branch switch 0건, 12개 어려운 crop의 반복 QGIS
  추적에서 anchor+undo 중앙값 20% 이상 감소, 모든 오류·취소에서 byte-identical Ink
  champion 유지
- 배포 검증: clean QGIS profile의 model 없음·정상 설치·손상 model, 지원 QGIS CI
  matrix와 ZIP import를 모두 확인

실제 원본·권리 manifest·독립 주석과 위 결과가 모두 준비되기 전에는 ranking이나
stable 기본 승격을 하지 않습니다.

## 현재 의도적으로 남은 경계

- LSD, HED, MobileSAM, SAM과 Legacy Canny 코드는 삭제하지 않고 Advanced/Legacy에서
  기존 index 0–5를 유지합니다.
- 사용자 trace나 지도를 수집하는 telemetry, continual learning과 원격 inference는
  없습니다.
- 커스텀 학습 모델, SAM 3.1과 EdgeTAM은 이번 구현 범위가 아닙니다.
- 실제 48-crop 데이터와 holdout 통과 결과가 없으므로 Ink v2나 Recovery를 새로운
  안정 기본값 또는 성능 우위로 홍보하지 않습니다.
