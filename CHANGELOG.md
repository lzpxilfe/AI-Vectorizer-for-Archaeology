# Changelog

이 파일은 사용자가 체감하는 변경을 기록합니다. 배포 버전의 source of truth는
`ai_vectorizer/metadata.txt`이며, 실제 공개일은 QGIS plugin 저장소나 GitHub Release에
같은 artifact를 게시한 날에만 기록합니다.

## Unreleased

다음 변경은 아직 동결된 `0.1.6` 후보의 metadata·tag·ZIP을 바꾸지 않습니다.

### Added

- Ink v2가 작은 글자 내부 공간을 원본 Byte 영상에서 검출하고, 양쪽 모두 원본의
  실제 끝점일 때 일부 숫자 공백을 `G` 없이 연결합니다. 마스크 밖의 정상 연결은
  보존하며 짧은 경쟁 조각·끝점 동률·이웃 선 교차·보조율 적용 후 접촉을 검사합니다.
  실제 QGIS의 `700`·`600` U자는 일반 클릭으로 저장·Undo했고 기존 7건 × 세 방법의
  21개 저장 좌표 쌍은 같았습니다. 대각선 `600`의 이전 참조는 사용자 지적과 원본
  재검토로 이웃 등고선 연결임을 확인해 성공 판정·점수를 철회했습니다. 해당 쌍은
  자동 연결을 거부하며 올바른 짝은 미확정입니다. 숨은 선 정답이나 일반 성능 검증은
  아닙니다. 재학습·모델·공식 버전은 바꾸지 않습니다.
- Ink v2의 짧고 상호 유일한 공백 후보는 원본 선 조각과 별도의 곡선으로 연결하고
  경로당 추론 공백을 하나로 제한합니다. 원본 연결·회피 힌트·0% 동작을 보존하며
  경쟁 끝점·분기·긴 선 교차는 거부합니다. 문자나 숨은 등고선을 판독하는 기능은 아닙니다.
- `G` 라벨 공백 미리보기에서 거의 수직인 진입·출구와 큰 U자도 부호 있는 실선
  방향으로 연결할 수 있습니다. 가까운 다른 끝점을 건너뛰는 요청과 긴 옆 선을
  가로지르는 곡선은 거부하며, 확정·취소·Undo 흐름은 유지합니다. 복원된 정답이 아닌
  검수할 추정 곡선이며 방향이 불명확하면 기존 미리보기를 유지합니다.
  실제 QGIS의 `700`·`600` U자 미리보기는 저장·Undo했고 옆 끝점 요청은 거부했습니다.
  당시에는 자동 숫자 연결을 해결하지 못했습니다. 대각선 `600`의 당시 양끝 참조는
  후속 사용자 검수로 철회했습니다. 기존 7건의 세 방법은 자동 후보 ON/OFF 저장 좌표가 같았습니다.
- Ink v2에서 두 클릭 사이에 이어지는 원본 중심선 경로를 우선 사용해 깊은 U자를
  짧은 옆 선으로 가로지르던 문제를 수정했습니다. 같은 탐색 창에서 압축 연결 트리를
  준비하며 끊김·과도한 우회·수동/색상/학습 회피·기존 공백 연결은 픽셀 비용 경로를
  유지합니다. 실제 분기 정체성은 보장하지 않습니다. 원본을 다시 감사한 7건의 QGIS
  저장 비교에서 기존 성공 5건을 유지하고 두 U자의 F1@2px가 1.000·0.991이 됐습니다.
- 이전 `kr-10-crop-01-clear-w` 참조는 시작 클릭이 이웃 등고선에 있고 첫 구간이
  종이를 가로지르는 오류가 확인돼 해당 행과 포함 합계를 철회했습니다. 기존 참조·
  결과는 보존하고 별도로 동결한 올바른 클릭을 v3 비교에 사용합니다. 비교 도구는
  철회된 사례를 기본 거부합니다. 사람 승인·독립 성능 평가·재학습은 없습니다.
- 학습 모델에 평가 해시로 결속한 경로 비용 선택 기록이 없으면 등고선 지지만
  사용합니다. 미보정 문자·도로 회피 점수가 실제 등고선을 벌점화하던 기본 비용 6을
  자동 적용하지 않습니다. UI·Live-Wire·Recovery에 같은 설정을 전달하며, 선택 비용이
  있는 모델·수동 회피·색상 힌트는 유지합니다. 유용한 학습 회피도 빠질 수 있습니다.
  당시 Validation 7건의 "3→5건" 합계는 위의 참조 오류로 철회하고 역사 기록으로 보존합니다.
- 원본 Byte Ink에서 긴 저채도 가로·세로 격자 후보를 백그라운드 검출하고,
  Live-Wire에는 따라가는 방향의 유한 비용을, Recovery에는 약한 유한 비용을
  전달합니다. 격자를 가로지르는 등고선을 막지 않으며 두꺼운·회전한·끊긴·유색
  격자는 실패할 수 있습니다. 합성 회귀 시험과 QGIS 저장 비교를 추가했으며,
  당시 4건 중 한 참조는 위의 감사에서 철회했습니다. 격자 단독 정확도 향상 주장은 없습니다.
- 학습 경로를 기존 Ink로 되돌리는 보호와 관측 잉크에 학습 지지를 제한하는
  후보를 분리 진단하고 둘 다 기본 OFF로 남깁니다. 당시 정답 거부 퇴행으로 해석한
  사례는 위의 참조 오류로 그 해석을 철회했습니다. 학습 모델은 기본 OFF를 유지합니다.
- USGS와 AMS 한국 1:50,000 크롭을 합친 약한 라벨 재학습의 Colab T4 세 seed 결과,
  ONNX 내보내기와 Mac CPU/QGIS 비교를 실행 기록에 추가했습니다. 새 모델은 기존 모델이
  회피하던 검정·파랑 합성 선을 등고선으로 지지합니다. 실제 한국 크롭의 같은 클릭
  비교는 기계 기준선이 옆 등고선으로 건너뛰고 일부 클릭이 격자선 위에 있어 철회했으며,
  실제 등고선 추적 개선은 아직 확인하지 않았습니다.
- 검정 인쇄 AMS 한국 1:50,000 도엽의 약한 라벨 데이터 생성 도구
  (`scripts/build_korea_weak_experiment.py`)와 별도 Colab 노트북. 획 기하로 등고선·
  비등고선·미확정 라벨을 만들며 사람 검수를 대신하지 않습니다. 학습 옵션
  `ink_recolor`(등고선 잉크 색 바꾸기)와 `label_gaps`(고도 숫자로 끊긴 선을 이어진
  정답으로 학습)를 추가했고, 둘 다 기본 OFF입니다.
- 약한 라벨 실험의 세 seed Colab T4 학습·ONNX 내보내기·Mac CPU/QGIS 측정 결과를
  실행 기록에 추가했습니다. Validation만으로 seed 3408을 골랐고, 갈색이 아닌
  검정·파랑 등고선은 회피하는 한계를 함께 기록합니다. 모델 파일은 저장소에 넣지 않습니다.
- QGIS clean-profile smoke가 학습 보조 경로의 합성선 이탈 거리와 경고를 결과에
  기록합니다. 이전에는 학습 보조가 선을 벗어나도 `ok`만 남았습니다.

- 사람 검수 파일럿과 분리된 `--allow-weak-labels` 실험 학습·재개·ONNX 내보내기와
  별도 Colab 노트북. 기계 생성 라벨의 출처와 사람 검수 필요 상태를 결속하며,
  기존 승인 요구를 유지하고 기계 라벨 모델의 공식 파일럿 평가·승격을 거부합니다.

- 기본 OFF 로컬 `Learned contour guide`: 원본 격자에 고정된 512px 타일의
  등고선 지지·문자/도로 회피 점수를 Ink Live-Wire와 Recovery 비용에 결합합니다.
  FP32 ONNX·출처 매니페스트를 명시적으로 가져오며 준비·추론 실패 시 Ink를 사용합니다.
  지연 결과는 표시된 미리보기, 대기 클릭, 수동 제안이나 확정된 선을 바꾸지 않습니다.
- QGIS 학습 크롭 주석 편집·검수·내보내기와 USGS 16개 도엽/144개 크롭 파일럿
  생성 도구. 도엽 가족 분할·원본/크롭/수정 이력 해시·검수/ignore 마스크를
  검증하며 자동 초안은 학습 정답으로 승인하지 않습니다.
- Colab 학습 노트북, ResNet18/작은 U-Net 두 logit 헤드, 마스킹 손실,
  optimizer/RNG/진행 위치 재개와 FP32 ONNX 출력 비교 검증.
  Calibration 6개 설정 동결과 별도 CPU Holdout 평가 도구를 제공합니다.
  사람 검수 지도 학습·성능 기준 통과·한국 지도 일반화는 아직 검증하지 않았습니다.
- `TraceGuidance`: line evidence와 분리된 QGIS-independent, immutable soft
  avoidance score 계약. Ink Live-Wire는 선택 영역을 hard wall이나 mask OR로
  쓰지 않고 비용만 높여 더 좋은 중심선 우회로가 있을 때만 선택합니다.
- Ink 추적 중 `Alt`+두 번 클릭으로 문자·기호·얼룩 회피 사각형을 표시하는
  session-only Manual Avoid Guidance. `Alt+Backspace`는 마지막 영역을,
  `Alt+Shift+Backspace`는 모든 영역을 제거합니다.
- Ink 추적 시작 후 마지막 hover 위치에서 `C` 또는 `선 색상 지정`으로 유색
  등고선을 지정하는 선택적 Ink Color Guidance. `Shift+C`로 해제하며, 준비된 native
  Byte RGB Ink v2·SciPy와 `0%`보다 큰 assist가 필요합니다. 회색·배경·모호한 색은
  거부하고 선택한 RGB를 표시합니다. 실제 지도 정확도·작업시간 개선 주장은 없습니다.
- 색 있는 등고선이 숫자 라벨 때문에 실제로 끊긴 경우를 위한 anchor-bound
  continuity bridge. 시작 anchor와 양쪽 centerline endpoint가 같은 비중성 색·방향을
  보일 때에만 Live-Wire 내부의 연속 score를 보강하며, `LineEvidence.centerline`과
  0.1.5 Ink 경로는 변경하지 않습니다.
- 검은·갈색 등고선의 숫자 공백에 쓰는 명시적 `⤴ 라벨 공백 연결` preview. 공백 전
  anchor를 확정하고 반대편 선에 커서를 둔 뒤 사용자가 버튼과 같은 끝점의 두 번째
  클릭으로 채택해야 하며, 이미지 분석·OCR·model 없이 두 사용자 endpoint와 local
  tangent만으로 제한된 Hermite segment를 만듭니다.

### Safety

- Colab에서 학습 패키지를 설치·교체한 뒤 이전 Pillow·NumPy 모듈이 남아 있으면
  학습 전에 중단하고 세션 재시작을 요구합니다. 기계 라벨 GPU 실행과 Drive
  체크포인트 확인 기록은 사람 검수·공식 등고선 성능 평가와 분리합니다.
- 학습 주석 버튼에서 생성한 래스터·5개 주석 레이어를 실제 프로젝트·지도 화면에
  추가하고, 영역은 투명한 윤곽으로 표시합니다. 승인한 평가선은 선택한 QGIS feature를
  주석 ID로 변환하며, 미승인 초안은 선택한 선이 있어도 평가 target을 만들지 않습니다.
  실제 dock·canvas·GeoPackage·JSON 내보내기 경로에서 확인했습니다.
- DEM·hillshade와 sidecar의 승인 시 파일 상태를 보관하고, 계산 중 생성·수정·교체·삭제된
  출력이 있으면 게시를 중단합니다. 기존 선 연장도 시작 시 기하와 달라지면 저장하지
  않고 확정 앵커와 사용자 편집을 보존합니다.
- 검출 배열 좌표와 지도 좌표를 픽셀 중심 기준으로 통일해 반 픽셀 위치 오차를 수정하고,
  회피 영역도 같은 기준을 사용합니다. Recovery 후보에는 보조 강도를 적용한 뒤 안전성을
  평가하며, 평가·미리보기·확정에 같은 경로를 사용합니다. benchmark worker도 반영했습니다.
- 라벨 공백의 바깥쪽 방향 추정에서 분리된 잡점·짧은 선 조각·가까운 평행선을
  하나의 방향으로 평균내던 경우를 차단했습니다. 가장 가까운 8-connected 중심선
  하나만 읽고, 비슷한 거리의 다른 선이나 지속적인 분기가 있으면 기존 Ink를 유지합니다.
  기존 숫자 인접 성공 사례와 스캔 누락·분기·회전 부정 사례를 제품 경로에서 검증합니다.
- QGIS 4/PyQt6에서 test event가 반환한 정수 modifier와 scoped `KeyboardModifier`
  enum을 직접 비트 연산하던 오류를 수정했습니다. 이제 마우스·키보드 단축키와
  manual avoidance는 Qt5 정수와 Qt6 enum/QFlags를 같은 정수 mask로 비교합니다.
- 라벨 공백 연결의 제품·benchmark 방향 추정을 통일했습니다. 가까운 선 방향의 충돌과
  공백에 수직인 방향을 거부하며, 종전 고정 tangent 기하 점수를 제품 성능과 구분합니다.
- 숫자 획이 3px 방향장을 오염시켜도, 두 명시적 anchor에서 공백 반대편으로 이어지는
  12px one-sided centerline 지지가 충분할 때만 방향을 다시 읽습니다. 숫자 쪽 공백은
  탐색하지 않고, 가까운 지지·4px 이상 span·한 방향성·45° chord 정렬을 모두 요구하므로
  지지가 부족하거나 평행선만 보이면 기존 Ink를 유지합니다.
- 수동 미리보기를 늦은 Live-Wire 결과가 덮어쓰던 경합을 수정했습니다. 확정은 같은
  cache·anchor·표시 경로에서 화면/원본 2px 안의 끝점 클릭만 허용합니다. `G`로 요청하고
  `Esc`로 미리보기만 취소하며, hover·Alt 선택은 미리보기를 확정하지 않습니다.
- 수동 bridge에도 assist 강도를 적용하고, 지도 좌표 변환 실패와 cache 밖 곡선에서
  기존 preview를 보존합니다.
- manual guidance는 Freehand/0% assist/SAM 경로를 바꾸지 않습니다. Recovery를 켠
  경우에도 같은 immutable soft cost를 challenger corridor에 전달하며, model mask를
  선으로 저장하거나 hard exclusion으로 쓰지 않습니다. 지도·selection은 저장·업로드·
  telemetry로 전송되지 않습니다.
- 선 색상은 원본 evidence를 바꾸지 않는 유한 soft mismatch cost이며, 회피 guidance와
  픽셀별 최댓값으로 합쳐 Ink와 Recovery에 같은 snapshot을 전달합니다. pan 뒤 새
  cache에서 다시 적용하고 추적 종료·취소·reset·raster 데이터 변경에서 지우며,
  표본을 저장·업로드하지 않습니다. 같은 색의 글자·기호는 여전히 사용자 검수가 필요합니다.
- continuity bridge는 grayscale/neutral anchor, 색 불일치, 방향 불일치, 12px보다
  짧거나 96px보다 긴 공백, 복잡하게 갈라진 색 endpoint에서 동작하지 않습니다. 이때
  기존 Ink champion을 그대로 유지하며 virtual support를 이진 mask와 OR하지 않습니다.
- manual gap bridge는 3–128px 사이의 명시적 endpoint만 받고, local tangent를 읽지
  못하거나 detour가 1.25배를 넘으면 preview조차 만들지 않습니다. 저장은 같은
  endpoint를 다시 클릭해 사용자가 green preview를 확인한 뒤에만 일어납니다.

## 0.1.6 — experimental release candidate (not yet published)

2026-08-31에 현재 Ink v2·Smart Recovery 소스의 metadata를 `0.1.6`으로
정렬했습니다. 이는 설치·검증을 위한 후보이며 아직 QGIS plugin 저장소, GitHub
Release 또는 tag로 게시한 릴리스가 아닙니다. 공개일은 정확히 같은 ZIP을 실제로
게시할 때만 추가합니다.

### Added

- 9·15·31 source-pixel, RGB/명도, source-grid tile 정규화를 사용하는 연속
  `LineEvidence`와 이를 직접 소비하는 Ink Live-Wire v2
- 기본 OFF의 `Smart Recovery (Experimental)`: 검증된 EfficientSAM-Ti mask를
  저신뢰 Ink 구간의 corridor prior로만 사용하고, 안전 검사를 통과한 challenger만 채택
- `Ink`, `Recovering`, `Enhanced`, `Ink fallback` 상태, 명시적 model 설치와 현재
  구간 재시도 UI
- worker request v2의 선택적 `previous_xy`, 네 개의 독립 benchmark method ID와
  8개 도엽·48개 crop 공개 dataset 권리·주석 검증 template
- USGS HTMC public-domain 원본 1개와 source-pixel 그대로의 512px PNG crop 6개,
  immutable provenance·권리 snapshot·draft 중심선을 첫 공개 benchmark 묶음으로 추가

### Changed

- LSD, HED, MobileSAM, SAM과 Legacy Canny를 접힌 `Advanced / Legacy methods`로
  이동하되 기존 model index 0–5와 backend·사용자 model 파일은 유지
- raster provider read는 main thread에 남기고 Ink evidence, Recovery model
  hash/session 준비와 추론을 취소·generation·source identity를 검사하는 background
  task로 분리
- Ink v2는 block stretch 대신 native integer source DN과 합산 31px context를
  사용하며, 고정 range가 없는 float source에서는 사유를 표시하고 Ink v1을 유지
- v2용 확장 block과 0.1.5 호환 visible-extent fallback cache를 분리해, v2 실패가
  기준선의 mask나 map transform을 바꾸지 않도록 고정
- Recovery 입력을 native Byte raster로 제한하고 UInt16 등 더 넓은 정수형은 Ink v2를
  유지하도록 변경; raster `dataChanged` 때 stale evidence와 challenger를 즉시 폐기

### Safety

- model/runtime/hash/shape/inference/cancel/stale 실패에서 다른 backend로 조용히
  전환하지 않고 계산 전과 동일한 Ink champion을 유지
- model 설치 취소를 network read loop까지 전달하고, 이미 `Enhanced`인 preview를
  새 Ink champion으로 재사용하는 반복 Recovery를 차단
- 다음 anchor의 Live-Wire tree가 준비되기 전 빠른 클릭이 임시 직선 chord를 저장하던
  race를 막고, 계산된 Ink 경로와 필요한 Recovery quality gate가 끝난 뒤 같은 click
  target을 확정
- Recovery inference 실패 뒤 남던 stale request를 제거해 같은 Ink champion에서
  명시적으로 다시 시도할 수 있도록 수정
- 손상된 Recovery regular object는 사용자가 `Repair Recovery Model`을 명시적으로
  누른 경우에만 격리·재다운로드하고, 실패·취소 시 미완료 object는 원본을 복원하되
  이미 hash 검증·게시된 replacement는 유지하며 symlink·directory·Windows junction/
  reparse point 등 unsafe cache는 자동 변경하지 않도록 수정
- Recovery install/repair task 등록 실패는 버튼·상태를 즉시 복원하고, 마지막 artifact가
  이미 검증·게시된 뒤 도착한 cancel flag는 성공한 store transaction을 취소로 오표시하지
  않도록 commit boundary를 고정
- Ctrl+Z와 checkpoint rewind가 진행 중 Recovery·Live-Wire·pending click을 함께
  무효화하고, polygon close는 elevation 취소·layer 저장 실패에도 확정 경로와 화면의
  Enhanced 후보를 바꾸지 않는 transaction으로 처리
- 개발 source의 기본 package 명령이 동결된 `0.1.5` ZIP을 덮어쓰지 못하도록 하고
  CI는 격리된 `--output` current-source ZIP만 생성·검사
- QGIS Plugins Website의 Bandit `B110`/`B112` false-positive 대상인 best-effort
  cleanup·fallback에 지점별 suppression 근거를 기록하고, model SHA-256·upstream
  commit pin을 credential과 구분하는 검토된 `.secrets.baseline`을 current-source
  package에 포함; 같은 범위를 고정 Bandit·detect-secrets CI로 재검사
- 실제 재배포 권리 자료와 독립 주석 검수가 완료될 때까지 benchmark의
  `publication_ranking_eligible`을 false로 고정하고, 시작·끝점이 같거나 기하 길이가
  0인 prompt/reference는 hash가 맞아도 거부

## 0.1.5 — experimental QGIS release (2026-08-26)

`0.1.5`는 공식 QGIS plugin 저장소에 experimental release로 공개됐습니다.
공식 다운로드는 1,483,635 bytes, 30 entries, SHA-256
`24f1def6acd63d483ea6bf7c20b944f56507ead52190667ec4f35562fca6c964`입니다.
29개 일반 plugin entry는 commit `0675cad`와 byte-identical이지만, upload ZIP의
`.secrets.baseline`은 해당 commit blob과 다릅니다. repository에 보존된
`dist/ai_vectorizer-0.1.5.zip`(SHA-256 `d2925198…`, 29 entries)은 commit
`89b9f20`의 이전 로컬 후보이며 공식 artifact identity가 아닙니다.

Git history의 `0.1.5–0.1.7`과 미공개 worktree의 `0.1.8` 표시는 추가 QGIS
릴리스가 아닌 개발 metadata였습니다. 이력은 rewrite하지 않으며,
Ink v2·Smart Recovery와 공개 이후 수정은 위 `0.1.6` 후보에서 관리합니다.

### Added

- 검은 인쇄 획을 한 픽셀 중심선으로 만드는 기본 Ink Centerline과 제한된
  direction-aware Live-Wire
- literal 0–100% assist, green path preview, anchor undo와 Freehand baseline
- contour elevation, Spot Heights, 실험적 linear-TIN DEM과 GDAL hillshade workflow
- 검증된 MobileSAM/SAM/HED model storage와 상태 진단
- 최종 ordered centerline을 평가하는 격리 benchmark 및 EfficientSAM-Ti ONNX
  benchmark adapter
- 결정적 release builder, Python/QGIS CI matrix와 QGIS runtime safety tests

### Changed

- 새 feature와 기존 contour 연장, elevation field·attribute 변경을 QGIS edit buffer와
  하나의 Undo command로 통합
- model을 plugin 폴더 대신 QGIS profile에 보존하고 고정 URL·크기·SHA-256,
  staged write, atomic publication과 rollback 적용
- 기본 ZIP은 추가 pip 없이 Ink/Freehand를 유지하고 OpenCV·SAM stack을 선택
  dependency로 분리
- release 버전을 `0.1.5`로 정규화하고 문서를 기능·구현·안전·검증 중심으로 재구성

### Fixed

- QGIS에 로드된 Shapefile과 같은 경로를 새 output으로 선택할 때 발생할 수 있던
  dataset 교체와 data loss 차단
- DEM/hillshade pair 게시 중 path race, 기존 output 변경과 partial result rollback
- model download·migration 실패 시 기존 checkpoint 손상과 symlink target 우회
- raster block의 과대한 allocation, 잘못된 dtype 해석, trace cost overflow와 NaN 입력
- plugin unload, layer removal, dialog 종료 뒤 남을 수 있던 task, timer, rubber band와
  preview resource 정리

### Known limitations

- 완성·독립 검수된 역사 지도 benchmark dataset과 제품 정확도 근거는 아직
  없습니다.
- topology QA, DEM uncertainty/NoData와 provenance sidecar는 아직 구현되지 않았습니다.
- 공식 0.1.5 source 시점의 [CI run 32924318782](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/32924318782)은
  QGIS 3.22/4.2 runtime job이 green이 아니었습니다. 후속 commit `30e18f6`의
  [run 33339770178](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/33339770178)에서
  3.22.16/3.44.13/4.2.1 current-source 행렬이 통과했지만 이는 공식 ZIP을
  소급해 다시 검증한 것이 아닙니다.
- Windows clean-profile GUI, `0.1.4 → 0.1.5` upgrade와 역사 지도 독립 검수는
  stable 전환·다음 release 전 남은 gate입니다. QGIS 3.22/3.44/4.2 원격
  matrix와 macOS 3.44.8 clean-profile current-source smoke는 통과했습니다.

## 0.1.4 — previous QGIS repository baseline

이 저장소 정리의 기준이 된 직전 QGIS plugin 저장소 버전입니다. 이전 변경의 세부
내용은 Git history와 QGIS plugin 저장소의 version record를 참고하세요.
