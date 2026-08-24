# Security policy

ArchaeoTrace는 raster, vector layer, model artifact와 연구 output을 다루므로 일반적인
crash뿐 아니라 조용한 geometry 변경, dataset overwrite와 오해를 부르는 DEM도 안전
문제로 취급합니다.

## Reporting a vulnerability

다음 문제는 공개 issue를 만들기 전에 `lzpxilfe@gmail.com`으로 알려 주세요. 제목에
`[ArchaeoTrace security]`를 넣으면 구분하기 쉽습니다.

- 의도하지 않은 file overwrite, geometry·attribute 손실이나 Undo 우회
- path alias, symlink, archive나 output publication을 통한 경로 이탈
- model source·size·SHA 검증, download, migration 또는 rollback 우회
- 사용자가 시작하지 않은 network 요청이나 raster/vector data 전송
- 악성 raster/model/manifest로 인한 과대 allocation 또는 code execution
- plugin unload, layer removal과 task callback 사이의 use-after-free 성격 문제
- 진단 report에서 예상하지 못한 credential·민감 데이터 노출

보고서에는 영향, 최소 재현 단계, 영향을 받는 commit/plugin/QGIS version과 가능한
경우 non-sensitive fixture를 포함해 주세요. 실제 현장 지도, credential, personal
path나 조직 내부 URL을 보내지 마세요. 암호화된 전달이 필요하면 첫 메일에는 민감한
내용을 넣지 말고 안전한 채널을 먼저 협의해 주세요.

maintainer는 best-effort로 수신을 확인하고 재현 범위, 임시 완화책과 공개 시점을
협의합니다. 이 개인 오픈소스 project는 고정 응답 시간을 보장하지 않지만 data-loss와
검증 우회는 우선순위가 높은 문제로 다룹니다.

## Supported versions

보안 수정은 현재 `main`과 공식 QGIS plugin 저장소의 최신 공개판을 우선합니다.
`0.1.5` source는 아직 experimental 후보이고 `0.1.4`가 이 정리 시점의 공식 baseline입니다.
미공개 개발판 `0.1.6–0.1.8`은 별도 지원 line이 아닙니다. 오래된 ZIP에서 문제가
재현되면 먼저 최신 검증 artifact로 재현 여부를 확인해 주세요.

## Diagnostic-data warning

`SAM Status Report`는 clipboard에 복사되며 다음 값을 포함할 수 있습니다.

- current working directory
- QGIS prefix와 Python executable/version
- `PYTHONPATH` 같은 environment 값
- model weights와 metadata file path
- dependency 및 system 상태

공유 전에 전체 JSON을 읽고 username, home/project/network path, token·key가 섞인
environment 값과 조직 정보를 삭제하세요. bug report에 원본 지도나 project file을
공개 첨부하지 말고 재배포 가능한 최소 fixture를 사용하세요.

## Scope and limitations

- 기본 trace inference와 geometry 처리는 local입니다. HED/SAM download와 local
  SAM file이 없을 때의 pinned-source availability check는 사용자가 시작하는 network
  작업입니다.
- artifact hash는 기대한 file identity를 확인하지만 upstream source가 악의적이지
  않다는 보증이나 native dependency 취약점 검사를 대신하지 않습니다.
- DEM/hillshade는 reviewable hypothesis입니다. 현재 topology QA,
  uncertainty/NoData와 provenance sidecar가 없어 과학적 오해의 위험이 남습니다.
- QGIS/Python/native library와 선택 backend도 각 upstream의 security support 범위에
  따릅니다. 가능하면 지원 중인 QGIS/Python을 사용하고 profile을 백업하세요.
