# 지정 임상 데이터로 Qwen3-32B 비교 재실행 (연구용)

이 기록은 사용자가 지정한 `Members/sang/datas/clincal/clinical_clean.csv`와 `Members/sang/datas/clincal/final_split.csv`를 서버의 `std3` 전용 경로에 복사하고, 그 서버 파일을 직접 읽어 구성한 실험이다. 앞선 `model/ML/data` 소스 사용은 잘못된 입력이었다.

## 데이터·실험 방법

- 임상 CSV SHA-256: `d00bf5da71fecf65132c82182365d3e4af3adee917ab32b839d06255cf60982d`; 분할 CSV SHA-256: `7dfb515e9753027ecbf31d4f05e5780fdd90e3a5cc25911c0c9345ae26e1b059`. 서버 파일과 생성 입력의 해시를 대조했다.
- 원본 train 285, validation 71, test 89. 지정 임상 데이터와 Holter 분할의 환자·라벨을 검증했으며 개발 환자의 분할 불일치·라벨 불일치·누락은 0건이다. test는 모델 입력이나 사례에 사용하지 않았다.
- Holter 특징은 이전에 계산한 캐시 356명분을 코드·파형 메타데이터·품질 인덱스·설정 기준으로 검증해 서버로 전달했다. 원시 파형 전체를 이번에 다시 계산하지 않았다. 임상 데이터 및 분할을 포함한 입력 구성은 서버에서 실행했다.
- Holter train 285명으로 임상 변수 23개를 표준화하고, validation 환자별 거리가 가까운 train 양성 2명과 음성 2명을 사례로 선택했다. 네 사례는 발생률을 대표하지 않는다. 검증 환자 ID와 정답은 요청에 넣지 않는다. 사례 ID가 모두 train 소속임을 확인했다.
- 이전과 동일한 Qwen3-32B-FP8 스냅샷, RNGD 4장, temperature 0, thinking 비활성, JSON schema, 3개 입력 조건을 사용했다. 임상 측정 시점의 예측 시점 이전 여부는 미확인이다. 점수는 연구용이며 임상 확률로 검증되지 않았다.

## 결과

validation 71명(양성 8, 음성 63), 조건별 71/71건이 성공했다. 0.5 기준 혼동행렬 순서는 `[[TN, FP], [FN, TP]]`다.

| 조건 | AUROC | AP | 혼동행렬 |
|---|---:|---:|---|
| Holter | 0.274 | 0.093 | [[56, 7], [8, 0]] |
| Holter + 임상 | 0.670 | 0.200 | [[63, 0], [8, 0]] |
| Holter + 임상 + 사례 | 0.713 | 0.389 | [[63, 0], [7, 1]] |

앞선 파일과 지정 파일 사이에 validation 임상값·라벨은 같았으나, train에서 1명의 임상 변수와 2명의 라벨이 달랐다. 이로 인해 사례 구성은 validation 2명에서 바뀌었지만 213개 모델 점수와 집계 지표는 이번 재실행에서 모두 동일했다. 작은 validation 표본, 기존 반복 사용, 추론 모드 비활성 및 원시 파형 대신 추정 특징 사용을 고려해야 한다. 성능 향상이나 임상 적용을 확정하지 않는다.

서버 입력: `/root/.cache/std3_holter/llm_compare/clinical_source_20261102/`; 결과: `/root/.cache/std3_holter/llm_compare/run_20261102_clincal/`; 개인별 결과와 임상 원본은 비공개로 보관한다. 로컬 집계: `Members/sang/results/furiosa_llm_compare/run_20261102_clincal/`. 재현 코드는 `export_holter.py`, `prepare_server.py`, `run_server.py`다.
