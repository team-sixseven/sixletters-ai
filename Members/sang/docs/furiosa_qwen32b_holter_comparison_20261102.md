> 정정: 이 문서의 첫 실행은 사용자가 지정한 임상 원본이 아닌 `model/ML/data`의 다른 파일을 사용했다. 사용자가 지정한 `datas/clincal` 파일과 분할로 재실행한 최종 결과는 [수정 실험 기록](furiosa_qwen32b_clincal_comparison_20261102.md)을 참고한다. 아래는 첫 실행의 이력이다.

# Qwen3-32B Holter·임상·학습 사례 비교 (연구용)

## 목적과 조건

2026-11-02 Holter validation 원본 71명에서 같은 특징을 사용하여 Qwen3-32B-FP8의 세 입력 조건을 비교했다. Furiosa RNGD 4장, furiosa-llm 2026.4.0, PyTorch 2.10.0+cpu, 모델 스냅샷 `b33946f6cba6502114f12603c6caa4529cf396df`를 사용했다. Temperature 0, thinking 비활성, 최대 출력 1024 토큰, JSON schema로 출력을 제한하고 0~100 숫자를 별도 검증했다. 기존 제한 없는 20명 LLM 실험과 생성 조건이 다르다.

- Holter: 기존 전체 기록 RR·HRV 및 감속능 추정치. [계산·검증 한계](holter_llm_feature_method.md).
- Holter+임상: 동일 특징에 임상 23개 변수(`MODEL_4`) 추가. 임상 변수의 실제 기록 시점이 Holter 기준 시점 이전인지 미확인이다.
- Holter+임상+사례: 추가로 Holter train에서 임상변수 표준화 거리 기준 가까운 음성 2명·양성 2명을 제공한다. 표준화는 Holter train으로만 적합하며 사례는 예측 환자와 분리된다. 균형 사례는 실제 발생률을 반영하지 않는다. validation 정답과 환자 ID는 요청에 포함하지 않는다.

기존 Holter train 285명 중 임상 CSV와 SCD 라벨이 충돌한 2명을 사례 후보에서 제외했다. validation 71명은 누락이나 라벨 충돌이 없었다. 임상 ML의 과거 분할과 Holter 분할은 다르며 Holter validation 71명 중 임상 분할도 validation인 환자는 9명뿐이다. 과거 임상 ML과 독립된 홀드아웃이라고 주장할 수 없다. 기존 test도 이미 사용됐으므로 독립 최종 검증이 아니다.

## 결과

양성 8명·음성 63명, 세 조건 모두 71/71 응답의 범위와 스키마 검증을 통과했다. 특이도·민감도는 정해둔 점수 0.5 기준이다.

| 조건 | AUROC | AP | 민감도 | 특이도 |
|---|---:|---:|---:|---:|
| Holter | 0.274 | 0.093 | 0/8 | 56/63 |
| Holter+임상 | 0.670 | 0.200 | 0/8 | 63/63 |
| Holter+임상+사례 | 0.713 | 0.389 | 1/8 | 63/63 |

환자 단위 3,000회 비층화 bootstrap(seed 42, 한 클래스만 표본화된 반복 제외)의 AUROC 95% 분위 구간은 각각 [0.110, 0.476], [0.445, 0.855], [0.464, 0.939]였다. 짝지은 `사례 추가 - 임상만` AUROC 차이 구간은 [-0.121, 0.198]이다. 표본 및 양성 수가 적으므로 예시 추가로 인한 개선을 확정할 수 없다. 0.5는 임계값을 최적화한 결과가 아니며 높은 정확도는 불균형 데이터에서 양성을 놓치는 상황을 가릴 수 있다. 예측 점수는 확률 보정되지 않았고 진료에 사용할 수 없다.

## 재현과 산출물

입력: `Members/sang/furiosa/furiosa_llm_compare/prepare.py` (로컬 기존 특징 캐시의 코드·경로·설정·메타데이터 무결성 키를 확인). 서버 실행: `run_server.py`; 모델·조건·입력 SHA-256·설정은 결과 metadata에 저장. 서버 결과 및 노트북: `/root/.cache/std3_holter/llm_compare/run_20261102/`와 `/root/.cache/std3_holter/llm_compare/furiosa_qwen32b_comparison.ipynb`. 환자별 CSV와 입력에는 개인 정보가 있으므로 비공개로 보관한다. 집계 JSON 사본: `Members/sang/results/furiosa_llm_compare/run_20261102/`.

남은 일: 임상 측정 시점·SCD 라벨 불일치 원인 확인, 사례 선택 및 임상 변수 제공의 사전 정의, 독립 환자 집단에서 확률 보정 및 외부 검증. 이번 validation 결과를 보고 개선을 반복하면 해당 validation도 선택에 사용된 데이터가 된다.
