# Claude Haiku 5.5 Holter train 전체 추론

## 대상 및 방법

- MonoRouter `~anthropic/claude-haiku-5-5`, train 원본 285명, SCD 양성 32명·음성 253명. validation과 test는 추론하지 않았다.
- 기존 MonoRouter Holter 노트북의 RR/HRV 요약과 시스템 프롬프트를 사용했다. 정상박동 주석 없는 RR/HRV는 검증되지 않은 추정치다. 환자 ID와 정답은 API로 전송하지 않았다.
- 모델을 환자 데이터로 학습시키지 않았다. train 대상 후향적 결과이며 독립 검증 성능이 아니다. 모델 위험도 출력은 보정된 실제 확률이 아니다.
- 모델 요청 성공 285/285, temperature는 기존 노트북의 API 기본값, max_completion_tokens 1024, reasoning_effort low. 두 키 HTTP 402 전환을 지원한다. 순수 JSON과 JSON 코드 블록을 동일 스키마로 검증했다.

## 성능

| 지표 | 결과 |
|---|---:|
| AUROC | 0.5259 |
| AP | 0.1141 |
| 민감도 (50% 기준) | 0/32 = 0.0% |
| 특이도 (50% 기준) | 253/253 = 100.0% |
| 위양성 / 위음성 | 0 / 32 |
| 점수 범위 | 0.30% ~ 3.00% |

## 분포도와 임계값

![실제 양성·음성별 점수 분포](../results/monorouter_holter/20261103T024831Z_d69f199a/score_distribution_zoom.png)

파랑은 실제 음성 253명, 빨강은 실제 양성 32명이며 각 집단 내 비율을 보여준다. 점수 0~5% 구간을 0.25%p씩 확대했다. 기존 50% 기준은 표시 범위 밖이다. [구간별 인원 CSV](../results/monorouter_holter/20261103T024831Z_d69f199a/score_distribution_zoom_bins.csv), [임계값 CSV](../results/monorouter_holter/20261103T024831Z_d69f199a/threshold_summary.csv).

민감도 80%·90% 행은 같은 train 환자에서 사후 선택된 기준이며 동점으로 목표를 넘을 수 있다. 별도 검증이나 임상 임계값이 아니다. GPT-6-Luna의 기존 356명 결과는 validation 71명도 포함해 직접 수치를 비교할 수 없다.

## 결과 및 검증

- 환자별 예측 CSV: `Members/sang/results/monorouter_holter/20261103T024831Z_d69f199a/_anthropic_claude-haiku-5-5_09f2d44e/predictions.csv`. 식별자와 라벨이 포함되므로 접근을 제한한다.
- 실행 노트북: `Members/sang/notebooks/monorouter_holter_claude_haiku55_train.ipynb`.
- 원본 환자·예측 중복, 285명, 실제 양성 32명, 점수 범위와 성공 상태, 독립 재계산 AUROC·AP·혼동행렬 일치, 히스토그램 합계를 확인했다.
- 지표는 연구용이며 임상 의사결정에 사용할 수 없다.

최초 요청에서 시간 초과 3건, 응답 검증 실패 6건이 발생했고 성공한 276건은 재사용했다. 실패 9건만 재시도해 모두 성공했다. 최초 결과와 재시도 결과는 별도 폴더에 보존했다.
