# SCD ML·DL·LLM 실험별 예측 분포와 CSV 목록

## 결과 열기

- [실험별 분포도 전체 보기](../results/experiment_inventory_20261103_final_v2/index.html) — 실험 원본 경로, 분할, 조건을 제목에 표시한 목록. 차트는 SVG 파일로 개별 보관한다.
- [점수 구간별 집계 CSV](../results/experiment_inventory_20261103_final_v2/histogram_bins.csv) — 5%p 구간의 실제 양성·음성 인원. 100% 점수는 마지막 구간에 포함한다.
- [결과 지표 CSV](../results/experiment_inventory_20261103_final_v2/metrics_long.csv) — 확인된 실험 지표 JSON 225개에서 변환한 `source, field, value` 형식이다. 숫자 외 원본 설명·설정값도 포함한다.
- [파일 처리 목록 CSV](../results/experiment_inventory_20261103_final_v2/inventory.csv) — 실험별 원본 경로, 분할, 유효 건수, 제외 건수, 처리 상태를 확인한다.

## 범위와 결과

`Members/sang/results`, `Members/sang/model`과 Furiosa 서버 `/root/.cache/std3_holter/llm_compare`에서 정해진 점수·라벨 열과 환자 ID가 있는 예측 CSV를 찾았다. 기존 파일을 수정하지 않고, 서버의 환자별 예측값은 서버에 둔 채 구간별 집계만 로컬로 복사했다. train·validation·test, 입력 조건, 별도로 기록된 외부 fold와 calibration/evaluation 역할은 분리했다. 242개 실험·분할·조건의 분포도를 생성했다. 여러 실행이 동일한 모델이나 환자 자료를 공유할 수 있으므로 242개를 독립 데이터셋이나 서로 다른 모델의 수로 해석하지 않는다.

| 원본 위치 또는 실험 | 분포도 수 |
|---|---:|
| 임상 ML 기본 및 파라미터·버전 실험 | 85 |
| RunPod ECG 딥러닝 | 60 |
| DeepECG·ECGFounder·ECG-FM | 21 |
| 기존 MonoRouter·Furiosa 파일 | 11 |
| Furiosa 서버 기존·신규 LLM 및 fold 분석 | 41 |
| 기존 model 폴더의 ML·DL·fusion 및 동작 확인 | 24 |
| **합계** | **242** |

이는 위 세 위치와 인식 가능한 CSV 형식에 대한 목록이다. 노트북 출력만 있거나 다른 저장 위치·열 이름을 사용하는 모든 과거 실험이 포함됐다고 단정할 수 없다. 일부 파일명에 smoke 또는 check가 있는 11개 그래프는 동작 확인 결과로, 성능 실험으로 해석하지 않는다. 별도 실험 및 연구 문서는 `Members/sang/docs`와 기존 실험별 결과 디렉터리에서 확인한다.

`metrics_only` 22건은 **해당 JSON과 같은 폴더에** 사용할 수 있는 예측 CSV를 찾지 못한 것이다. 예측 파일이 다른 폴더나 서버에 존재할 가능성이 있으며 전체 저장소의 영구적 부재를 뜻하지 않는다. 예를 들어 `ecg_fm_pilot`과 `ecg_fm_adapter`의 동일 폴더에서는 지표만 발견했다. 지표만으로 환자별 예측이나 분포를 역산하지 않았다. 성공한 예측이 없어 분포도를 만들 수 없는 CSV도 2건이며 상세 이유는 목록에 기록했다. 성공하지 않은 예측 등 8행은 유효 그래프에서 제외했다.

## 그래프 읽는 방법

파랑은 실제 음성, 빨강은 실제 양성이다. 각각 자기 집단의 수를 분모로 정규화했으므로 높이는 **집단 내 비율**이지 전체 인원이 아니다. 가로축은 모델 점수, 점선은 50% 기준이다. 동점·점수 집중과 양성/음성 분포의 겹침을 볼 수 있다. 대상 환자와 테스트 분할, 입력·실험 조건이 서로 다르므로 그래프만으로 모델을 직접 순위화하지 않는다. LLM 점수는 보정된 실제 SCD 위험 확률이 아니며 임상 적용 기준도 아니다.

## 검증 및 재현

- 분포별 환자 ID 중복, 점수 범위·유한성, 이진 라벨을 확인했다. 각 히스토그램에서 구간별 양성·음성 인원 합이 해당 집단 수와 일치하는지 확인했다.
- SVG 242개의 XML 구문을 확인했다. 각 CSV는 원본 경로를 보존하며 JSON 수치는 원본 지표의 평탄화 결과다.
- 재현 코드: `Members/sang/experiments/experiment_inventory/inventory.py`. 서버 결과는 `--aggregate-only`로 생성하고, 로컬에서 `--extra-bins`, `--extra-manifest`, `--extra-metrics`로 합친다.
- 환자별 예측 원본 일부에는 환자 ID와 정답이 포함된다. 산출 집계는 환자 행을 포함하지 않지만 소수 환자 집단의 세부 집계가 있으므로 공유 전 접근 범위를 확인한다.

## 기존 결과 문서와 재대조 (2026-11-03)

- GPT-6-Luna 개발 환자 356명 결과는 기존 목록에 포함돼 있다. [해당 분포도](../results/experiment_inventory_20261103_final_v2/plots/bcc81c47d740137f7814.svg)와 [결과 문서](holter_gpt6_luna_development_results.md)를 직접 연결한다. 별도 20명 예비 실험과 중복해서 독립 환자군으로 해석하지 않는다.
- [GPT 예비 실험](holter_gpt_4model_pilot_results.md)의 GPT-5.4-mini 18건 성공, GPT-5.4·GPT-5.5·GPT-6-Luna 각 20건 성공 결과가 분포도 목록에 있다. 각 그래프는 성공한 환자만 표시하며 공통 환자 18명으로 재계산한 그래프는 아니다.
- [Claude 예비 실험](holter_claude_4model_pilot_results.md)은 Claude Sonnet 5 성공 13건이 목록에 있다. 나머지 세 후보는 기록상 성공한 예측이 없어 그래프가 없다.
- [RunPod DL 60 trial](runpod_ecg_60_experiments_20261101.md)은 trial별 validation 예측 분포 60개가 목록에 있다. test 평가는 시행되지 않았다.
- 기존 Qwen3-32B 및 GPT-OSS·Qwen3-30B·EXAONE 추가 실험은 각 입력 조건의 서버 예측 결과가 목록에 있다. 환자별 서버 원본은 로컬로 복사하지 않았다.
- ECG-FM pilot과 adapter는 해당 결과 폴더에 지표 JSON과 보고서가 있지만 환자별 예측 CSV는 확인하지 못했다. JSON 지표는 통합 CSV에 포함되며 점수 분포는 생성하지 않았다. 예측값이 다른 위치에 존재하거나 모델 재실행으로 다시 계산할 수 있는지는 이번 대조로 확정하지 않았다.
- 결과가 없는 호출·실패 건과 시험용 smoke/check 실행을 실제 성능 실험으로 합산하지 않는다. 이번 확인은 `Members/sang`의 결과·모델·노트북 및 Furiosa 서버의 해당 작업 디렉터리를 대상으로 했으며, 외부 장비와 삭제된 자료까지 전수한 것은 아니다.

## ECG-FM 초기 실험의 추가 CSV

- [ECG-FM pilot 분할별 지표 CSV](../results/ecg_fm_pilot/split_metrics_20261103.csv)
- [ECG-FM adapter 분할별 지표 CSV](../results/ecg_fm_adapter/split_metrics_20261103.csv)

기존 JSON에 저장된 집계 지표를 그대로 변환했다. 환자별 예측값은 확인되지 않았으므로 점수 분포도는 생성하지 않았다. 결과 재현 코드: `Members/sang/experiments/experiment_inventory/metric_only_reports.py`.
