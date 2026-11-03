# Holter 데모 연결용 LLM 입력·출력 초안

## 범위

기존에 검증한 200Hz Holter의 **선택한 30초 구간 하나**에서 RR 요약을 추출해 Furiosa NPU의 4개 LLM에 보내는 연구용 계약이다. 데모 프론트·백엔드 소스와 실제 모델 API 계약이 아직 없으므로 아래 형식은 **제안 초안**이며 데모와 100% 연동되었다는 뜻이 아니다. 24시간 전체에서 위험 구간을 찾거나, SCD 확률·SHAP·QT 연장을 계산하는 기능은 포함하지 않는다.

## LLM 입력

기존 품질 인덱스에서 선택 구간이 통과한 경우에만 실행한다. 로컬 NeuroKit2 R파 검출 후 RR 기반 요약을 만들고 다음 필드만 LLM에 전송한다. 표시 수치는 예시다.

```json
{
  "segment_summary": {
    "duration_seconds": 30,
    "sample_rate_hz": 200,
    "lead": "X",
    "r_peak_count": 30,
    "rr_valid_count": 29,
    "rr_mean_ms": 1000.0,
    "rr_sd_ms": 50.0,
    "rr_rmssd_ms": 40.0,
    "mean_heart_rate_bpm": 60.0
  }
}
```

ID, SCD 정답, 환자·기록 시각, 원시 파형, 자유 텍스트는 LLM에 전송하지 않는다. 서버는 시간 위치를 별도로 보관한다. `rr_sd_ms`·`rr_rmssd_ms`는 계산할 수 없으면 null이다. 유효 구간이 부족하거나 입력 스키마가 다르면 호출 전에 중단한다. R파 검출과 정상박동 여부는 임상 검증되지 않았다.

## LLM 원본 출력

각 모델은 JSON만 출력하도록 제한하며 다른 키, 숫자 범위 오류, 비정상 종료는 실패로 처리한다.

```json
{"review_score": 50}
```

`review_score`는 0~100의 실험용 출력일 뿐 **SCD 위험 확률, 진단, 검증된 우선순위, 의료 알림 기준이 아니다.** LLM은 파형 형태나 T파 이미지를 보지 않는다. 4개 모델의 점수는 독립적으로 산출되며 같은 척도라는 근거도 없다.

## 모델 실행부가 데모에 전달할 제안 출력

원본 LLM 점수는 `model.scores[].review_priority_score`에 0~1로 정규화한다. 값이 실제 SCD 확률처럼 전달되지 않도록 데모 요구 필드는 `null`로 유지하고 이유를 전달한다.

```json
{
  "schema": "holter-demo-research-provisional-v1",
  "analysis_scope": {
    "kind": "selected_30_second_window",
    "start_seconds": 0,
    "duration_seconds": 30,
    "lead": "X",
    "sample_rate_hz": 200
  },
  "model": {
    "kind": "experimental_rr_summary_llm",
    "scores": [{"model": "qwen32", "review_priority_score": 0.5}],
    "contract_version": "holter-rr30-review-v1",
    "is_calibrated": false,
    "is_scd_probability": false
  },
  "prob": null,
  "factors": null,
  "segments": null,
  "qt_prolongation": null,
  "unavailable_reasons": {
    "prob": "No validated SCD probability model",
    "factors": "No verified SHAP or attribution",
    "segments": "No validated 24-hour dangerous-segment detector",
    "qt_prolongation": "QT candidate method not validated"
  },
  "waveform": {"sample_rate_hz": 200, "samples": "실제로는 원본 수치 6000개 배열"},
  "intended_use": "research demonstration only; not clinical care"
}
```

예시의 `samples` 문자열은 구조 설명용이며, **실제 출력은 숫자 6,000개의 배열**이다. 실제 결과에는 `features`와 `record`도 들어 있다. `null`은 검출 결과가 정상이라는 뜻이 아니라 기능이 없거나 검증되지 않았다는 뜻이다. 값 비교·알림 코드는 `prob=null`을 먼저 처리해야 한다. 실제 데모 API의 필드명과 타입이 정해지면 그 계약에 맞춘 변환 계층이 추가로 필요하다.

## 실행

```bash
cd /home/sang/sixletters-poc
Global/.venv/bin/python Members/sang/experiments/holter_viewer/demo_adapter.py \
  --patient '<환자_ID>' --start-seconds 0 --lead X \
  --output Members/sang/results/holter_viewer_30s/adapter_새_실행명
```

새 출력 폴더가 필요하다. 단계별 JSON 상태를 stdout에 기록하고 Furiosa 모델을 실제 실행한다. 원본 파형·환자 ID·SCD 정답은 LLM 요청에 보내지 않는다. `provisional_output.json`과 CSV는 로컬 환자 파형 및 민감 정보를 포함할 수 있으므로 접근을 제한한다. 유효하지 않은 구간·품질·모델 출력은 결과를 꾸며내지 않고 실패한다.

## 적용 전 남은 작업

- 실제 데모 웹 소스, `.npz` 파일의 키·shape·유도·샘플링 메타데이터와 API 응답 규격 확정.
- 24시간 전체 분석 및 위험 구간 후보의 정의·검증. 현재는 선택된 30초만 처리한다.
- SCD 예측의 기간·라벨·모델 교정·독립 검증, 요인/SHAP의 근거, QT 경계와 연장 판정 검증.
- 100Hz 합성 NPZ와 200Hz 원본 NPY의 전처리 호환성 검토. 현재 계약은 200Hz NPY에만 적용한다.

## 실제 NPU 연결 확인 (2026-11-03)

기존 원본 한 명의 X 유도 시작 후 0~30초를 선택해 실행했다. 품질 인덱스 통과 후 RR 기반 요약만 Furiosa로 전달했고, 네 모델 모두 새 `review_score` JSON 스키마에 정상 응답했다. 로컬 모델 실행부 결과에서 6,000개 원본 샘플 일치, 모델 점수 4개 범위, 식별자·정답의 LLM 입력 제외, `prob`·`factors`·`segments`·`qt_prolongation`의 null 처리를 확인했다.

- 로컬 결과: `Members/sang/results/holter_viewer_30s/adapter_contract_20261103_v2/provisional_output.json`
- LLM 출력 CSV: `Members/sang/results/holter_viewer_30s/adapter_contract_20261103_v2/model_scores.csv`
- 실제 결과 JSON의 waveform은 개인별 원본 파형이므로 비공개로 취급한다.
- 중간에 기존 원격 작업의 장치 점유로 실패한 `adapter_contract_20261103_v1`은 실패 이력으로 남겼고, 겹침이 해소된 후 v2를 별도 실행했다.

이 검증은 데모 API 연동 완료나 전체 24시간 분석이 아니다. 사용자 설명서의 백엔드 API와 실행부 코드가 제공되면, 위 초안을 실제 필드 형식에 맞추고 null 처리와 업로드·진행 이벤트를 함께 검증해야 한다.
