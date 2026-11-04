# SCD 위험 예측 모델 (로지스틱 회귀, 변수 12개)

최종 모델 `LR_core_iterative_only445` (C=0.01, class_weight=balanced).
학습 데이터: 원래 445명 코호트 중 개발용 356명 (결측값 없음)
test 89명 성능: AUROC 0.834 (0.690–0.944), AP 0.371

## 파일
| 파일 | 내용 |
|---|---|
| `model.json` | 모델 계수와 전처리 통계. 아래 `scd_model.py`와 함께 배포 |
| `scd_model.py` | 예측 코드. 순수 Python이라 numpy와 sklearn이 필요 없음 |
| `model.joblib` | `total_flow.py`가 읽는 학습된 전처리·로지스틱 회귀 파이프라인 |
| `app.py`, `Dockerfile`, `requirements.txt` | `total_flow.run_model()`을 호출하는 HTTP API 서버 |
| `handler.py`, `Dockerfile.serverless` | 같은 함수를 호출하는 RunPod Serverless용 |
| `sample_input.json` | 가상 환자 예시 (예측 확률 0.634) |

## 입력 (12개 모두 필수)

`POST /predict`의 JSON 본문으로 환자 한 명의 값을 전달합니다. 아래의 모델 변수명 또는 `total_flow.py`의 짧은 별칭을 사용할 수 있습니다. API는 입력 파일을 읽거나 만들지 않습니다.
| 컬럼명 | 단위 / 값 |
|---|---|
| `Age` | 세 |
| `Gender (male=1)` | 남 1, 여 0 |
| `NYHA class` | 1–4 |
| `LVEF (%)` | % |
| `Left atrial size (mm)` | mm |
| `eGFR` | mL/min/1.73m² |
| `Pro-BNP (ng/L)` | ng/L (모델 안에서 log1p 변환) |
| `Ischemic etiology` | 예 1, 아니오 0 |
| `Prior Myocardial Infarction (yes=1)` | 예 1, 아니오 0 |
| `Holter ventricular premature beats in 24h` | 24시간 횟수 (모델 안에서 log1p 변환) |
| `Holter ventricular couplets` | 24시간 횟수 (모델 안에서 log1p 변환) |
| `Holter NSVT (yes=1)` | 예 1, 아니오 0 |

## Docker API 출력
```json
{
  "probability": 0.6336517020627861,
  "variables": [
    {"Prior Myocardial Infarction (yes=1)": 1},
    {"Holter ventricular premature beats in 24h": 800}
  ]
}
```
- `variables` 예시는 처음 두 항목만 표시했습니다. 실제 응답에는 12개 변수가 SHAP 절댓값 내림차순으로 들어갑니다. 값은 입력받은 환자의 원래 값입니다.
- API는 `total_flow.py`의 `run_model()` 반환값을 그대로 전달합니다. 반환 형태를 바꾸려면 그 함수만 수정하면 됩니다.
- `probability`는 class_weight=balanced로 학습한 보정되지 않은 모델 점수이므로 실제 발생 위험률로 해석하지 않습니다.

## 로컬에서 함수 호출
```python
import json
from model_total_process.total_flow import run_model

with open("sample_input.json", encoding="utf-8") as file:
    patient = json.load(file)
result = run_model(patient)
print(result)
```

## Docker HTTP API

저장소 루트에서 `Members/minjeong/notebooks`를 빌드 컨텍스트로 지정합니다.

```bash
docker build -f Members/minjeong/notebooks/model_ml/Dockerfile -t scd-model Members/minjeong/notebooks
docker run --rm -p 8080:8080 scd-model
```

다른 터미널에서:

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  --data-binary @Members/minjeong/notebooks/model_ml/sample_input.json
```

`GET /health`와 `GET /features`로 서버 상태와 필요한 입력 변수명을 확인할 수 있습니다.

## RunPod 배포 (GPU 불필요, CPU로 충분)

### A. Serverless (요청 올 때만 실행, 추천)
1. 저장소 루트에서 이미지 빌드·푸시: `docker build -f Members/minjeong/notebooks/model_ml/Dockerfile.serverless -t <dockerhub계정>/scd-lr-core:1 Members/minjeong/notebooks && docker push <dockerhub계정>/scd-lr-core:1`
2. RunPod 콘솔 → Serverless → New Endpoint → 위 이미지 지정 (CPU worker)
3. 호출:
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" -H "Content-Type: application/json" \
  -d '{"input": {"Age": 65, "Gender (male=1)": 1, ... 12개}}'
```
응답 `{"output": {"probability": ..., "variables": [...]}}`. 입력 오류 시 `{"error": "Missing inputs: [...]"}`.
RunPod API 키가 필요하므로 브라우저가 아니라 데모 백엔드에서 호출할 것.

### B. Pod (항상 켜진 서버)
Pod 생성 시 위 `Dockerfile` 빌드 이미지를 사용합니다.
Pod 설정에서 HTTP 포트 8080을 노출하면 `https://<POD_ID>-8080.proxy.runpod.net/predict` 로 호출.

### C. 독립형 순수 Python 모델
`scd_model.py`와 `model.json`은 기존 순수 Python 구현입니다. 현재 Docker API는 이 파일들을 사용하지 않고 `total_flow.py`와 `model.joblib`을 사용합니다.

## 다시 만들기
레포 루트에서 `Global/.venv/bin/python Members/mingyu/deploy/build_scd_lr_core.py`를 실행하세요.
원래 `final_model.joblib`에서 다시 추출하고, test 89명 확률이 같은지(오차 1e-9 이하) 확인해요.
