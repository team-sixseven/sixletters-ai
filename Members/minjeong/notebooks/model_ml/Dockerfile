FROM python:3.12-slim
WORKDIR /app
COPY model_ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_ml/app.py ./
COPY model_ml/model.joblib model_ml/model.joblib
COPY model_total_process/total_flow.py model_total_process/total_flow.py
ENV PORT=8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
