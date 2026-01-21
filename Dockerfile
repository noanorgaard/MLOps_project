FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY app.py .
COPY models ./models
COPY src ./src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
