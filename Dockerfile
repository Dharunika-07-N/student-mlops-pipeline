FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY model ./model

CMD ["python", "src/predict.py"]