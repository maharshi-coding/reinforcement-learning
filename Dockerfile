FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run training
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]
