# Cafe-Focusing — Gradio 데모/API 서버
# 실행: docker build -t cafe-focusing . && docker run -p 7860:7860 cafe-focusing
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

# opencv-python-headless 런타임 의존성
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# U2-Net 모델(~170MB)은 AI 세그멘테이션 첫 사용 시 models/ 에 자동 다운로드됨
EXPOSE 7860
CMD ["python", "app.py"]
