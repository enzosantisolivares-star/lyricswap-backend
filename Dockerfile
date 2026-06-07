FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools==68.2.2 wheel && \
    pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install git+https://github.com/openai/whisper.git

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
