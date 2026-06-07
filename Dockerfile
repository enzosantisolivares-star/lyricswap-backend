FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar torch CPU primero (wheels precompilados)
RUN pip install --upgrade pip setuptools==68.2.2 wheel && \
    pip install torch==2.3.0+cpu torchaudio==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar whisper desde GitHub (evita el problema de build wheel)
RUN pip install git+https://github.com/openai/whisper.git

# Copiar e instalar el resto de dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
