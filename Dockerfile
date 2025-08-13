FROM python:3.11-slim

# Dependencias de sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Actualizar pip y wheel para evitar problemas de compilación
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Variables para logging en tiempo real
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=debug

# Comando de inicio con logs detallados
CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker services.recs_api:app \
 --bind 0.0.0.0:$PORT \
 --workers 1 \
 --log-level debug \
 --access-logfile - \
 --error-logfile - \
 --preload"]
