# Baseline: slim para imagen pequeña
FROM python:3.11-slim

# dependencias del sistema necesarias para compilar paquetes (annoy) y pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libatlas-base-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia requirements primero para cachear capas
COPY requirements.txt .

# Instala paquetes
RUN pip install --no-cache-dir -r requirements.txt

# Copia código
COPY . .

# Exponer puerto (Render inyecta $PORT al run)
ENV PORT=8000

# Comando de inicio usando gunicorn + uvicorn worker (más robusto)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "services.recs_api:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--log-level", "info"]